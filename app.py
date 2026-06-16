"""
app.py — FastAPI: Image→Text 감시 API

  POST /image-to-text : 이미지 → 텍스트 (VLM)
  POST /video-to-text : 영상 → 프레임 추출 → VLM (multi-image 시계열)
  GET  /health · GET /models

[격리] Text→Image(text-to-image, img2img)는 t2i/ 로 분리(보류). 이 API 는 Image→Text 감시 라인만.
[모델] VLM lazy 로드 + 슬롯 캐시(1개 상주). GPU 는 _lock 으로 직렬 사용.
[실행] uvicorn app:app --host 0.0.0.0 --port 8000   (로컬: run_local.bat = 4bit)
"""
import config  # ★ torch 보다 먼저 (GPU 격리/HF 캐시)

import os
import threading

import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import models
from image_to_text import VLMCaptioner

app = FastAPI(title="3D-Vision Image→Text API", version="2.0")

_lock = threading.Lock()               # GPU 직렬 사용
_vlm = {"name": None, "obj": None}


def _get_vlm(name):
    if _vlm["name"] != name:
        if _vlm["obj"] is not None:
            _vlm["obj"].unload()
        _vlm["obj"] = VLMCaptioner(name).load()
        _vlm["name"] = name
    return _vlm["obj"]


def _store():
    """프로세스 공유 단일 VectorStore(임베더 1회 로드) — 조회/검색 매 요청 재로드 방지."""
    from memory.vector_store import default_store
    return default_store()


def _video_frames(file: UploadFile, n):
    """업로드 영상 → 균등 n 장 프레임을 PIL 로 추출."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    tmp = os.path.join(config.OUTPUT_DIR, "_upload" + os.path.splitext(file.filename or ".mp4")[1])
    with open(tmp, "wb") as f:
        f.write(file.file.read())
    cap = cv2.VideoCapture(tmp)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    raw = []
    if total <= 0:
        ok, fr = cap.read()
        if ok:
            raw.append(fr)
    else:
        idxs = [max(0, int(total * (i + 1) / (n + 1))) for i in range(n)]
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            if ok:
                raw.append(fr)
    cap.release()
    return [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in raw]


@app.get("/health")
def health():
    return {"status": "ok", "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "vlm_loaded": _vlm["name"]}


@app.get("/models")
def list_models():
    return {"vlm": list(models.VLM_REGISTRY)}


@app.post("/image-to-text")
def image_to_text_ep(file: UploadFile = File(...),
                     backend: str = Form("qwen2-vl"),
                     prompt: str = Form(None)):
    if backend not in models.VLM_REGISTRY:
        raise HTTPException(400, f"unknown vlm '{backend}'. available: {list(models.VLM_REGISTRY)}")
    img = Image.open(file.file).convert("RGB")
    with _lock:
        text = _get_vlm(backend).caption(img, prompt or None)
    return {"backend": backend, "text": text}


@app.post("/video-to-text")
def video_to_text_ep(file: UploadFile = File(...),
                     backend: str = Form("qwen2-vl"),
                     prompt: str = Form(None),
                     num_frames: int = Form(3),
                     multi_image: bool = Form(True)):
    if backend not in models.VLM_REGISTRY:
        raise HTTPException(400, f"unknown vlm '{backend}'. available: {list(models.VLM_REGISTRY)}")
    frames = _video_frames(file, max(1, num_frames))
    if not frames:
        raise HTTPException(400, "프레임 추출 실패(영상 코덱 미지원일 수 있음)")
    with _lock:
        cap = _get_vlm(backend)
        if multi_image:
            text = cap.caption_frames(frames, prompt or None)
            return {"backend": backend, "frames": len(frames), "mode": "multi-image", "text": text}
        texts = [cap.caption(f, prompt or None) for f in frames]
    return {"backend": backend, "frames": len(frames), "mode": "per-frame", "texts": texts}


# ── video-memory: UI 서빙 + 영상 이력 색인(RAG) ──────────────────────────────
_UI_DIST = os.path.join(config.BASE_DIR, "ui-dist")
if os.path.isdir(os.path.join(_UI_DIST, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(_UI_DIST, "assets")), name="assets")


@app.on_event("startup")
def _warmup_embedder():
    """서버 시작 시 임베더(bge-m3)를 백그라운드로 미리 로드 → 첫 검색도 빠르게(블로킹 없음)."""
    def go():
        try:
            _store().search("warmup", k=1)
            print("[warmup] 임베더 준비 완료 — 검색 즉시 응답", flush=True)
        except Exception as e:
            print(f"[warmup] skip: {e}", flush=True)
    threading.Thread(target=go, daemon=True).start()


@app.get("/")
def ui_root():
    # no-cache: 재빌드 때마다 자산 해시가 바뀌므로 index.html 은 캐시하지 않아야 옛 자산 404(백지)를 막는다.
    return FileResponse(os.path.join(_UI_DIST, "index.html"),
                        headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/video/{video_id}")
def serve_video(video_id: str, type: str = "normal"):
    """일반/열화상 영상 서빙. type=thermal 이면 '{id}__thermal.*' 우선, 없으면 일반으로 폴백.

    (열화상 실데이터가 없을 때도 PIP 가 항상 렌더되도록 폴백 — 프론트에서 CSS 필터로 열화상처럼 표시.)
    """
    videos = os.path.join(config.MEMORY_DIR, "videos")
    names = [video_id + "__thermal", video_id] if type == "thermal" else [video_id]
    for name in names:
        for ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
            p = os.path.join(videos, name + ext)
            if os.path.exists(p):
                return FileResponse(p)
    raise HTTPException(404, "video not found")


@app.get("/thumb")
def serve_thumb(path: str):
    # 저장된 절대경로(다른 서버/옛 위치/모델별 thumbs_xxx)를 현재 MEMORY_DIR 하위로 재매핑.
    p = path.replace("\\", "/")
    if "/vmem/" in p:                                               # 서버 절대경로 → 로컬 vmem (thumbs_internvl3 등 포함)
        cand = os.path.join(config.MEMORY_DIR, p.split("/vmem/", 1)[1])
    elif "thumbs/" in p:
        cand = os.path.join(config.MEMORY_DIR, "thumbs", p.split("thumbs/", 1)[1])
    else:
        cand = path
    rp = os.path.abspath(cand)              # MEMORY_DIR 하위만 허용(경로 탈출 방지)
    if not rp.startswith(os.path.abspath(config.MEMORY_DIR)) or not os.path.exists(rp):
        raise HTTPException(404, "thumb not found")
    return FileResponse(rp)


@app.get("/compare")
def compare_page():
    """VLM 캡션 비교 페이지(영상 + 3모델 구간별 캡션). 16프레임 색인 결과 검토용."""
    return FileResponse(os.path.join(config.BASE_DIR, "compare.html"),
                        headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/compare-data")
def compare_data():
    for name in ("captions_v16.json", "captions_f16.json"):   # 16프레임 video(v16) 우선, 없으면 f16
        p = os.path.join(config.MEMORY_DIR, name)
        if os.path.exists(p):
            return FileResponse(p)
    raise HTTPException(404, "captions_*.json 없음 — dump_captions_json.py 먼저 실행")


@app.get("/memory-status")
def memory_status():
    try:
        return {"count": _store().count()}
    except Exception:
        return {"count": 0}


@app.get("/videos")
def list_videos_ep():
    """색인된 영상 라이브러리 목록(+카메라/날짜 메타)."""
    import cctv_meta
    return {"videos": cctv_meta.enrich_videos(_store().list_videos())}


@app.get("/cameras")
def cameras_ep():
    """모든 CCTV 카메라(영상 파일 기준) + 가용 일자. 특이사항 0건 카메라도 그리드엔 표시(이력만 빔)."""
    import cctv_meta
    import glob
    vdir = os.path.join(config.MEMORY_DIR, "videos")
    vids = []
    for p in sorted(glob.glob(os.path.join(vdir, "*"))):
        name, ext = os.path.splitext(os.path.basename(p))
        if ext.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm") and "__thermal" not in name:
            vids.append({"video_id": name, "segments": 0})
    cams, dates = cctv_meta.list_cameras(vids)
    return {"cameras": cams, "dates": dates}


@app.get("/alerts")
def alerts_ep(date: str = None, cams: str = None):
    """열화상 고온 알림(온도 ≥ 주의). date·cams(콤마구분) 로 필터, 온도 높은 순."""
    import cctv_meta
    cam_set = {c.strip() for c in cams.split(",") if c.strip()} if cams else None
    return {"alerts": cctv_meta.build_alerts(_store().all_segments(), date=date or None, cams=cam_set)}


@app.get("/history")
def history_ep(date: str = None, cams: str = None):
    """전체 구간 이력(평소 보는 단일 타임라인). 고온 구간은 is_alert=True 로 마킹."""
    import cctv_meta
    cam_set = {c.strip() for c in cams.split(",") if c.strip()} if cams else None
    return {"segments": cctv_meta.build_history(_store().raw_segments(), date=date or None, cams=cam_set)}


@app.get("/segments")
def segments_ep(video_id: str):
    """영상의 전체 구간(타임라인) + 카메라/날짜 메타 + 구간별 온도/등급."""
    import cctv_meta
    meta = cctv_meta.video_meta(video_id)
    segs = _store().get_segments(video_id)
    for s in segs:
        s["video_id"] = video_id                       # 열화상 훅(_real_temp) 조회 키
        s["camera_id"] = meta["camera_id"]
        s["temp"], s["level"] = cctv_meta.seg_temp(s)
    return {"video_id": video_id, **meta, "segments": segs}


@app.post("/index-video")
def index_video_ep(file: UploadFile = File(...),
                   video_id: str = Form(None), backend: str = Form(None)):
    """mp4 업로드 → 영상 영구 저장(player용) + 구간 색인(ChromaDB). 반환: {video_id, segments}."""
    from memory import video_memory
    vid = video_id or os.path.splitext(os.path.basename(file.filename or "video"))[0]
    videos = os.path.join(config.MEMORY_DIR, "videos")
    os.makedirs(videos, exist_ok=True)
    path = os.path.join(videos, vid + os.path.splitext(file.filename or ".mp4")[1])
    with open(path, "wb") as f:
        f.write(file.file.read())
    with _lock:                      # VLM 직렬 사용
        return video_memory.index_video(path, video_id=vid, vlm_backend=backend)


@app.post("/query")
def query_ep(question: str = Form(...), k: int = Form(5),
             special_only: bool = Form(False), event_type: str = Form(None),
             camera: str = Form(None), backend: str = Form(None)):
    """자연어 질문 → 벡터검색(retrieval-only, 빠름). 반환: {segments[]} (사건에 카메라/온도 부착).

    special_only=True → 특이사건(severity≥1)만. event_type 지정 시 해당 유형만(예: fall, loitering).
    [속도] 임베더는 _store() 로 1회 로드 캐시. VLM RAG 답변은 생략(검색은 즉시 결과 반환).
    """
    from memory import video_memory
    import cctv_meta
    where = None
    if event_type:
        where = {"event_type": event_type}
    elif special_only:
        where = {"severity": {"$gte": 1}}
    out = video_memory.query(question, k=k, where=where, answer=False)   # VLM 생략 → 빠름
    seen, deduped = set(), []
    for s in out.get("segments", []):
        vid = s.get("video_id")
        key = s.get("event_id") or f"{vid}:{int(s.get('start_s') or 0)}"
        if key in seen:                                  # 같은 사건 1회만(클립이 여러 카메라에 펼쳐져도)
            continue
        seen.add(key)
        if camera:                                        # 포커스: 그 카메라 배치로 한정
            off = next((o for c, o in cctv_meta._clip_index().get(vid, []) if c == camera), None)
            if off is None:
                continue                                  # 그 카메라에 없는 사건 제외
            s["camera_id"] = camera
            s["abs_s"] = int(off + (s.get("start_s") or 0))
        else:                                             # 그리드: 대표 카메라 1개 귀속
            place = cctv_meta.primary_placement(vid)
            if place:
                s["camera_id"] = place[0]
                s["abs_s"] = int(place[1] + (s.get("start_s") or 0))
            else:
                s["camera_id"] = cctv_meta.video_meta(vid).get("camera_id")
        s["temp"], s["level"] = cctv_meta.seg_temp(s)
        deduped.append(s)
    out["segments"] = deduped
    return out


@app.get("/timeline")
def timeline_ep():
    """9카메라 24h 플레이리스트(시각→클립 표). 프론트 PlaylistPlayer 가 사용."""
    import cctv_meta
    tl = cctv_meta._timeline() or {}
    return {"day_seconds": tl.get("day_seconds", 86400), "cameras": tl.get("cameras", [])}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
