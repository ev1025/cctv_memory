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

import io
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


@app.get("/")
def ui_root():
    return FileResponse(os.path.join(_UI_DIST, "index.html"))


@app.get("/video/{video_id}")
def serve_video(video_id: str):
    videos = os.path.join(config.MEMORY_DIR, "videos")
    for ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        p = os.path.join(videos, video_id + ext)
        if os.path.exists(p):
            return FileResponse(p)
    raise HTTPException(404, "video not found")


@app.get("/thumb")
def serve_thumb(path: str):
    rp = os.path.abspath(path)              # MEMORY_DIR 하위만 허용(경로 탈출 방지)
    if not rp.startswith(os.path.abspath(config.MEMORY_DIR)) or not os.path.exists(rp):
        raise HTTPException(404, "thumb not found")
    return FileResponse(rp)


@app.get("/memory-status")
def memory_status():
    from memory.vector_store import VectorStore
    from memory.text_embedder import TextEmbedder
    try:
        return {"count": VectorStore(TextEmbedder()).count()}
    except Exception:
        return {"count": 0}


@app.get("/videos")
def list_videos_ep():
    """색인된 영상 라이브러리 목록."""
    from memory.vector_store import VectorStore
    from memory.text_embedder import TextEmbedder
    return {"videos": VectorStore(TextEmbedder()).list_videos()}


@app.get("/segments")
def segments_ep(video_id: str):
    """영상의 전체 구간(타임라인)."""
    from memory.vector_store import VectorStore
    from memory.text_embedder import TextEmbedder
    return {"video_id": video_id, "segments": VectorStore(TextEmbedder()).get_segments(video_id)}


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
             fire_only: bool = Form(False), backend: str = Form(None)):
    """자연어 질문 → 벡터검색 + RAG 답변. 반환: {answer, segments[]}."""
    from memory import video_memory
    where = {"fire": True} if fire_only else None
    with _lock:
        return video_memory.query(question, k=k, where=where, vlm_backend=backend)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
