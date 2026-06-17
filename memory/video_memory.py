"""memory/video_memory.py — 오케스트레이터 (index / query).

index_video(): mp4 → segmenter → VLM 행동 캡션 → parse_event → ChromaDB upsert(+썸네일).
query():       질문 → 벡터검색(+메타필터) → 대표 프레임+캡션 → VLM RAG 답변(근거 인용).

CLI:
  python -m memory.video_memory index <mp4> [video_id]
  python -m memory.video_memory query "<질문>" [--type fight] [--k 5]
"""
import config  # ★ torch 보다 먼저

import os
import re
import sys
from datetime import datetime

from PIL import Image

from image_to_text import VLMCaptioner
from memory.vector_store import default_store
from memory import segmenter

# VLM 슬롯 캐시 — query/index 간 재사용(매 호출 ~1.5분 재로드 방지). 같은 프로세스에 상주.
_VLM_SLOT = {"backend": None, "obj": None}


def _get_vlm(backend):
    if _VLM_SLOT["backend"] != backend:
        if _VLM_SLOT["obj"] is not None:
            _VLM_SLOT["obj"].unload()
        _VLM_SLOT["obj"] = VLMCaptioner(backend).load()
        _VLM_SLOT["backend"] = backend
    return _VLM_SLOT["obj"]


def parse_event(text):
    """SEGMENT_EVENT_PROMPT 응답 → 캡션 문자열.

    프롬프트는 '캡션: …' 한 줄 형식(소형 모델엔 JSON 보다 안정적 — hailo 교훈).
    유형(event_type)은 색인과 분리된 임베딩 분류기(classifier)가 reclassify 에서 별도로 붙인다.
    """
    t = text or ""
    m = re.search(r"캡션\s*[:：]\s*(.+)", t)
    if m:
        caption = m.group(1)
    else:                                            # 형식 이탈 시 가장 의미있는 한 줄
        caption = next((l for l in t.splitlines()
                        if len(re.sub(r"[^가-힣A-Za-z]", "", l)) >= 6), "")
    caption = re.split(r"\s*(?:활동|유형)\s*[:：]", str(caption))[0]   # 옛 형식 꼬리표 제거
    caption = re.sub(r'[*#`{}"]+', "", caption).strip(" ,.\t")[:200]
    return caption or "특이 행동 감지"


def _video_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def _fmt_ts(s):
    s = int(s)
    return f"{s // 60:02d}:{s % 60:02d}"


def _resolve_thumb(p):
    """저장된 thumb 절대경로가 옛 위치(예: 3d_vision)를 가리켜도 현재 MEMORY_DIR/thumbs 로 재매핑."""
    if not p:
        return p
    q = p.replace("\\", "/")
    return os.path.join(config.MEMORY_DIR, "thumbs", q.split("thumbs/", 1)[1]) if "thumbs/" in q else p


def index_video(video_path, video_id=None, vlm_backend=None):
    """mp4 색인 → ChromaDB. 5초 구간 캡션을 '구간 1개 = 레코드 1줄'로 저장(+썸네일).

    흐름:
      ① segmenter.segment    : 5초 그리드 구간
      ② VLM(SEGMENT_EVENT_PROMPT)+parse_event : 구간별 행동 캡션(원문 그대로)
      ③ store.add            : 구간 캡션 저장. 유형(event_type)은 reclassify(분류기)가 별도로 채움.
    재색인 멱등(같은 video_id:s{i} upsert).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    video_id = video_id or _video_id(video_path)
    vlm_backend = vlm_backend or config.VLM_BACKEND

    # ① 구간 분할
    segs = segmenter.segment(video_path)
    if not segs:
        print("[index] 구간 없음(코덱 미지원?)", flush=True)
        return {"video_id": video_id, "segments": 0}
    _max = int(os.environ.get("MAX_SEGMENTS", "0"))
    if _max > 0 and len(segs) > _max:
        segs = segs[:_max]                      # 긴 영상 색인 시간 제한(데모)

    # ② 구간별 VLM 행동 캡션 → ③ 썸네일 + 레코드
    vlm = _get_vlm(vlm_backend)
    store = default_store()
    thumbs_dir = os.path.join(config.MEMORY_DIR, config.THUMBS_SUBDIR, video_id)
    os.makedirs(thumbs_dir, exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")

    records = []
    for i, seg in enumerate(segs):
        raw = vlm.caption_frames(seg.frames, config.SEGMENT_EVENT_PROMPT)
        caption = parse_event(raw)
        thumb = ""
        if seg.frames:
            thumb = os.path.join(thumbs_dir, f"s{i}.jpg")
            seg.frames[len(seg.frames) // 2].save(thumb, quality=85)
        records.append({"id": f"{video_id}:s{i}", "document": caption, "metadata": {
            "video_id": video_id, "source": os.path.basename(video_path),
            "start_s": seg.start_s, "end_s": seg.end_s, "event_type": "unknown",
            "thumb": thumb, "embed_model": config.EMBED_BACKEND,
            "vlm_backend": vlm_backend, "indexed_at": stamp,
        }})
        print(f"  seg {i:2d} [{_fmt_ts(seg.start_s)}-{_fmt_ts(seg.end_s)}] · {caption[:40]}", flush=True)

    n = store.add(records)
    print(f"[index] {video_id}: 구간 {n} 색인 (총 {store.count()})", flush=True)
    return {"video_id": video_id, "segments": len(segs)}


def query(question, k=5, where=None, vlm_backend=None, answer=True):
    """질문 → 구간 벡터검색 → 시간 근접 중복 제거 → top-k 반환. 반환: {answer, segments[]}.

    구간 단위로 매칭(의미 희석 없음). 같은 영상에서 시간이 가까운(같은 상황) 구간은 최고점만 남겨
    중복을 막는다(시간근접 dedup). start_s = 정확 seek.
    """
    store = default_store()
    hits = store.search(question, k=max(k * 6, 18), where=where)        # dedup 로 줄어드니 넉넉히 뽑음
    hits = [h for h in hits if h["score"] >= config.SEARCH_MIN_SCORE]   # 유사도 임계 미만 제외(무관 질의 차단)
    if not hits:
        return {"answer": "색인된 구간이 없거나 검색 결과가 없습니다.", "segments": []}

    # 점수 높은 순으로 구간 직접 선택 — 같은 영상에서 SEARCH_DEDUP_GAP_S 초 이내 구간은 1개만(인접 중복 제거)
    hits.sort(key=lambda h: h["score"], reverse=True)
    gap = config.SEARCH_DEDUP_GAP_S
    top = []
    for h in hits:
        vid, st = h["metadata"].get("video_id"), float(h["metadata"].get("start_s") or 0)
        if any(p["metadata"].get("video_id") == vid
               and abs(float(p["metadata"].get("start_s") or 0) - st) <= gap for p in top):
            continue
        top.append(h)
        if len(top) >= k:
            break

    segments = [{
        "video_id": h["metadata"].get("video_id"),
        "start_s": h["metadata"]["start_s"], "end_s": h["metadata"]["end_s"],   # 대표 구간 = 정확 seek
        "caption": h["document"], "thumb": h["metadata"].get("thumb"),
        "score": h["score"], "event_type": h["metadata"].get("event_type"),
    } for h in top]
    if not answer:
        return {"answer": None, "segments": segments}

    ctx = "\n".join(f"[{_fmt_ts(s['start_s'])}] {s['caption']}" for s in segments)
    frames = [Image.open(_resolve_thumb(s["thumb"])).convert("RGB")
              for s in segments if _resolve_thumb(s["thumb"]) and os.path.exists(_resolve_thumb(s["thumb"]))]
    prompt = config.RAG_ANSWER_PROMPT.format(question=question, context=ctx)
    if frames:
        vlm = _get_vlm(vlm_backend or config.VLM_BACKEND)
        ans = vlm.caption_frames(frames, prompt)
    else:
        ans = "검색된 구간:\n" + ctx   # 프레임 없으면 캡션 컨텍스트만 반환(폴백)
    return {"answer": ans, "segments": segments}


def _main():
    if len(sys.argv) < 3:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == "index":
        vid = sys.argv[3] if len(sys.argv) > 3 else None
        print(index_video(sys.argv[2], vid))
    elif cmd == "query":
        where = None
        if "--type" in sys.argv:                                  # 예: --type fight / falldown
            where = {"event_type": sys.argv[sys.argv.index("--type") + 1]}
        k = 5
        if "--k" in sys.argv:
            k = int(sys.argv[sys.argv.index("--k") + 1])
        out = query(sys.argv[2], k=k, where=where)
        print("\n=== 답변 ===\n" + str(out["answer"]))
        print("\n=== 근거 구간 ===")
        for s in out["segments"]:
            print(f"[{_fmt_ts(s['start_s'])}-{_fmt_ts(s['end_s'])}] "
                  f"score={s['score']} {s['event_type']} · {s['caption'][:40]}")
    else:
        print(__doc__)


if __name__ == "__main__":
    _main()
