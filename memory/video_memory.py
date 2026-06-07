"""memory/video_memory.py — 오케스트레이터 (index / query).

index_video(): mp4 → segmenter → VLM 통합 캡션 → parse_risk → ChromaDB upsert(+썸네일).
query():       질문 → 벡터검색(+메타필터) → 대표 프레임+캡션 → VLM RAG 답변(근거 인용).

CLI:
  python -m memory.video_memory index <mp4> [video_id]
  python -m memory.video_memory query "<질문>" [--fire] [--k 5]
"""
import config  # ★ torch 보다 먼저

import os
import re
import sys
from datetime import datetime

from PIL import Image

from image_to_text import VLMCaptioner
from memory.text_embedder import TextEmbedder
from memory.vector_store import VectorStore
from memory import segmenter

_VALID_TYPES = ("fire", "smoke", "fall", "machine", "none")

# VLM 슬롯 캐시 — query/index 간 재사용(매 호출 ~1.5분 재로드 방지). 같은 프로세스에 상주.
_VLM_SLOT = {"backend": None, "obj": None}


def _get_vlm(backend):
    if _VLM_SLOT["backend"] != backend:
        if _VLM_SLOT["obj"] is not None:
            _VLM_SLOT["obj"].unload()
        _VLM_SLOT["obj"] = VLMCaptioner(backend).load()
        _VLM_SLOT["backend"] = backend
    return _VLM_SLOT["obj"]


def parse_risk(text):
    """SEGMENT_RISK_PROMPT 응답 → (caption, label).

    4bit VLM 이 형식을 안 지키고 번호·마크다운·영어로 답하는 경우까지 강건하게 파싱한다.
    설명 추출 우선순위: '설명:' → 'Description:' → 라벨/번호/시간이 아닌 첫 긴 줄.
    """
    # ── 설명(캡션) ──
    m = re.search(r"설명\s*[:：]\s*(.+)", text)
    if not m:
        m = re.search(r"[Dd]escription\s*[:：*]*\s*(.+)", text)
    if m:
        caption = m.group(1)
    else:
        caption = None
        for line in text.splitlines():
            l = line.strip().lstrip("0123456789.-*#) ").strip()
            if len(l) >= 8 and not re.match(r"(위험|유형|심각도|risk|type|severity|시간|time)", l, re.I):
                caption = l
                break
        caption = caption or text
    caption = re.sub(r"[*#`]+", "", caption.strip())[:200]

    # ── 위험: '없음' 명시 우선 ──
    risk = None
    rm = re.search(r"위험\s*[:：]?\s*(있음|없음|있|없)", text)
    if rm:
        risk = rm.group(1).startswith("있")

    # ── 유형: '유형:' 줄 우선, 없으면 본문 키워드 ──
    rtype = "none"
    tm = re.search(r"유형\s*[:：]?\s*([A-Za-z]+)", text)
    if tm and tm.group(1).lower() in _VALID_TYPES:
        rtype = tm.group(1).lower()
    else:
        for t in ("fire", "smoke", "fall", "machine"):
            if re.search(rf"\b{t}\b", text, re.I):
                rtype = t
                break

    # 위험 미표시 + 유형 있으면 위험으로 간주(일관성)
    if risk is None:
        risk = rtype != "none"

    sm = re.search(r"심각도\s*[:：]?\s*([0-3])", text)
    sev = int(sm.group(1)) if sm else (2 if risk else 0)
    return caption, {"fire": rtype == "fire", "risk": bool(risk), "risk_type": rtype, "severity": sev}


def _video_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def _fmt_ts(s):
    s = int(s)
    return f"{s // 60:02d}:{s % 60:02d}"


def index_video(video_path, video_id=None, vlm_backend=None):
    """mp4 색인 → ChromaDB. 같은 video_id 재색인은 멱등(id 동일 upsert)."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    video_id = video_id or _video_id(video_path)
    vlm_backend = vlm_backend or config.VLM_BACKEND

    segs = segmenter.segment(video_path)
    if not segs:
        print("[index] 구간 없음(코덱 미지원?)", flush=True)
        return {"video_id": video_id, "segments": 0}

    vlm = _get_vlm(vlm_backend)
    store = VectorStore(TextEmbedder())
    thumbs_dir = os.path.join(config.MEMORY_DIR, "thumbs", video_id)
    os.makedirs(thumbs_dir, exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")

    records = []
    for i, seg in enumerate(segs):
        raw = vlm.caption_frames(seg.frames, config.SEGMENT_RISK_PROMPT)
        caption, lab = parse_risk(raw)
        thumb = os.path.join(thumbs_dir, f"seg_{i}.jpg")
        seg.frames[len(seg.frames) // 2].save(thumb, quality=85)
        records.append({
            "id": f"{video_id}:{i}",
            "document": caption or raw[:200],
            "metadata": {
                "video_id": video_id, "source": os.path.basename(video_path),
                "start_s": seg.start_s, "end_s": seg.end_s,
                "trigger": seg.trigger, "person_count": seg.person_count,
                "fire": lab["fire"], "risk": lab["risk"],
                "risk_type": lab["risk_type"], "severity": lab["severity"],
                "thumb": thumb, "embed_model": config.EMBED_BACKEND,
                "vlm_backend": vlm_backend, "indexed_at": stamp,
            },
        })
        print(f"  seg {i} [{_fmt_ts(seg.start_s)}-{_fmt_ts(seg.end_s)}] "
              f"{seg.trigger}/{seg.person_count}명 · {lab['risk_type']}(sev{lab['severity']}) · {caption[:30]}", flush=True)
    n = store.add(records)
    print(f"[index] {video_id}: {n} 구간 색인 완료 (총 {store.count()} 레코드)", flush=True)
    return {"video_id": video_id, "segments": n}


def query(question, k=5, where=None, vlm_backend=None, answer=True):
    """질문 → 벡터검색 → (선택) VLM RAG 답변. 반환: {answer, segments[]}."""
    store = VectorStore(TextEmbedder())
    hits = store.search(question, k=k, where=where)
    if not hits:
        return {"answer": "색인된 구간이 없거나 검색 결과가 없습니다.", "segments": []}

    segments = [{
        "video_id": h["metadata"].get("video_id"),
        "start_s": h["metadata"]["start_s"], "end_s": h["metadata"]["end_s"],
        "caption": h["document"], "thumb": h["metadata"].get("thumb"),
        "score": h["score"], "risk_type": h["metadata"].get("risk_type"),
        "severity": h["metadata"].get("severity"),
    } for h in hits]
    if not answer:
        return {"answer": None, "segments": segments}

    ctx = "\n".join(f"[{_fmt_ts(s['start_s'])}] {s['caption']}" for s in segments)
    frames = [Image.open(s["thumb"]).convert("RGB")
              for s in segments if s["thumb"] and os.path.exists(s["thumb"])]
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
        where = {"fire": True} if "--fire" in sys.argv else None
        k = 5
        if "--k" in sys.argv:
            k = int(sys.argv[sys.argv.index("--k") + 1])
        out = query(sys.argv[2], k=k, where=where)
        print("\n=== 답변 ===\n" + str(out["answer"]))
        print("\n=== 근거 구간 ===")
        for s in out["segments"]:
            print(f"[{_fmt_ts(s['start_s'])}-{_fmt_ts(s['end_s'])}] "
                  f"score={s['score']} {s['risk_type']} · {s['caption'][:40]}")
    else:
        print(__doc__)


if __name__ == "__main__":
    _main()
