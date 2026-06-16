"""memory/video_memory.py — 오케스트레이터 (index / query).

index_video(): mp4 → segmenter → VLM 행동 캡션 → parse_event → ChromaDB upsert(+썸네일).
query():       질문 → 벡터검색(+메타필터) → 대표 프레임+캡션 → VLM RAG 답변(근거 인용).

CLI:
  python -m memory.video_memory index <mp4> [video_id]
  python -m memory.video_memory query "<질문>" [--fire] [--k 5]
"""
import config  # ★ torch 보다 먼저

import json
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


# ── 행동 이벤트(SEGMENT_EVENT_PROMPT) 파싱 — JSON 우선 + 휴리스틱 폴백 ──────────
_EVENT_KEYWORDS = (
    ("fall",                ("쓰러", "넘어", "낙상", "fall", "collapse", "lying")),
    ("smoking",             ("담배", "흡연", "smok", "cigarette")),
    ("flammable",           ("인화", "기름", "유출", "쏟", "spill", "flammable", "gasoline", "fuel")),
    ("vehicle_interaction", ("트렁크", "보닛", "승하차", "차량", "차문", "trunk", "hood", "vehicle", "car door", "boarding")),
)


def _extract_json(text):
    """VLM 출력에서 첫 번째 균형 잡힌 JSON 객체를 관대하게 추출(코드펜스·잡텍스트 허용)."""
    if not text:
        return None
    t = re.sub(r"```(?:json)?|```", "", text, flags=re.I).strip()
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                blob = t[start:i + 1]
                for cand in (blob, re.sub(r",\s*([}\]])", r"\1", blob)):  # 후행 콤마 보정 재시도
                    try:
                        return json.loads(cand)
                    except json.JSONDecodeError:
                        continue
                return None
    return None


def _event_fallback(text):
    """JSON 파싱 실패 시 키워드 휴리스틱. VLM 이 무언가 말했으면 활동으로 본다."""
    t = text or ""
    et = "unknown"
    for name, kws in _EVENT_KEYWORDS:
        if any(re.search(k, t, re.I) for k in kws):
            et = name
            break
    line = next((l.strip(" -*#`\t") for l in t.splitlines()
                 if len(re.sub(r"[^가-힣A-Za-z]", "", l)) >= 6), "")
    caption = re.sub(r"[*#`]+", "", line).strip()[:200] or "특이 행동 감지"
    return caption, {"activity": bool(t.strip()), "event_type": et, "objects": [],
                     "severity": 2 if et not in ("normal", "unknown") else 0}


def parse_event(text):
    """SEGMENT_EVENT_PROMPT 응답 → (caption, label).

    현재 프롬프트는 '캡션: … / 활동: 있음' 줄 형식(소형 모델엔 JSON 보다 안정적 — hailo 교훈).
    옛 JSON 형식도 폴백 파싱. event_type/severity 는 분류기(classifier)가 별도로 채우므로 여기선 unknown/0.
    """
    t = text or ""
    m = re.search(r"캡션\s*[:：]\s*(.+)", t)
    caption = re.split(r"활동\s*[:：]", m.group(1))[0] if m else None     # 같은 줄에 '활동:' 붙으면 잘라냄
    am = re.search(r"활동\s*[:：]?\s*(있음|없음|있|없|true|false|yes|no)", t, re.I)
    activity = am.group(1).lower().startswith(("있", "t", "y")) if am else None

    if caption is None:                                                  # JSON 폴백(옛 형식 호환)
        obj = _extract_json(t)
        if obj is not None:
            caption = obj.get("caption")
            if activity is None and isinstance(obj.get("activity"), bool):
                activity = obj.get("activity")
    if caption is None:                                                  # 휴리스틱 폴백
        return _event_fallback(t)

    caption = re.sub(r'[*#`{}"]+', "", str(caption)).strip(" ,.\t")[:200]
    if activity is None:
        activity = bool(caption)
    if not caption:
        caption = "특이 행동 감지"
    return caption, {"activity": bool(activity), "event_type": "unknown", "objects": [], "severity": 0}


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
    """mp4 색인 → ChromaDB. 추적→VLM 행동캡션→사건 병합→'사건' 단위 저장.

    재색인 멱등(event_id 동일 upsert). 흐름:
      ① tracker.track_video  : 사람·차량 ID 타임라인(체류·이동) — VLM 전, 8GB 위해 먼저
      ② segmenter.segment    : 5초 그리드 구간
      ③ VLM(SEGMENT_EVENT_PROMPT)+parse_event : 구간별 행동 캡션·유형
      ④ event_builder        : 활동 구간 인접 병합 → 사건(배회는 dwell_s 로 승격)
      ⑤ store.add            : 사건 1개 = 레코드 1줄(+썸네일)
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    video_id = video_id or _video_id(video_path)
    vlm_backend = vlm_backend or config.VLM_BACKEND

    # ① 추적 패스(YOLO+ByteTrack) — 끝나면 GPU 해제(track_video 내부)
    from memory import tracker
    tracks = tracker.track_video(video_path)

    # ② 구간 분할
    segs = segmenter.segment(video_path, tracks=tracks)   # 모션 가이드: 추적으로 '동작 순간'에 프레임 집중
    if not segs:
        print("[index] 구간 없음(코덱 미지원?)", flush=True)
        return {"video_id": video_id, "segments": 0, "events": 0}
    _max = int(os.environ.get("MAX_SEGMENTS", "0"))
    if _max > 0 and len(segs) > _max:
        segs = segs[:_max]                      # 긴 영상 색인 시간 제한(데모)

    # ③ 구간별 VLM 행동 캡션 → parse_event
    vlm = _get_vlm(vlm_backend)
    seg_results = []
    for i, seg in enumerate(segs):
        raw = vlm.caption_frames(seg.frames, config.SEGMENT_EVENT_PROMPT)
        caption, lab = parse_event(raw)
        seg_results.append((caption, lab))
        print(f"  seg {i:2d} [{_fmt_ts(seg.start_s)}-{_fmt_ts(seg.end_s)}] "
              f"{'활동' if lab['activity'] else '·정적'} {lab['event_type']}(sev{lab['severity']})"
              f" · {caption[:32]}", flush=True)

    # ④ 사건 병합
    from memory.event_builder import build_events
    events = build_events(video_id, segs, seg_results, tracks)
    if not events:
        print(f"[index] {video_id}: 사건 없음(활동 구간 0)", flush=True)
        return {"video_id": video_id, "segments": len(segs), "events": 0}

    # ⑤ 썸네일 + '구간(child)' 레코드 저장 — 사건은 event_id 로 묶임(Parent-Child)
    #    구간마다 캡션 원문을 그대로 저장(병합·요약 X) → 검색·분류에서 의미 희석 없음.
    #    유형/심각도는 색인과 분리(오픈셋 분류) — 여기선 unknown/0, reclassify 가 구간별로 채운다.
    store = default_store()
    thumbs_dir = os.path.join(config.MEMORY_DIR, config.THUMBS_SUBDIR, video_id)
    os.makedirs(thumbs_dir, exist_ok=True)
    stamp = datetime.now().isoformat(timespec="seconds")

    def base_meta(ev):
        return {
            "video_id": video_id, "source": os.path.basename(video_path),
            "event_id": ev.event_id, "ev_start_s": ev.start_s, "ev_end_s": ev.end_s,
            "person_count": ev.person_count, "has_vehicle": ev.has_vehicle, "dwell_s": ev.dwell_s,
            "track_ids": json.dumps(ev.track_ids), "objects": json.dumps(ev.objects, ensure_ascii=False),
            "embed_model": config.EMBED_BACKEND, "vlm_backend": vlm_backend, "indexed_at": stamp,
        }

    records = []
    for ev in events:
        if not ev.seg_indices:                       # 추적기반 사건(배회) — 구간 없음, 합성 1레코드
            records.append({"id": ev.event_id, "document": ev.summary, "metadata": {
                **base_meta(ev), "start_s": ev.start_s, "end_s": ev.end_s,
                "event_type": ev.event_type, "severity": ev.severity, "by_tracker": True, "thumb": "",
            }})
            continue
        for si in ev.seg_indices:                    # 구간마다 1레코드(캡션 원문 보존)
            seg, cap = segs[si], seg_results[si][0]
            thumb = ""
            if seg.frames:
                thumb = os.path.join(thumbs_dir, f"s{si}.jpg")
                seg.frames[len(seg.frames) // 2].save(thumb, quality=85)
            records.append({"id": f"{video_id}:s{si}", "document": cap, "metadata": {
                **base_meta(ev), "start_s": seg.start_s, "end_s": seg.end_s,
                "event_type": "unknown", "severity": 0, "thumb": thumb,
            }})
    n = store.add(records)
    print(f"[index] {video_id}: 구간 {n} 색인 / 사건 {len(events)} (총 {store.count()})", flush=True)
    return {"video_id": video_id, "segments": len(segs), "events": len(events)}


def query(question, k=5, where=None, vlm_backend=None, answer=True):
    """질문 → 구간 벡터검색 → event_id 로 묶어 사건 단위 반환. 반환: {answer, segments[]}.

    구간 단위로 매칭(의미 희석 없음)하고, 같은 사건이 여러 구간 걸리면 '최고 점수 구간'을 대표로
    1건만 보여준다(start_s = 그 구간 시각 = 정확 seek). 정렬 = 최고 유사도순.
    """
    store = default_store()
    hits = store.search(question, k=max(k * 4, 12), where=where)        # 구간 단위라 넉넉히 뽑아 묶음
    hits = [h for h in hits if h["score"] >= config.SEARCH_MIN_SCORE]   # 유사도 임계 미만 제외(무관 질의 차단)
    if not hits:
        return {"answer": "색인된 구간이 없거나 검색 결과가 없습니다.", "segments": []}

    by_event = {}                                                       # event_id → 최고점수 구간
    for h in hits:
        eid = h["metadata"].get("event_id") or h["id"]
        if eid not in by_event or h["score"] > by_event[eid]["score"]:
            by_event[eid] = h
    top = sorted(by_event.values(), key=lambda h: h["score"], reverse=True)[:k]

    segments = [{
        "video_id": h["metadata"].get("video_id"),
        "start_s": h["metadata"]["start_s"], "end_s": h["metadata"]["end_s"],   # 대표 구간 = 정확 seek
        "caption": h["document"], "thumb": h["metadata"].get("thumb"),
        "score": h["score"], "event_type": h["metadata"].get("event_type"),
        "severity": h["metadata"].get("severity"), "dwell_s": h["metadata"].get("dwell_s"),
        "person_count": h["metadata"].get("person_count"),
        "has_vehicle": h["metadata"].get("has_vehicle"),
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
        if "--type" in sys.argv:                                  # 예: --type fall / loitering
            where = {"event_type": sys.argv[sys.argv.index("--type") + 1]}
        k = 5
        if "--k" in sys.argv:
            k = int(sys.argv[sys.argv.index("--k") + 1])
        out = query(sys.argv[2], k=k, where=where)
        print("\n=== 답변 ===\n" + str(out["answer"]))
        print("\n=== 근거 사건 ===")
        for s in out["segments"]:
            print(f"[{_fmt_ts(s['start_s'])}-{_fmt_ts(s['end_s'])}] "
                  f"score={s['score']} {s['event_type']}(sev{s['severity']}) · {s['caption'][:40]}")
    else:
        print(__doc__)


if __name__ == "__main__":
    _main()
