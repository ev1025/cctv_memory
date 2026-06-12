"""compare_prompts_full.py — 같은 16프레임 구간에 두 프롬프트를 돌려 '묘사 차이' 비교.

  EVENT   (SEGMENT_EVENT_PROMPT)    : 캡션 + 활동(유형 안 시킴) — bench16 채택
  CLASSIFY(SEGMENT_CLASSIFY_PROMPT) : 캡션 + 유형 분류 함께

가설: 유형을 함께 시키면 묘사가 유형 쪽으로 끌려가 편향·빈약해진다.
출력: outputs/vmem/prompt_compare_<backend>.json + 콘솔 요약.
실행: VLM_BACKEND=qwen3-vl SEGMENT_FRAMES=16 MAX_SEGMENTS=10 MAX_DURATION=60 EMBED_DEVICE=cuda python -m scripts.compare_prompts_full
"""
import config

import os
import re
import json

from memory import segmenter
from memory import video_memory
from image_to_text import VLMCaptioner

VID = os.path.join(config.MEMORY_DIR, "videos")
GT = {"fall": "낙상", "fight": "싸움", "intrusion": "침입", "crowd": "군집", "density": "인파밀집", "flood": "침수"}
TYPES = ("falldown", "fight", "invasion", "gathering", "crowd", "flood", "normal")
_MAXSEG = int(os.environ.get("MAX_SEGMENTS", "10"))


def parse_classify(raw):
    """CLASSIFY 출력(캡션:/유형:) → (caption, type)."""
    t = raw or ""
    m = re.search(r"캡션\s*[:：]\s*(.+)", t)
    cap = re.split(r"유형\s*[:：]", m.group(1))[0] if m else t.replace("\n", " ")
    cap = re.sub(r'[*#`{}"]+', "", cap).strip(" ,.\t")[:200]
    tm = re.search(r"유형\s*[:：]?\s*(falldown|fight|invasion|gathering|crowd|flood|normal)", t, re.I)
    return (cap or "(빈 캡션)"), (tm.group(1).lower() if tm else "?")


def main():
    backend = config.VLM_BACKEND
    clips = sorted(os.path.basename(p)[:-4] for p in __import__("glob").glob(os.path.join(VID, "*.mp4"))
                   if os.path.basename(p).split("_")[0] in GT)
    vlm = VLMCaptioner(backend).load()
    out = []
    for vid in clips:
        path = os.path.join(VID, vid + ".mp4")
        segs = segmenter.segment(path)[:_MAXSEG]
        gt = GT.get(vid.split("_")[0], "?")
        print(f"\n===== {vid} (GT: {gt}) =====", flush=True)
        for seg in segs:
            e_raw = vlm.caption_frames(seg.frames, config.SEGMENT_EVENT_PROMPT)
            e_cap, e_lab = video_memory.parse_event(e_raw)
            c_raw = vlm.caption_frames(seg.frames, config.SEGMENT_CLASSIFY_PROMPT)
            c_cap, c_type = parse_classify(c_raw)
            out.append({"video_id": vid, "gt": gt, "start_s": seg.start_s,
                        "event_cap": e_cap, "classify_cap": c_cap, "classify_type": c_type})
            print(f"  [{int(seg.start_s):>3}s] 묘사만 : {e_cap}", flush=True)
            print(f"        분류형({c_type}): {c_cap}", flush=True)
    vlm.unload()
    dst = os.path.join(config.MEMORY_DIR, f"prompt_compare_{backend}.json")
    json.dump({"backend": backend, "rows": out}, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nwrote {dst} — {len(out)} segments\nCOMPARE_DONE", flush=True)


if __name__ == "__main__":
    main()
