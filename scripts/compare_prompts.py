"""compare_prompts.py — 캡션 프롬프트 비교: 유형분류 시킴 vs 안 시킴(묘사만).

같은 구간 프레임에 두 프롬프트를 돌려 '묘사(caption)' 차이를 본다.
  - 분류형 (SEGMENT_CLASSIFY_PROMPT): 묘사 + 유형 분류를 함께 시킴
  - 묘사형 (SEGMENT_EVENT_PROMPT)   : 묘사만 (현재 채택)

가설: 분류를 함께 시키면 묘사가 유형 쪽으로 끌려가 편향·빈약해진다.
실행: CUDA_VISIBLE_DEVICES=0 LOAD_IN_4BIT=1 python -m scripts.compare_prompts
"""
import config

import os

import cv2
from PIL import Image

from memory import video_memory
from image_to_text import VLMCaptioner

VID = os.path.join(config.MEMORY_DIR, "videos")
CLIPS = ["fall_E02_041", "fight_E03_029", "intrusion_E01_053",
         "crowd_E04_067", "density_E05_006", "flood_E06_007"]


def mid_frames(path, n=4):
    """영상 중앙 부근에서 n장(시간순) — 사건은 보통 가운데에 있음."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frames = []
    for i in range(n):
        idx = int(total * (0.4 + 0.2 * i / max(1, n - 1))) if total else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def _cap(raw):
    obj = video_memory._extract_json(raw)
    return (obj.get("caption") if obj else None) or (raw or "").strip().replace("\n", " ")[:120]


def main():
    vlm = VLMCaptioner(config.VLM_BACKEND).load()
    for vid in CLIPS:
        path = os.path.join(VID, vid + ".mp4")
        if not os.path.exists(path):
            print(f"[skip] {vid} (없음)")
            continue
        frames = mid_frames(path)
        if not frames:
            continue
        c_raw = vlm.caption_frames(frames, config.SEGMENT_CLASSIFY_PROMPT)
        d_raw = vlm.caption_frames(frames, config.SEGMENT_EVENT_PROMPT)
        print(f"\n[{vid}]")
        print(f"  분류형(유형까지): {_cap(c_raw)}")
        print(f"  묘사형(묘사만)  : {_cap(d_raw)}")
    vlm.unload()
    print("\nCOMPARE_DONE")


if __name__ == "__main__":
    main()
