"""smoke_multi.py — #3 multi-image(caption_frames) 로컬 동작 확인.
도형 이미지 여러 장을 한 프롬프트에 넣어 시계열 추론이 도는지 + VRAM 확인(8GB 초과 여부)."""
import config

import os
import time
import torch
from PIL import Image

from image_to_text import VLMCaptioner
from make_sample_image import make_sample_image

BACKEND = os.environ.get("SMOKE_VLM", "qwen2.5-vl")
N = int(os.environ.get("SMOKE_FRAMES", "3"))


def main():
    if not os.path.exists(config.SAMPLE_IMAGE):
        make_sample_image(config.SAMPLE_IMAGE)
    img = Image.open(config.SAMPLE_IMAGE).convert("RGB")
    frames = [img] * N                              # 같은 도형 N장(구조·VRAM 확인용)

    cap = VLMCaptioner(BACKEND).load()
    t = time.time()
    text = cap.caption_frames(frames)               # multi-image
    sec = round(time.time() - t, 2)
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    cap.unload()

    print(f"\n[multi-image] {N}장 입력 → {text}")
    print(f"[multi-image] peak {peak}GB (8GB {'OK' if peak < 8 else '초과'}) / 추론 {sec}s")
    print("MULTI_SMOKE_DONE")


if __name__ == "__main__":
    main()
