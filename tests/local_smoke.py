"""local_smoke.py — 로컬(RTX 4060 8GB) 4bit 추론 스모크 테스트.
실행: CUDA_VISIBLE_DEVICES=0 LOAD_IN_4BIT=1 python local_smoke.py   (모델은 SMOKE_VLM 환경변수)"""
import config

import os
import time
import torch
from PIL import Image

from image_to_text import VLMCaptioner
from make_sample_image import make_sample_image

BACKEND = os.environ.get("SMOKE_VLM", "qwen2.5-vl")


def main():
    if not os.path.exists(config.SAMPLE_IMAGE):
        make_sample_image(config.SAMPLE_IMAGE)
    img = Image.open(config.SAMPLE_IMAGE).convert("RGB")

    print(f"[로컬] GPU={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"4bit={os.environ.get('LOAD_IN_4BIT')} 모델={BACKEND}", flush=True)
    t0 = time.time()
    cap = VLMCaptioner(BACKEND).load()
    load_sec = round(time.time() - t0, 1)
    t = time.time()
    text = cap.caption(img)
    infer_sec = round(time.time() - t, 2)
    peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    cap.unload()

    print(f"\n[결과] 출력 : {text}")
    print(f"[결과] VRAM : peak {peak}GB (로컬 8GB 중)")
    print(f"[결과] 시간 : 로드 {load_sec}s / 추론 {infer_sec}s")
    print("LOCAL_SMOKE_DONE")


if __name__ == "__main__":
    main()
