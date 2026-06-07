"""quant_bench.py — Qwen2.5-VL-7B fp16 vs bitsandbytes 4bit 의 VRAM·속도 비교 (단일 GPU 기준).
device_map={"":0} 로 한 장(cuda:0=물리2)에만 올려, '로컬 8GB 에 들어가는지'를 정확히 측정."""
import config

import time
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from make_sample_image import make_sample_image
import os

MID = "Qwen/Qwen2.5-VL-7B-Instruct"
PROMPT = config.VLM_PROMPT


def _caption(model, processor, img):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def measure(use_4bit, img):
    processor = AutoProcessor.from_pretrained(MID)
    kwargs = dict(dtype=torch.bfloat16, device_map={"": 0})   # 단일 GPU(cuda:0=물리2)
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

    torch.cuda.set_device(0)
    torch.zeros(1, device="cuda:0")                            # CUDA context 초기화
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(MID, **kwargs).eval()
    load_sec = round(time.time() - t0, 1)
    load_gb = round(torch.cuda.memory_allocated() / 1e9, 2)

    _caption(model, processor, img)                            # warmup
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(3):
        t = time.time()
        txt = _caption(model, processor, img)
        times.append(time.time() - t)
    peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    del model, processor
    torch.cuda.empty_cache()
    return {"load_sec": load_sec, "load_gb": load_gb, "peak_gb": peak_gb,
            "infer_sec": round(sum(times) / len(times), 2), "text": txt}


def main():
    if not os.path.exists(config.SAMPLE_IMAGE):
        make_sample_image(config.SAMPLE_IMAGE)
    img = Image.open(config.SAMPLE_IMAGE).convert("RGB")

    print("=== fp16 측정 ===")
    fp16 = measure(False, img)
    print(fp16)
    print("=== 4bit 측정 ===")
    q4 = measure(True, img)
    print(q4)

    print("\n================ 비교 (단일 GPU) ================")
    print(f"  적재 VRAM : fp16 {fp16['load_gb']:>5}GB → 4bit {q4['load_gb']:>5}GB")
    print(f"  추론 peak : fp16 {fp16['peak_gb']:>5}GB → 4bit {q4['peak_gb']:>5}GB")
    print(f"  추론 속도 : fp16 {fp16['infer_sec']:>5}s  → 4bit {q4['infer_sec']:>5}s  "
          f"({q4['infer_sec']/max(fp16['infer_sec'],0.01):.2f}x)")
    print(f"  8GB 적합  : fp16 {'O' if fp16['peak_gb']<8 else 'X'} / 4bit {'O' if q4['peak_gb']<8 else 'X'}")
    print("QUANT_BENCH_DONE")


if __name__ == "__main__":
    main()
