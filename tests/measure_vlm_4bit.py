"""measure_vlm_4bit.py — 받은 VLM 들을 4bit 단일 GPU(cuda:0) 로 적재·추론해 peak VRAM 측정.
'로컬 RTX 4060 8GB 에 어떤 모델이 들어가는지'를 확정한다(device_map={"":0} 로 한 장 강제)."""
import config

import time
import gc
import os
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from make_sample_image import make_sample_image

MODELS = {
    "qwen2-vl":    "Qwen/Qwen2-VL-7B-Instruct",
    "qwen2.5-vl":  "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3-vl":    "Qwen/Qwen3-VL-8B-Instruct",
    "internvl3":   "OpenGVLab/InternVL3-8B-hf",
    "pixtral":     "mistral-community/pixtral-12b",
}
PROMPT = config.VLM_PROMPT


def _caption(model, proc, img):
    msg = [{"role": "user", "content": [
        {"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
    inp = proc.apply_chat_template(msg, add_generation_prompt=True, tokenize=True,
                                   return_dict=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inp, max_new_tokens=64, do_sample=False)
    return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def measure(mid, img):
    proc = AutoProcessor.from_pretrained(mid)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    torch.cuda.set_device(0)
    torch.zeros(1, device="cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForImageTextToText.from_pretrained(
        mid, dtype=torch.bfloat16, device_map={"": 0}, quantization_config=bnb).eval()
    load_gb = round(torch.cuda.memory_allocated() / 1e9, 2)
    _caption(model, proc, img)                       # warmup
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    txt = _caption(model, proc, img)
    sec = round(time.time() - t, 2)
    peak_gb = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    del model, proc
    gc.collect()
    torch.cuda.empty_cache()
    return load_gb, peak_gb, sec, txt


def main():
    if not os.path.exists(config.SAMPLE_IMAGE):
        make_sample_image(config.SAMPLE_IMAGE)
    img = Image.open(config.SAMPLE_IMAGE).convert("RGB")

    print(f"{'model':12} {'load':>6} {'peak':>7} {'8GB':>4} {'sec':>6}  text")
    print("-" * 70)
    for name, mid in MODELS.items():
        try:
            load_gb, peak_gb, sec, txt = measure(mid, img)
            fit = "O" if peak_gb < 8 else "X"
            print(f"{name:12} {load_gb:>5}G {peak_gb:>6}G {fit:>4} {sec:>5}s  {txt[:40]}", flush=True)
        except Exception as e:
            print(f"{name:12} FAIL: {str(e)[:60]}", flush=True)
    print("MEASURE_DONE")


if __name__ == "__main__":
    main()
