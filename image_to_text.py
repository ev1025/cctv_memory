"""
image_to_text.py — VLM 으로 5초 구간 프레임을 한국어 캡션으로 변환.

영상 행동 캡션 전용: 여러 프레임을 video 로 묶어 한 프롬프트에 시계열 투입(caption_frames).
백엔드는 transformers AutoModelForImageTextToText + apply_chat_template 계열(InternVL3 / Qwen-VL).

[VRAM] device_map="auto"(다중 GPU 분산) + bf16, 또는 4bit 단일 GPU. unload() 로 회수.
"""
import config  # ★ torch 보다 먼저 (GPU 격리/HF 캐시)

import re
import gc
import torch
from PIL import Image

import models


def _to_pil(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.open(image).convert("RGB")


def _shrink(img, max_side):
    """긴 변이 max_side 를 넘으면 비율 유지하며 축소 — multi-image OOM 을 '프레임 수' 대신 '해상도'로 관리(hailo 교훈)."""
    w, h = img.size
    m = max(w, h)
    if max_side and m > max_side:
        s = max_side / m
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    return img


def _subsample(items, n):
    """앞쪽 편향 없이 균일 간격 n 개로 고른다(원소 수 ≤ n 이면 그대로)."""
    items = list(items)
    if n <= 0 or len(items) <= n:
        return items
    if n == 1:
        return [items[len(items) // 2]]
    return [items[int(round(i * (len(items) - 1) / (n - 1)))] for i in range(n)]


def _strip_think(text):
    """GLM 등의 <think>...</think> 추론을 제거하고 최종 답만 남긴다(다른 모델엔 무해).

    안 닫힌 <think>(추론이 max_new_tokens 안에 안 끝나 잘림)는 '답이 없는 상태'이므로,
    사고과정 원문을 캡션으로 흘리지 않고 명시적 미완 표시를 반환한다.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    elif "<think>" in text:
        return "[thinking 미완 — max_new_tokens 부족]"
    text = re.sub(r"</?answer>", "", text)   # GLM 등의 <answer> 래퍼 정리
    return text.strip()


class VLMCaptioner:
    """영상 구간 캡셔너. load() → caption_frames() → unload(). name 은 models.VLM_REGISTRY 의 키."""

    def __init__(self, name=None):
        self.name = name or config.VLM_BACKEND
        if self.name not in models.VLM_REGISTRY:
            raise KeyError(f"알 수 없는 VLM 백엔드 '{self.name}'. 사용 가능: {list(models.VLM_REGISTRY)}")
        spec = models.VLM_REGISTRY[self.name]
        self.model_id = spec["id"]
        self.label = spec["label"]
        self.model = None
        self.processor = None

    def load(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        print(f"[VLM] 로딩: {self.label} ({self.model_id})")
        quant = config.build_quant_config()
        # 4bit + 단일 GPU 는 CPU offload(4bit 미지원, ValueError) 회피 위해 단일 GPU 고정. 다중 GPU fp16 은 auto 분산.
        device_map = {"": 0} if (quant is not None and torch.cuda.device_count() == 1) else "auto"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, dtype=config.TORCH_DTYPE, device_map=device_map,
            quantization_config=quant, low_cpu_mem_usage=True)
        self.model.eval()
        dev = getattr(self.model, "hf_device_map", None) or next(self.model.parameters()).device
        print(f"[VLM] 적재 완료. device={dev if not isinstance(dev, dict) else '(분산)'}")
        return self

    @torch.inference_mode()
    def caption_frames(self, images, prompt=None):
        """여러 프레임(구간)을 video 로 묶어 한 프롬프트에 시계열 추론.

        [왜 video] N개 이미지로 넣으면 각자 타일링돼 토큰이 폭증(8장≈12.5k tok)하지만, video 로 넣으면
        시간 병합 + 프레임당 고정 격자라 토큰이 1/3(16장≈4.4k tok) → 8GB 에서도 다(多)프레임이 가능하다.
        video 미지원 모델/오류 시 이미지 경로로 폴백.
        """
        # 구간 전체를 균일 샘플(앞쪽 편향 제거) + 원본 해상도 정리 후 VLM_MAX_FRAMES 장 투입.
        frames = [_shrink(_to_pil(im), config.VLM_FRAME_MAX_SIDE)
                  for im in _subsample(images, config.VLM_MAX_FRAMES)]
        prompt = prompt or "다음 연속 프레임(시간순)의 변화를 바탕으로 장면을 한국어로 간단히 설명하세요."
        try:
            inputs = self._video_inputs(frames, prompt)
        except Exception as e:
            print(f"[VLM] video 입력 실패({type(e).__name__}) → 이미지 경로 폴백", flush=True)
            inputs = self._image_inputs(frames, prompt)
        in_len = inputs["input_ids"].shape[1]
        out = self.model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS_MULTI, do_sample=False)
        text = self.processor.batch_decode(
            out[:, in_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return _strip_think(text)

    def _video_inputs(self, frames, prompt):
        """프레임을 video 로 투입 — apply_chat_template(텍스트만) 후 processor(videos=) 로 프레임 전달(hailo 방식)."""
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:                                   # do_sample_frames=False: 추출 프레임 그대로(프로세서 내부 재샘플 차단)
            inputs = self.processor(text=[text], videos=[frames], padding=True,
                                    return_tensors="pt", do_sample_frames=False)
        except (TypeError, ValueError):        # 일부 프로세서는 do_sample_frames 미지원
            inputs = self.processor(text=[text], videos=[frames], padding=True, return_tensors="pt")
        return inputs.to(self.model.device)

    def _image_inputs(self, frames, prompt):
        """폴백: 프레임을 N개 이미지로 투입(토큰 많음 — video 미지원 모델용)."""
        content = [{"type": "image", "image": im} for im in frames]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        return self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.model.device)

    def unload(self):
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[VLM] 언로드 완료(VRAM 회수)")
