"""
image_to_text.py — VLM 으로 이미지를 한국어 캡션으로 변환 (다중 백엔드 통합)

[백엔드 타입]
  - "standard"  : AutoModelForImageTextToText + apply_chat_template
                  (Qwen2/2.5/3-VL, InternVL3, GLM-4.1V, Idefics3, Pixtral, LLaVA 등)
  - "moondream" : moondream2 전용(custom)
  - "ovis2"     : AIDC-AI/Ovis2 전용(custom, preprocess_inputs + get_*_tokenizer)
  - "minicpm"   : openbmb/MiniCPM-V 전용(custom, model.chat)

[VRAM] device_map="auto"(2,3번 분산) + bf16. custom 은 단일 GPU(.cuda()). unload() 로 회수.
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
    """앞쪽 편향 없이 균일 간격 n 개로 고른다(원소 수 ≤ n 이면 그대로). 옛 images[:2](앞 2장만) 대체."""
    items = list(items)
    if n <= 0 or len(items) <= n:
        return items
    if n == 1:
        return [items[len(items) // 2]]
    return [items[int(round(i * (len(items) - 1) / (n - 1)))] for i in range(n)]


def _strip_think(text):
    """GLM-4.1V 등의 <think>...</think> chain-of-thought 를 제거하고 최종 답만 남긴다(다른 모델엔 무해).

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
    """VLM 캡셔너. load() → caption() → unload(). name 은 models.VLM_REGISTRY 의 키."""

    def __init__(self, name=None):
        self.name = name or config.VLM_BACKEND
        if self.name not in models.VLM_REGISTRY:
            raise KeyError(f"알 수 없는 VLM 백엔드 '{self.name}'. 사용 가능: {list(models.VLM_REGISTRY)}")
        spec = models.VLM_REGISTRY[self.name]
        self.model_id = spec["id"]
        self.type = spec["type"]
        self.label = spec["label"]
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.text_tokenizer = None
        self.visual_tokenizer = None

    def load(self):
        print(f"[VLM] 로딩: {self.label} ({self.model_id}) type={self.type}")
        loader = {
            "moondream": self._load_moondream,
            "ovis2": self._load_ovis2,
            "minicpm": self._load_minicpm,
        }.get(self.type, self._load_standard)
        loader()
        dev = getattr(self.model, "hf_device_map", None) or next(self.model.parameters()).device
        print(f"[VLM] 적재 완료. device={dev if not isinstance(dev, dict) else '(분산)'}")
        return self

    # ── 로더 ──────────────────────────────────────────────────────────
    def _load_standard(self):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        quant = config.build_quant_config()
        # 4bit + 단일 GPU 는 CPU offload(4bit 미지원, ValueError)를 막기 위해 단일 GPU 로 고정.
        # 서버(다중 GPU) fp16 은 auto 분산 유지.
        device_map = {"": 0} if (quant is not None and torch.cuda.device_count() == 1) else "auto"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id, dtype=config.TORCH_DTYPE, device_map=device_map,
            quantization_config=quant, low_cpu_mem_usage=True)
        self.model.eval()

    def _load_moondream(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, trust_remote_code=True, dtype=config.TORCH_DTYPE, device_map={"": 0})
        self.model.eval()

    def _load_ovis2(self):
        from transformers import AutoModelForCausalLM
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=config.TORCH_DTYPE, trust_remote_code=True,
            multimodal_max_length=8192).cuda()
        self.model.eval()
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()

    def _load_minicpm(self):
        from transformers import AutoModel, AutoTokenizer
        self.model = AutoModel.from_pretrained(
            self.model_id, trust_remote_code=True, torch_dtype=config.TORCH_DTYPE,
            attn_implementation="sdpa").eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

    # ── 추론 ──────────────────────────────────────────────────────────
    @torch.inference_mode()
    def caption(self, image, prompt=None):
        image = _to_pil(image)
        prompt = prompt or "이 이미지를 한국어로 간단히 설명하세요."
        fn = {
            "moondream": self._caption_moondream,
            "ovis2": self._caption_ovis2,
            "minicpm": self._caption_minicpm,
        }.get(self.type, self._caption_standard)
        return fn(image, prompt)

    @torch.inference_mode()
    def caption_frames(self, images, prompt=None):
        """여러 프레임(이미지 리스트)을 한 프롬프트에 넣어 시계열 추론(multi-image). standard 백엔드 전용.

        #3 영상 행동 인식: 프레임을 '영상(video)'으로 한 번에 넣어 시간 흐름을 추론하게 한다.
        [왜 video] N개 이미지로 넣으면 각자 타일링돼 토큰이 폭증(8장≈12.5k tok)하지만, video 로 넣으면
        시간 병합 + 프레임당 고정 격자라 토큰이 1/3(16장≈4.4k tok) → 8GB 에서도 다(多)프레임이 가능하다.
        video 미지원 모델/오류 시 이미지 경로로 폴백.
        """
        if self.type != "standard":
            raise NotImplementedError(f"multi-image 는 standard 백엔드만 지원합니다(현재 type={self.type}).")
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
        max_tok = getattr(config, "MAX_NEW_TOKENS_MULTI", config.MAX_NEW_TOKENS)
        out = self.model.generate(**inputs, max_new_tokens=max_tok, do_sample=False)
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

    def _caption_standard(self, image, prompt):
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        return _strip_think(text)   # GLM <think> 제거

    def _caption_moondream(self, image, prompt):
        if hasattr(self.model, "query"):
            return str(self.model.query(image, prompt)["answer"]).strip()
        enc = self.model.encode_image(image)
        return str(self.model.answer_question(enc, prompt, self.tokenizer)).strip()

    def _caption_ovis2(self, image, prompt):
        query = f"<image>\n{prompt}"
        _, input_ids, pixel_values = self.model.preprocess_inputs(query, [image], max_partition=9)
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.model.device)
        pixel_values = pixel_values.to(dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device)
        out = self.model.generate(input_ids, pixel_values=[pixel_values], attention_mask=attention_mask,
                                  max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False)
        return self.text_tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def _caption_minicpm(self, image, prompt):
        msgs = [{"role": "user", "content": [image, prompt]}]
        res = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer)
        return str(res).strip()

    def unload(self):
        for attr in ("model", "processor", "tokenizer", "text_tokenizer", "visual_tokenizer"):
            setattr(self, attr, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[VLM] 언로드 완료(VRAM 회수)")


def image_to_text(image, name=None):
    cap = VLMCaptioner(name=name).load()
    try:
        return cap.caption(image)
    finally:
        cap.unload()


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else None
    path = sys.argv[2] if len(sys.argv) > 2 else config.SAMPLE_IMAGE
    print("출력:", repr(image_to_text(path, name=name)))
