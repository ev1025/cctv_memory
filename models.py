"""
models.py — 벤치마크에 사용할 VLM / T2I 모델 레지스트리

여기에 한 줄만 추가하면 run_benchmark.py 가 자동으로 그 모델까지 비교한다.
(로딩/추론이 실패하는 모델은 벤치마크에서 graceful 하게 '실패'로 기록되고 건너뛴다.)
"""

# ── Image → Text (VLM) ───────────────────────────────────────────────
#   type "standard"  : transformers AutoModelForImageTextToText + apply_chat_template
#                      (Qwen2-VL / Qwen2.5-VL / LLaVA-1.5 / LLaVA-NeXT 등 대부분)
#   type "moondream" : moondream2 전용(custom remote code) — 별도 추론 경로
VLM_REGISTRY = {
    "qwen2-vl":   {"id": "Qwen/Qwen2-VL-7B-Instruct",         "type": "standard",  "label": "Qwen2-VL-7B"},
    "qwen2.5-vl": {"id": "Qwen/Qwen2.5-VL-7B-Instruct",       "type": "standard",  "label": "Qwen2.5-VL-7B"},
    "qwen3-vl":   {"id": "Qwen/Qwen3-VL-8B-Instruct",         "type": "standard",  "label": "Qwen3-VL-8B"},
    # ── 다른 계열 7~12B (통합 인터페이스) ──
    "internvl3":  {"id": "OpenGVLab/InternVL3-8B-hf",         "type": "standard",  "label": "InternVL3-8B"},
    "glm-4.1v":   {"id": "THUDM/GLM-4.1V-9B-Thinking",        "type": "standard",  "label": "GLM-4.1V-9B"},
    "idefics3":   {"id": "HuggingFaceM4/Idefics3-8B-Llama3",  "type": "standard",  "label": "Idefics3-8B"},
    "pixtral":    {"id": "mistral-community/pixtral-12b",     "type": "standard",  "label": "Pixtral-12B"},
    # ── 다른 계열 (custom 백엔드, trust_remote_code) ──
    "ovis2":      {"id": "AIDC-AI/Ovis2-8B",                  "type": "ovis2",     "label": "Ovis2-8B"},
    "minicpm-v":  {"id": "openbmb/MiniCPM-V-4_5",             "type": "minicpm",   "label": "MiniCPM-V-4.5"},
    # (Molmo-7B: transformers 5.x 비호환 / Gemma3-12B: gated → 제외)
    "llava-1.5":  {"id": "llava-hf/llava-1.5-7b-hf",          "type": "standard",  "label": "LLaVA-1.5-7B"},
    "llava-next": {"id": "llava-hf/llava-v1.6-mistral-7b-hf", "type": "standard",  "label": "LLaVA-NeXT-7B (Mistral)"},
    "moondream":  {"id": "vikhyatk/moondream2",               "type": "moondream", "label": "moondream2 (1.8B)"},
}

# ── Text → Image (diffusers) ─────────────────────────────────────────
#   variant  : fp16 가중치만 받을지(없으면 자동 폴백)
#   steps    : 디노이징 스텝 (turbo 류는 1~4)
#   guidance : CFG 강도 (turbo 류는 0 = negative prompt 미사용)
T2I_REGISTRY = {
    "sdxl":       {"id": "stabilityai/stable-diffusion-xl-base-1.0",    "variant": "fp16", "steps": 30, "guidance": 6.5, "label": "SDXL Base"},
    "sd15":       {"id": "stable-diffusion-v1-5/stable-diffusion-v1-5", "variant": None,   "steps": 30, "guidance": 7.5, "label": "SD v1.5"},
    "sdxl-turbo": {"id": "stabilityai/sdxl-turbo",                      "variant": "fp16", "steps": 4,  "guidance": 0.0, "label": "SDXL-Turbo"},
    "sd21":       {"id": "stabilityai/stable-diffusion-2-1",            "variant": "fp16", "steps": 30, "guidance": 7.5, "label": "SD 2.1"},
}

# ── Text → Embedding (sentence-transformers, 다국어) ─────────────────
#   video-memory 색인/검색용. CPU 로드 기본(8GB VLM 과 VRAM 경쟁 회피).
EMBED_REGISTRY = {
    "bge-m3":   {"id": "BAAI/bge-m3",                    "label": "BGE-M3 (다국어, 1024d)"},
    "e5-small": {"id": "intfloat/multilingual-e5-small", "label": "multilingual-e5-small (384d)"},
}
