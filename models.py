"""
models.py — cctv_memory 모델 레지스트리 (영상 캡션 VLM + 임베딩).

VLM 은 transformers AutoModelForImageTextToText(apply_chat_template) 계열 3종.
config.VLM_BACKEND 로 선택(기본 internvl3). 임베딩은 video-memory 색인/검색용 bge-m3.
"""

# ── 영상 캡션 VLM (프레임 → 한국어 캡션) ──────────────────────────────
VLM_REGISTRY = {
    "internvl3":  {"id": "OpenGVLab/InternVL3-8B-hf",   "label": "InternVL3-8B"},
    "qwen3-vl":   {"id": "Qwen/Qwen3-VL-8B-Instruct",   "label": "Qwen3-VL-8B"},
    "qwen2.5-vl": {"id": "Qwen/Qwen2.5-VL-7B-Instruct", "label": "Qwen2.5-VL-7B"},
}

# ── 임베딩 (sentence-transformers, 다국어) — video-memory 색인/검색용 ──
EMBED_REGISTRY = {
    "bge-m3": {"id": "BAAI/bge-m3", "label": "BGE-M3 (다국어, 1024d)"},
}
