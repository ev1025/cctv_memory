#!/bin/bash
# 16프레임 + 로컬 줄형식 프롬프트 재색인/평가. git pull 없음(수동 업로드한 config/video_memory 보존).
cd ~/cctv_memory
FR=16
echo "[bench16] frames=$FR, 로컬프롬프트(줄형식 캡션:/활동:), 3모델 병렬 색인 시작 $(date)"
CUDA_VISIBLE_DEVICES=1 SEGMENT_FRAMES=$FR EMBED_DEVICE=cuda VLM_BACKEND=internvl3  CHROMA_SUBDIR=chroma_internvl3_f16 THUMBS_SUBDIR=thumbs_internvl3_f16 MAX_SEGMENTS=10 MAX_DURATION=60 .venv/bin/python -m scripts.index_aihub > idx_internvl3_f16.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 SEGMENT_FRAMES=$FR EMBED_DEVICE=cuda VLM_BACKEND=qwen3-vl   CHROMA_SUBDIR=chroma_qwen3vl_f16   THUMBS_SUBDIR=thumbs_qwen3vl_f16   MAX_SEGMENTS=10 MAX_DURATION=60 .venv/bin/python -m scripts.index_aihub > idx_qwen3vl_f16.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 SEGMENT_FRAMES=$FR EMBED_DEVICE=cuda VLM_BACKEND=qwen2.5-vl CHROMA_SUBDIR=chroma_qwen25vl_f16  THUMBS_SUBDIR=thumbs_qwen25vl_f16  MAX_SEGMENTS=10 MAX_DURATION=60 .venv/bin/python -m scripts.index_aihub > idx_qwen25vl_f16.log 2>&1 &
wait
echo "[bench16] 색인 완료 -> eval 병렬 $(date)"
CUDA_VISIBLE_DEVICES=1 EMBED_DEVICE=cuda VLM_BACKEND=internvl3  CHROMA_SUBDIR=chroma_internvl3_f16 THUMBS_SUBDIR=thumbs_internvl3_f16 .venv/bin/python -m scripts.eval_aihub > eval_internvl3_f16.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 EMBED_DEVICE=cuda VLM_BACKEND=qwen3-vl   CHROMA_SUBDIR=chroma_qwen3vl_f16   THUMBS_SUBDIR=thumbs_qwen3vl_f16   .venv/bin/python -m scripts.eval_aihub > eval_qwen3vl_f16.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 EMBED_DEVICE=cuda VLM_BACKEND=qwen2.5-vl CHROMA_SUBDIR=chroma_qwen25vl_f16  THUMBS_SUBDIR=thumbs_qwen25vl_f16  .venv/bin/python -m scripts.eval_aihub > eval_qwen25vl_f16.log 2>&1 &
wait
echo "DONE $(date)" > bench16.done
echo "[bench16] 전체 완료 $(date)"
