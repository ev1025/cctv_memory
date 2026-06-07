# 3D-Vision — PCTC 화재예방 VLM 파이프라인

자동차 운송 선박(PCTC) 화물칸 CCTV 기반 **화재·이상행동 감지**를 위한 VLM 파이프라인.
이미지·영상을 VLM(4bit)으로 읽어 위험을 분류하고, 영상은 색인해 **자연어로 검색**한다.

> 로컬 단말(RTX 4060 **8GB**)이 실제 운영 환경, 서버(RTX 5090)는 테스트용.
> VLM 은 **bitsandbytes 4bit 양자화**로 8GB 에서 구동.

---

## 세 가지 과정

| 과정 | 내용 | 상태 |
|---|---|---|
| **① Image→Text** | 이미지 → VLM 분류(위험 유무·유형) | ✅ 4종 평가 완료 — **InternVL3 채택** |
| **② Video→Text (video-memory)** | 영상 색인(ChromaDB) → 자연어 검색·RAG 답변 | ✅ 구현(색인·검색·React UI) |
| **③ Text→Image** | 텍스트 → PIL 구조화 + SDXL img2img | ⏸ 보류(`t2i/` 격리) |

---

## 디렉토리 구조

```
3d_vision/
├── (루트 = 코어)
│   ├── config.py            설정(프롬프트·GPU·4bit·MEMORY_DIR·임베딩)
│   ├── models.py            VLM · T2I · EMBED 레지스트리
│   ├── image_to_text.py     VLMCaptioner — 이미지→텍스트(4bit, multi-image)
│   ├── report_utils.py      리포트 공통 유틸(base64·CSS·OpenCV)
│   ├── make_sample_image.py 도형 테스트 이미지 생성
│   └── app.py               FastAPI(i2t · video-memory · UI 서빙)
├── memory/     video-memory (segmenter·text_embedder·vector_store·video_memory)
├── frontend/   React+Vite   (영상 검색 UI) → build → ui-dist/
├── reports/    i2t 평가     (eval_i2t · build_eval_i2t)
├── t2i/        Text→Image   (보류 격리)
├── video/      video-to-text(yolo_trigger · report_v2t)
├── tests/      검증·측정    (local_smoke · measure_vlm_4bit · quant_bench · *_test)
├── tools/      일회성       (download_all · setup)
├── assets/     입력 데이터  (이미지 · embedding/ 영상)
├── outputs/    생성물       (memory/ = chroma · thumbs)
├── results/    i2t 평가 HTML
└── docs/specs/ 설계 문서
```

실행은 **루트에서 `-m`** (import 경로 안정), 또는 로컬 FastAPI `run_local.bat`.

---

## ① Image→Text 분류 평가

데이터셋 34장(위험 fire·smoke·fall·machine + 정상 parking·worker) × 4종 VLM.
이미지당 2회 추론(위험 유무 "예/아니오" + 유형 단어)으로 채점.

| VLM (4bit) | 위험 판정 ↑ | 유형 정확도 |
|---|---|---|
| Qwen2-VL-7B | 56% (19/34) | 85% |
| Qwen2.5-VL-7B | 56% (19/34) | 79% |
| Qwen3-VL-8B | 85% (29/34) | 85% |
| **InternVL3-8B** | **91% (31/34)** | 85% |

→ **InternVL3 채택**(위험 인지 최고). 도형에선 비슷했지만 실제 화재·이상행동 분류는 InternVL3·Qwen3-VL 우세.
결과: `results/eval_i2t.html` (input 프롬프트 + 이미지별 4종 답 표)
실행: `python -m reports.eval_i2t` → `python -m reports.build_eval_i2t`

---

## ② Video→Text — video-memory (영상 RAG)

오프라인 mp4 를 구간별 캡션으로 색인(ChromaDB)하고 자연어로 검색 + RAG 답변. Twelve Labs 의 index→search→generate 구조.

```
[색인] mp4 → segmenter(5초 그리드 + YOLO 보강) → VLM 통합 캡션 → parse → ChromaDB upsert(+썸네일)
[검색] 질문 → 벡터검색(+메타필터 fire=true 등) → 대표 프레임+캡션 → VLM RAG 답변([MM:SS] 인용)
```

- **색인 모델** InternVL3 · **임베딩** bge-m3(CPU, VLM VRAM 경쟁 회피) · **DB** ChromaDB(로컬 HNSW)
- **프론트(React)**: 영상 업로드 → 자연어 검색 → **구간 카드 클릭 → 영상이 그 시각으로 점프**
- **재색인 멱등**(같은 video_id upsert), 구간 1개 = 레코드 1줄 = 검색 단위
- 실행: `uvicorn app:app` → http://localhost:8000 / CLI: `python -m memory.video_memory index <mp4>` · `... query "<질문>"`
- 설계: `docs/specs/video-memory.md`

---

## ③ Text→Image (보류)

도형 배치를 PIL 로 100% 확정한 뒤 SDXL **img2img** 로 입체감만 입히는 방식(diffusion 단독은 개수·배치 제어 불가).
화재 과제와 무관해 `t2i/` 로 격리. 도형 PoC 종합 결과는 바탕화면 `3D-Vision_report.html` + 커밋 이력에 보존.

---

## 모델

### VLM (Image→Text · 색인, 4bit nf4 · 단일 GPU `device_map={"":0}`)
| 모델 | 크기 | 8GB peak | 위험 판정 |
|---|---|---|---|
| Qwen2-VL / 2.5-VL | 7B | ~6.2 GB | 56% |
| Qwen3-VL | 8B | 6.65 GB | 85% |
| **InternVL3** | 8B | 6.25 GB | **91%** |
| ~~Pixtral-12B~~ | 12B | 9.2 GB | 8GB 초과 → 서버 전용 |

### 임베딩 (video-memory)
| 모델 | 차원 | 비고 |
|---|---|---|
| bge-m3 | 1024 | 다국어, CPU 로드 |

---

## 환경
- **로컬(운영)**: RTX 4060 8GB · `C:\Users\eg287\venvs\3dvision` · torch 2.11 cu128 · transformers 5.x · bitsandbytes 4bit · chromadb · sentence-transformers
- **서버(테스트)**: RTX 5090 (GPU 2,3) · fp16
- **프론트**: node 24 · Vite 6 · React 18
- GitHub: ev1025/3D-Vision
