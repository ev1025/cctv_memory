# video-memory 스펙 (3d_vision) — 초안

> 상태: **설계 확정 / 코드 0줄**. 이 문서 승인 → 구현 계획 → 구현.
> 진행: 재구성+eval+스펙 **먼저 커밋(→push)** → main 에서 바로 작업. add-only(기존 0 수정)라 **브랜치 불필요**(혼자 작업·매일 보고).

## 목적
PCTC 화재예방 CCTV 영상(오프라인 mp4)에 **이력(타임라인 메타데이터)** 을 만들어 DB 저장,
자연어로 **RAG 검색 + 답변**. Twelve Labs 의 index → search → generate 구조를 기존 자산으로 구현.

## 확정 결정
- 용도: PCTC 화재예방 CCTV 특화(이상행동/화재 타임라인 + 도메인 질의)
- 입력: 오프라인 mp4 색인(실시간 아님)
- 구간 분할: **하이브리드** — 고정 5초 그리드 전체 커버 + YOLO 가 사람/이벤트 잡은 구간 촘촘히
- 출력: 구간(타임스탬프+썸네일) + 자연어 답변 둘 다
- 접근: **캡션-RAG(A안)** — 구간을 한국어 캡션 → 텍스트 임베딩 색인, 질의도 텍스트 임베딩 검색
- DB: **ChromaDB**(로컬 PersistentClient, HNSW). 확장 시 동일 코드로 서버/클라우드 전환
- 임베딩: bge-m3 등 다국어, **CPU 로드 기본**(8GB VLM 과 VRAM 경쟁 회피)

## 모듈 위치 — ⚠️ 인수인계와 다름 (정리 반영)
인수인계는 "루트 flat 4개"였으나, 루트는 **Image→Text 코어만** 두기로 정리됨.
→ **`memory/` 디렉토리**에 신규 모듈 배치(루트·video/ 와 분리, 혼동 방지).

| 모듈 | 위치 | 역할 |
|---|---|---|
| `memory/segmenter.py` | 신규 | 하이브리드 구간 분할(5초 그리드 + YOLO 보강). VLM 미호출, 구간 목록만 |
| `memory/text_embedder.py` | 신규 | 다국어 텍스트 임베딩 래퍼(load/encode/unload, VLMCaptioner 동형). Chroma EmbeddingFunction |
| `memory/vector_store.py` | 신규 | ChromaDB 래퍼 — 저장 + 메타필터+벡터 검색 |
| `memory/video_memory.py` | 신규 | 오케스트레이터 index_video()/query(), `__main__` CLI |

**재사용(import)**: `image_to_text.VLMCaptioner.caption_frames`(루트), `video.yolo_trigger`(격리됨 → `from video.yolo_trigger import …` 경로 주의).
**기존 add-only**: `app.py`에 `POST /index-video`·`POST /query`, `config.py`(프롬프트·MEMORY_DIR·윈도우·임베딩 백엔드), `models.py`에 `EMBED_REGISTRY`.

## 데이터 모델 (ChromaDB)
```
outputs/memory/
├── chroma/                     # PersistentClient (HNSW)
└── thumbs/<video_id>/seg_N.jpg # 대표 프레임(경로만 메타데이터)
```
컬렉션 `cargo_cctv` — 구간 1개 = 레코드 1줄 = 검색 단위:
- id: `"{video_id}:{seg_id}"` (재색인 시 동일 → **upsert 멱등**, 중복 누적 방지)
- embedding: 캡션 임베딩(커스텀 EmbeddingFunction = text_embedder, 문서·질의 동일 모델)
- document: 한국어 캡션
- metadata(스칼라만): video_id, source, start_s, end_s, frame_idxs(**JSON 문자열**), trigger("grid"/"yolo"), person_count, fire(bool), risk_type, severity(int), thumb(경로), fps, embed_model, vlm_backend, indexed_at
- `where={"fire":true,…}` 메타 필터 + 벡터검색 동시. `embed_model` 박아 모델 변경 시 재색인 유도.

## 색인 흐름 index_video()
```
mp4 → segmenter.segment()  → [Segment(start_s,end_s,frames[],trigger,person_count)…]
 for seg:
   VLMCaptioner.caption_frames(seg.frames, SEGMENT_RISK_PROMPT)
     → "설명:… / 위험:있음|없음 / 유형:… / 심각도:0~3"
   parse_risk(text) → caption,{fire,type,severity}   # 관대한 파싱 + 폴백
   save_thumb(중간 프레임) → thumbs/<video_id>/seg_i.jpg
 vector_store.add(records)  # Chroma upsert(임베딩 자동)
```
VLM 1회 로드 후 전 구간 순회(슬롯 캐시), 임베딩은 CPU.

## 검색·답변 흐름 query()
```
질문 [+ 선택 필터 fire=true]
 vector_store.search(q, k, where)   # 질의 임베딩 → top-k 구간(+메타 필터)
 build_context(top_k)               # 타임스탬프+캡션 + 대표 프레임
 VLMCaptioner.caption_frames(대표프레임들, RAG_ANSWER_PROMPT(질문,컨텍스트))
   → 근거 인용 답변 "[02:14] 작업자 2명이 …"
 return { answer, segments:[{start_s,end_s,caption,thumb,score,risk}] }
```
멀티모달 RAG: 검색 구간의 캡션(텍스트) + 실제 대표 프레임을 VLM 에 함께 → 캡션 누락분 프레임 보정.

## 미정 / 다음
- 임베딩 모델 최종(bge-m3 vs multilingual-e5-small vs 한국어 특화), 5초 윈도우 튜닝
- 에러처리: 코덱 실패 / VLM OOM / 빈 인덱스 / top_k 클램프 / risk 파싱 폴백
- 테스트: 단위(segmenter·parse_risk·vector_store) + 통합(짧은 mp4 색인→검색)
- 인터페이스: API/CLI 시그니처, 데모 UI 여부

## 진행 절차 (꼬임 방지)
1. **재구성+eval+스펙 먼저 커밋 → push** (섞임 방지는 이걸로 충분 — 재구성과 신규가 다른 커밋)
2. main 에서 바로 작업 — add-only 라 브랜치 불필요
3. 구현은 `memory/` 안에서만(루트 변경 없음), app/config/models 는 **추가만**
