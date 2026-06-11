# cctv_memory — CCTV 행동 이력 메모리 (VLM 기반 관제 콘솔)

CCTV 녹화 영상을 **VLM이 분석해 사람의 행동·특이사항을 "사건(event)"으로 추출**하고,
ChromaDB에 색인해 **자연어로 검색**하고 **카메라·시각별 타임라인 이력**으로 관리하는 관제 콘솔.

> - 입력은 **저장된 CCTV 파일**(오프라인 mp4). 라이브 스트림 추론이 아니라 녹화본 배치 색인.
> - 화재 감지는 **별도 열화상 카메라**가 담당. 본 시스템은 **행동/이벤트 이력**에 집중.
> - 로컬 운영 = **RTX 4060 8GB** (bitsandbytes 4bit). 임베딩 bge-m3(CPU).

> **레포 범위:** 이 레포는 **행동 메모리 시스템 전용**입니다.
> 생성 라인(이미지→텍스트 평가 · 텍스트→이미지 t2i · img2img · 3D화)은 별도 레포 **[ev1025/3D-Vision](https://github.com/ev1025/3D-Vision)** 으로 분리됨.
> (공유되는 VLM 캡셔너 `image_to_text.py`만 양쪽이 각자 사본 보유 — 공용 코어 없음.)

---

## 1. 산출물 (Deliverables)

| 산출물 | 상태 | 위치 |
|---|---|---|
| **관제 콘솔** (React + FastAPI) — 3×3 카메라 그리드 + 실시간 상황 로그 + 자연어 행동 검색 + 사건 타임라인 | ✅ 동작 | `app.py` + `frontend/` → `ui-dist/`, http://localhost:8000 |
| **사건 메모리 파이프라인** — 영상 → 추적 → VLM 캡션 → 사건 병합 → ChromaDB | ✅ 동작 | `memory/`, 데모 8영상 = **23 사건** 색인 |
| **CCTV 운용 시뮬레이션** — 샘플 영상에 카메라·녹화시각 부여 → 실제 시각·날짜로 운용 | ✅ 동작 | `outputs/vmem/cctv_map.json` + `cctv_meta.py` |
| **수집(ingestion) 아키텍처 설계** — 실시간 녹화 → 파일 색인 (§5) | ✅ 설계 | 본 문서 §5 |
| **실 데이터 확보** — AI-Hub 71850(지능형 관제 CCTV) 실 mp4 영상 13GB | ✅ 보관 | `assets/07.지능형 관제 서비스 CCTV 영상 데이터/` |

> VLM 모델 비교(4종→InternVL3 채택, 91%) 등 **i2t 평가 산출물은 3d_vision 레포**로 이동.

---

## 2. 핵심 설계 — 행동 이벤트 메모리

오프라인 mp4 → 사람·차량 추적 → VLM 행동 캡션 → **연속 활동을 "사건"으로 병합** → ChromaDB 색인 → 자연어 검색 + RAG.

- **렌즈:** 초기엔 "화재 위험" 분류였으나, 화재는 열화상이 담당하므로 **"사람 행동/특이사항 이력"** 으로 전환.
  분류 기준도 *위험* → *활동(activity)* 으로 변경(활동 있으면 다 기록, 정적 배경만 제외, 위험은 severity로 알림).
- **사건 단위 병합:** 5초 구간을 그대로 저장하지 않고, 같은 유형의 연속 구간을 **하나의 사건(시작~종료+요약)** 으로 묶음(`event_builder`).
- **사람 추적(ByteTrack):** 입·출입·체류시간(dwell)을 결정적으로 산출 → **배회(loitering)는 VLM이 아니라 dwell_s 임계로 판정**.
- **taxonomy:** `fall(낙상) · vehicle_interaction(차량접촉) · smoking(흡연) · flammable(인화물) · loitering(배회) · normal · unknown`.
- **채택 VLM:** **InternVL3-8B(4bit)** (서버 비교에서 위험 인지 91%로 최고). 13종 후보는 `models.py VLM_REGISTRY`.

---

## 3. 현재 워크플로우 (How it works)

```
[녹화]   IP카메라 ─RTSP→ 녹화기(NVR/ffmpeg, -c copy) → 시간분할 mp4 파일      ← §5 (실 운영, 미구현)
            └─ (데모) AI-Hub 샘플 mp4 + cctv_map.json 으로 카메라·녹화시각 부여

[색인]   mp4 ──► memory.video_memory.index_video(path, video_id)
          ① tracker.track_video      YOLO+ByteTrack → 사람·차량 track(등장·퇴장·dwell·이동)   ← VLM 전, 8GB 위해 GPU 선점/해제
          ② segmenter.segment        5초 그리드 구간(+YOLO 사람 잡힌 구간 촘촘히)
          ③ VLM(InternVL3-8B 4bit)   구간별 SEGMENT_EVENT_PROMPT → parse_event(JSON)
          ④ event_builder.build_events  활동 구간을 유형별로 병합 → 사건(배회는 dwell 승격)
          ⑤ vector_store.add         사건 1개 = ChromaDB 레코드 1줄(+썸네일), event_id 멱등

[조회]   FastAPI(app.py) ──► React 콘솔(ui-dist)
          /cameras  카메라 그리드            /history  카메라·시각순 이력(실시간 상황 로그)
          /query    자연어 행동 검색(벡터)    /alerts   특이사건(severity≥1) 알림
          /segments 카메라 사건 타임라인       /video,/thumb  영상·썸네일 서빙
```

- **검색 흐름:** 질문 → bge-m3 임베딩 → ChromaDB 벡터검색(유사도 컷 `SEARCH_MIN_SCORE=0.52`, +`event_type` 메타필터) → 사건 카드(유형 칩·체류·시각). 클릭 → 영상 그 시각 점프.
- **시각/카메라:** `cctv_meta`가 `cctv_map.json`의 `recorded_at`(녹화 시작시각)에서 **사건 절대시각 = recorded_at + 구간오프셋** 을 계산 → 콘솔이 실제 시각(예: `10:22:15`)·날짜·카메라명으로 표시.
- **온도/열화상:** 현재 `cctv_meta._synth_temp` 합성 플레이스홀더(표시용). 실데이터 전환 = `outputs/vmem/thermal/{id}.json` + `{id}__thermal.mp4` 드롭(코드 수정 불필요).

---

## 4. 아키텍처 / 핵심 모듈

| 파일 | 역할 |
|---|---|
| `config.py` | 설정(GPU/4bit·MEMORY_DIR·임베딩)·프롬프트(`SEGMENT_EVENT_PROMPT`·`RAG_ANSWER_PROMPT`)·taxonomy·임계값(`LOITER_DWELL_S`·`EVENT_MERGE_GAP_S`·`SEARCH_MIN_SCORE`) |
| `models.py` | VLM(12종 레지스트리)·EMBED(bge-m3) 사양 |
| `image_to_text.py` | `VLMCaptioner` — 4bit VLM 로드/캡션(multi-image, `caption_frames`는 OOM 위해 프레임 2장 제한) |
| `memory/tracker.py` | YOLO+ByteTrack 추적 패스(사람·차량 ID·dwell·이동) |
| `memory/segmenter.py` | 5초 그리드 구간 분할(+YOLO 보강) |
| `memory/video_memory.py` | 오케스트레이터 `index_video`/`query` + `parse_event`(JSON 파싱) |
| `memory/event_builder.py` | 구간 → 사건 병합(유형별 분리 + dwell 배회 승격) |
| `memory/vector_store.py` | ChromaDB 래퍼(upsert 멱등 + 메타필터·벡터 검색) |
| `memory/text_embedder.py` | bge-m3 임베딩(CPU) + Chroma EmbeddingFunction |
| `cctv_meta.py` | 카메라·날짜·**절대시각**·온도 메타 레이어(`cctv_map.json` override) |
| `app.py` | FastAPI(콘솔 API + 원본 VLM API + ui-dist 서빙) |
| `index_all.py` | 데모 배치 색인(여러 영상) |
| `video/yolo_trigger.py` | YOLO 사람 트리거(segmenter가 `PERSON_CLASS` 재사용) |
| `frontend/` | React+Vite 콘솔 → `npm run build` → `../ui-dist/` |

### 데이터 모델 (ChromaDB, 컬렉션 `cargo_cctv`)
사건 1개 = 레코드 1줄. `id="{video_id}:e{n}"`(멱등). metadata(스칼라): `video_id, source, start_s, end_s, duration_s, event_type, severity, person_count, has_vehicle, dwell_s, track_ids(JSON), objects(JSON), thumb, embed_model, vlm_backend, indexed_at`. document = 한국어 사건 요약(임베딩 대상).

---

## 5. 실행 방법

```bash
# venv: 색인/VLM = GPU venv (C:/Users/eg287/venvs/3dvision, torch cu128). 서버(retrieval)만 CPU도 가능.

# 색인 (GPU) — 재색인 전 outputs/vmem/chroma 삭제 + 서버 중지 권장
CUDA_VISIBLE_DEVICES=0 LOAD_IN_4BIT=1 VLM_BACKEND=internvl3 MAX_SEGMENTS=12 MAX_DURATION=60 \
  python index_all.py
#   또는 단일:  python -m memory.video_memory index <mp4> [video_id]

# 검색 (CLI)
python -m memory.video_memory query "쓰러진 사람" [--type fall] [--k 5]

# 서버 (콘솔 + UI)
CUDA_VISIBLE_DEVICES=0 python -m uvicorn app:app --host 127.0.0.1 --port 8000   # → http://localhost:8000

# 프론트 개발/빌드
cd frontend && npm run dev      # http://localhost:5173 (API는 :8000 프록시)
cd frontend && npm run build    # → ../ui-dist (FastAPI가 서빙)
```
> 재색인 후에는 서버를 **재시작**해야 검색 HNSW가 완전 갱신됨. 색인(write)과 서버(read)의 chroma 동시접근은 피할 것.

### CCTV 운용 시뮬레이션 (샘플 → 실 CCTV처럼)
`outputs/vmem/cctv_map.json`에 영상별 `{camera_id, camera_name, recorded_at}`을 부여하면, 재색인 없이 콘솔이 **실제 카메라·날짜·시각**으로 동작한다(절대시각 = recorded_at + 구간오프셋).

---

## 6. 수집(ingestion) 아키텍처 — 실 운영 시 (설계)

실 CCTV는 NVR/ffmpeg가 RTSP를 **시간분할 파일**로 저장 → 그 **완결 파일을 배치 색인**(녹화는 GPU 무관, 색인만 GPU 소비 → 분리).

```
IP카메라 ─RTSP→ ffmpeg -c copy segment(5분, -strftime 1) → CAM03_20260611_093000.mp4
   → [워처] 폴더감시 + "직전 완결 세그먼트만" + SQLite 멱등 큐
   → [단일 GPU 워커] index_video(path, source_started_at, camera_id)   ← 직렬(8GB OOM 방지)
   → 기존 tracker→VLM→event_builder→ChromaDB (+ camera_id·date_local·absolute_ts)
```
- 녹화: **ffmpeg `-c copy` segment**(파일명에 카메라+절대시각) / 다카메라 장기운영은 Frigate 위임.
- 완료판정: **"직전(N-1) 세그먼트만 처리"**(미완성 파일 색인 방지). 처리량: tracker로 정적·사람0 구간 VLM skip.
- **절대시각 백본:** `index_video`에 `source_started_at` 인자 → 사건 `absolute_ts`·`date_local` 저장, 날짜 필터 앵커를 `indexed_at`→`date_local`로. (현재 데모는 cctv_map의 recorded_at로 대체)

---

## 7. 로드맵 / 다음 작업

- [ ] **타임라인 플레이어(시간 점프 재생)** — 카메라별 클립을 임의 타임라인에 이어붙여 "연속 녹화"처럼 만들고, DateTimePicker 시각 선택 → 영상이 그 시각으로 점프 + 클립 자동 연속재생. 이력 관리는 그대로.
- [ ] **07 데이터셋으로 footage 채우기** — AI-Hub 71850(실 CCTV 영상) 다수 클립을 카테고리별 카메라에 추가 색인 → 카메라별 하루 타임라인을 촘촘히.
- [ ] **taxonomy 확장** — 싸움(fight)·침입(intrusion)·군집(crowd) 등 07 카테고리 반영(현재는 normal/unknown으로 떨어짐).
- [ ] **실 수집 파이프라인 구현** — §6 (ffmpeg 녹화 + 워처 + 큐 + 절대시각 백본).
- [ ] **VLM 정식 선정 벤치마크** (`bench_vlm.py`) — i2t 평가로 추려진 **후보 3종(InternVL3-8B · Qwen3-VL-8B · Qwen2.5-VL-7B)** 을, 대표 클립(낙상·싸움·침입·군집)의 대표 구간에 `SEGMENT_EVENT_PROMPT` 로 돌려 **캡션 품질 · event_type · 속도 · peak VRAM** 비교 → 영상 색인용 1종 확정. (현재 채택 = InternVL3, i2t 위험판정 91% 기준. **영상 기준 최종 비교 결과 대기 — AI-Hub 색인 후 실행.**)

---

## 8. 환경 / 모델

- **로컬(운영):** RTX 4060 8GB · `C:\Users\eg287\venvs\3dvision` · torch cu128 · transformers 5.x · bitsandbytes 4bit · chromadb · sentence-transformers · ultralytics(YOLO+ByteTrack)
- **프론트:** node 24 · Vite 6 · React 18
- **모델:** 색인/캡션 VLM = **InternVL3-8B(4bit)** · 임베딩 = **bge-m3**(1024d, 다국어, CPU) · 추적 = **YOLOv8n + ByteTrack**
- **GitHub:** `ev1025/cctv_memory` (행동 메모리) · 생성 라인은 `ev1025/3D-Vision`
