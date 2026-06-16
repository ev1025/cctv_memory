"""config.py — cctv_memory 공통 설정 (영상 행동 메모리: 추적·캡션·분류·검색).

[매우 중요] 이 모듈은 *다른 어떤 모듈(torch/transformers)보다 먼저* import 되어야 한다.
  사용할 GPU(CUDA_VISIBLE_DEVICES)와 HF 캐시 경로(HF_HOME)는 torch import 순간 고정되므로,
  torch import 이전에 os.environ 을 세팅해야 한다. → 모든 모듈의 첫 import 가 `import config`.

  모델 목록(VLM/임베딩 후보)은 models.py 에 분리돼 있다.
"""
import os

# ── 1) 사용할 GPU 고정 — 이 서버(RTX 5090 ×4)에서는 2,3번만 사용 ──────────────
#    device_map="auto" 는 '보이는 GPU' 안에서만 분산하므로 0,1번은 건드리지 않는다.
#    이후 'cuda:0' = 물리 2번, 'cuda:1' = 물리 3번.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2,3")

# ── 2) HF 캐시 위치 — 홈 파티션 용량 절약 위해 대용량 디스크로 ────────────────
_DATA2 = "/workspace/data2"
if os.path.isdir(_DATA2) and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.path.join(_DATA2, "hf_cache")

import torch  # 위 환경변수 세팅 이후 import

# ── 3) 경로 ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")     # 데이터(영상·임베딩 등)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")    # 산출물(vmem: chroma·thumbs)
SAMPLE_IMAGE = os.path.join(ASSETS_DIR, "sample.jpg")  # image_to_text 단독 실행 시 기본 입력(없으면 인자로 지정)

# ── 4) 기본 VLM 백엔드 (models.py 레지스트리 키) ───────────────
VLM_BACKEND = os.environ.get("VLM_BACKEND", "internvl3")

# (i2t 분류 평가·도형 PoC·multi-image 프롬프트는 3d_vision 레포로 이동됨)

# (이상행동 분석 프롬프트 VLM_ANOMALY_PROMPT 는 3d_vision 레포로 이동)
MAX_NEW_TOKENS = 64
MAX_NEW_TOKENS_MULTI = 384   # reasoning(CoT) + JSON 이라 출력이 길어 넉넉히

# VLM 영상 캡션 입력 — 한 구간의 여러 프레임을 '균일 샘플'해 video 로 투입(image_to_text.caption_frames).
#   [측정] InternVL3-8B 4bit @ 8GB, video 경로(토큰=프레임당 ~280, 해상도 무관 — N개 이미지는 8장에 12.5k tok 폭증):
#   4장=6.1GB·2초 / 6장=6.5GB·2초(빠른 한계) / 8장=7.1GB·15초(스필) / 16장=10.9GB·49초(스필).
#   → 8GB 로컬 기본 6장(구간 전체 균일, 옛 2장보다 많고 빠름). 큰 GPU 서버는 VLM_MAX_FRAMES=16.
VLM_MAX_FRAMES = int(os.environ.get("VLM_MAX_FRAMES", "6"))            # 8GB 빠른 한계(스필 없음). 서버는 env 로 16
VLM_FRAME_MAX_SIDE = int(os.environ.get("VLM_FRAME_MAX_SIDE", "560"))  # 프레임 최대 변(px)

# ── 5) dtype / 양자화 — VRAM 효율 ────────────────────────────────────────────
#    bfloat16: fp32 대비 메모리 1/2, RTX 5090 bf16 네이티브. 4bit/8bit 은 작은 GPU용.
TORCH_DTYPE = torch.bfloat16
LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "0") == "1"
LOAD_IN_8BIT = os.environ.get("LOAD_IN_8BIT", "0") == "1"


def build_quant_config():
    """8GB 등 작은 GPU에서 7B VLM 을 올리기 위한 bitsandbytes 설정(없으면 None)."""
    if not (LOAD_IN_4BIT or LOAD_IN_8BIT):
        return None
    from transformers import BitsAndBytesConfig
    if LOAD_IN_4BIT:
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=TORCH_DTYPE,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    return BitsAndBytesConfig(load_in_8bit=True)


# ── video-memory (영상 이력 RAG) ─────────────────────────────────────────────
#    오프라인 mp4 를 구간별 캡션으로 색인(ChromaDB) → 자연어로 검색+답변.
MEMORY_DIR = os.path.join(OUTPUT_DIR, "vmem")        # chroma/(인덱스) + thumbs/ — memory/ 코드 폴더와 이름 분리
SEGMENT_SECONDS = float(os.environ.get("SEGMENT_SECONDS", "5"))   # 고정 그리드 윈도우(초)
SEGMENT_FRAMES = int(os.environ.get("SEGMENT_FRAMES", "16"))       # 구간당 VLM 입력 프레임 수 (기본=권장 16, 로컬 8GB만 env로 낮춤)
CHROMA_SUBDIR = os.environ.get("CHROMA_SUBDIR", "chroma")          # 모델 병렬 비교 시 모델별 별도 chroma
THUMBS_SUBDIR = os.environ.get("THUMBS_SUBDIR", "thumbs")          # 〃 모델별 별도 thumbs
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "bge-m3")         # models.EMBED_REGISTRY 키
# 검색 최소 유사도 — 이 미만은 '무관'으로 보고 결과에서 제외(무의미 질의가 top-k 로 안 뜨게).
SEARCH_MIN_SCORE = float(os.environ.get("SEARCH_MIN_SCORE", "0.52"))

# RAG 답변 프롬프트 — 검색된 구간(시각+캡션) + 대표 프레임을 근거로 답변. .format(question=, context=)
RAG_ANSWER_PROMPT = (
    "질문: {question}\n\n"
    "검색된 CCTV 행동 이력(시각 및 캡션):\n{context}\n\n"
    "위 검색된 이력과 제공된 영상 프레임을 종합하여 질문에 한국어로 답하세요. "
    "단, 답변 시 반드시 검색된 이력을 우선적인 근거로 사용하고, 인용 시 [MM:SS] 형식으로 시각을 표기하세요."
)

# ── 행동 이벤트 메모리 (주차장 CCTV 행동/특이사항 이력) ───────────────────────
#    화재 감지가 아니라 '사람 행동/이벤트' 렌즈로 구간을 분석한다(화재는 별도 열화상 담당).
#    역할 분담: VLM 은 '눈에 보이는 행동/사건'을 분류, 시간 기반 이벤트(배회)는 tracker 의 dwell_s 로 판정.
#    관제 기준 = 활동(activity) 있으면 다 기록, 정적 배경만 임베딩 제외. 위험은 그중 severity/유형으로 알림.

# 배회 판정 — tracker dwell_s 임계(초) + 이동범위(px) 이하면 '머무름'. (실주차장은 60~120s 권장)
LOITER_DWELL_S = float(os.environ.get("LOITER_DWELL_S", "30"))
LOITER_MAX_MOVE_PX = float(os.environ.get("LOITER_MAX_MOVE_PX", "250"))
# 사건 병합 — 연속 구간 간 최대 공백(초). 이 이하로 떨어진 같은 유형/track 구간을 한 사건으로 묶음.
EVENT_MERGE_GAP_S = float(os.environ.get("EVENT_MERGE_GAP_S", str(SEGMENT_SECONDS * 1.5)))

# 구간 묘사 프롬프트 — 캡션(묘사)만 생성. 활동 게이트는 tracker, 유형은 임베딩 분류기가 별도로.
#   체크리스트 누수 방지: 상황을 '예시'로만 녹이고, '없는 것 나열(부정 echo)'을 명시적으로 금지.
SEGMENT_EVENT_PROMPT = (
    "다음 연속 프레임은 같은 5초 구간을 시간순으로 본 것입니다. "
    "이 구간에서 사람과 사물이 실제로 무엇을 하는지 객관적 사실로 한국어로 묘사하세요.\n"
    "기호 없이 순수한 텍스트로 아래 한 줄 형식만 작성하세요.\n\n"
    "사람의 구체적 행동(밀치거나 다투기, 넘어져 쓰러지기, 담을 넘거나 한곳에 오래 머무르기, "
    "좁은 곳에 밀집하기)이나 바닥 침수가 보이면 그것을 캡션 맨 앞에 먼저 적으세요. "
    "보이지 않는 상황을 '없음'이라고 나열하지 말고, 화면에 실제로 보이는 것만 묘사하세요.\n\n"
    "캡션: 장면을 1~2문장으로 구체적으로"
)

# (비교용) 유형 분류까지 시키는 프롬프트 — 중립 묘사형(SEGMENT_EVENT_PROMPT) 대비 '묘사 차이' 확인용.
#   scripts/compare_prompts.py 가 같은 구간에 둘을 돌려, 분류를 시키면 묘사가 어떻게 달라지는지 보여준다.
SEGMENT_CLASSIFY_PROMPT = (
    "이 연속 프레임 영상에서 사람과 사물이 실제로 무엇을 하는지 객관적 사실로 한국어로 묘사하고, 상황 유형을 분류하세요.\n"
    "답변은 기호 없이 아래 두 줄 형식으로만 작성하세요.\n\n"
    "주의사항:\n"
    "1. 보이지 않는 상황을 '없음'이라고 나열하지 말고, 오직 화면에 실제로 보이는 것만 묘사하세요.\n\n"
    "캡션: 장면을 1~2문장으로 구체적으로\n"
    "유형: falldown / fight / invasion / gathering / crowd / flood / normal 중 하나"
)
