"""
config.py — 3D Vision PoC 공통 설정 (Image→Text→Image 파이프라인)

[매우 중요] 이 모듈은 *다른 어떤 모듈(torch/transformers/diffusers)보다 먼저* import 되어야 한다.
  사용할 GPU(CUDA_VISIBLE_DEVICES)와 HF 캐시 경로(HF_HOME)는 torch import 순간 고정되므로,
  torch import 이전에 os.environ 을 세팅해야 한다. → 모든 모듈의 첫 import 가 `import config`.

  모델 목록(VLM/T2I 후보)은 models.py 에 분리돼 있다.
"""
import os

# ── 1) 사용할 GPU 고정 — 이 서버(RTX 5090 ×4)에서는 2,3번만 사용 ──────────────
#    device_map="auto" 는 '보이는 GPU' 안에서만 분산하므로 0,1번은 건드리지 않는다.
#    이후 'cuda:0' = 물리 2번, 'cuda:1' = 물리 3번.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2,3")
VISIBLE = os.environ["CUDA_VISIBLE_DEVICES"].split(",")

# ── 2) HF 캐시 위치 — 홈 파티션 용량 절약 위해 대용량 디스크로 ────────────────
_DATA2 = "/workspace/data2"
if os.path.isdir(_DATA2) and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.path.join(_DATA2, "hf_cache")

import torch  # 위 환경변수 세팅 이후 import

# ── 3) 경로 ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")     # 입력(테스트) 이미지
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")    # 생성 결과 이미지
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")  # HTML 리포트 결과물 (outputs 하위 — reports/ 코드 폴더와 분리)
SAMPLE_IMAGE = os.path.join(ASSETS_DIR, "sample_all3.png")  # 단일 실행(main.py) 기본 입력

# ── 4) 단일 실행(main.py) 기본 백엔드 (models.py 레지스트리 키) ───────────────
VLM_BACKEND = os.environ.get("VLM_BACKEND", "internvl3")

# (i2t 분류 평가·도형 PoC·multi-image 프롬프트는 3d_vision 레포로 이동됨)

# (이상행동 분석 프롬프트 VLM_ANOMALY_PROMPT 는 3d_vision 레포로 이동)
MAX_NEW_TOKENS = 64
MAX_NEW_TOKENS_MULTI = 384   # reasoning(CoT) + JSON 이라 출력이 길어 넉넉히

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
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "bge-m3")         # models.EMBED_REGISTRY 키
# 전 구간 무조건 임베딩 X — VLM 이 '특이사항(위험/이상)'으로 판단한 구간만 임베딩(정상 구간 제외).
EMBED_NOTABLE_ONLY = os.environ.get("EMBED_NOTABLE_ONLY", "1") == "1"
# 검색 최소 유사도 — 이 미만은 '무관'으로 보고 결과에서 제외(무의미 질의가 top-k 로 안 뜨게).
SEARCH_MIN_SCORE = float(os.environ.get("SEARCH_MIN_SCORE", "0.52"))

# 구간 색인용 '통합' 분석 프롬프트 — 1회 추론으로 캡션+라벨을 함께. memory.video_memory.parse_risk 가 파싱.
SEGMENT_RISK_PROMPT = (
    "다음 연속 프레임(같은 구간, 시간순)을 분석해 정확히 아래 4줄 형식으로만 답하세요.\n"
    "번호(1.), 마크다운(*, #), 영어 문장을 절대 쓰지 말고 반드시 한국어로 답하세요.\n\n"
    "설명: 장면에서 무슨 일이 일어나는지 한국어 한 문장\n"
    "위험: 있음 또는 없음\n"
    "유형: fire, smoke, fall, machine, none 중 영어 한 단어\n"
    "심각도: 0, 1, 2, 3 중 숫자 하나"
)
# RAG 답변 프롬프트 — 검색된 구간(시각+캡션) + 대표 프레임을 근거로 답변. .format(question=, context=)
RAG_ANSWER_PROMPT = (
    "질문: {question}\n\n"
    "다음은 영상에서 검색된 관련 구간들입니다(시각·캡션):\n{context}\n\n"
    "위 구간 정보와 함께 제공된 대표 프레임을 근거로 질문에 한국어로 답하세요. "
    "답변에는 근거가 된 시각을 [MM:SS] 형식으로 인용하세요."
)

# ── 8) 행동 이벤트 메모리 (주차장 CCTV 행동/특이사항 이력) ─────────────────────
#    화재 위험 렌즈(SEGMENT_RISK_PROMPT)를 '사람 행동/이벤트' 렌즈로 확장.
#    역할 분담: VLM 은 '눈에 보이는 행동/사건'을 분류, 시간 기반 이벤트(배회)는 tracker 의 dwell_s 로 판정.
#    관제 기준 = 활동(activity) 있으면 다 기록, 정적 배경만 임베딩 제외. 위험은 그중 severity/유형으로 알림.
EVENT_TYPES = ("fall", "vehicle_interaction", "smoking", "flammable", "normal", "unknown")  # VLM 이 부여
EVENT_TYPES_ALL = EVENT_TYPES + ("loitering",)   # loitering 은 tracker(체류시간)에서 부여

# 배회 판정 — tracker dwell_s 임계(초) + 이동범위(px) 이하면 '머무름'. (실주차장은 60~120s 권장)
LOITER_DWELL_S = float(os.environ.get("LOITER_DWELL_S", "30"))
LOITER_MAX_MOVE_PX = float(os.environ.get("LOITER_MAX_MOVE_PX", "250"))
# 사건 병합 — 연속 구간 간 최대 공백(초). 이 이하로 떨어진 같은 유형/track 구간을 한 사건으로 묶음.
EVENT_MERGE_GAP_S = float(os.environ.get("EVENT_MERGE_GAP_S", str(SEGMENT_SECONDS * 1.5)))

# 구간 행동 분석 프롬프트 — 1회 추론으로 캡션+활동+이벤트유형을 JSON 으로. memory.video_memory.parse_event 가 파싱.
SEGMENT_EVENT_PROMPT = (
    "<role>CCTV 영상을 관찰해 장면을 사실대로 기록하는 관제 AI.</role>\n"
    "<task>아래 연속 프레임(같은 구간, 시간순)을 보고, 사람과 차량이 무엇을 하는지를 한국어로 구체적으로 묘사하라. "
    "특이하거나 위험해 보이는 행동(예: 다툼·넘어짐·밀집·무단진입·침수 등 무엇이든)이 있으면 분명히 포함하라.</task>\n"
    "<rules>1. 화면에 실제로 보이는 사실만 기술. 추측·과장하지 말고 조명 반사·그림자를 임의 해석하지 말 것.\n"
    "2. 특정 유형으로 '분류'하려 하지 말고, 무슨 일이 일어나는지 그대로 묘사하라(유형 분류는 별도 단계에서 함).\n"
    "3. 코드블록·여분 텍스트 없이 순수 JSON 객체 단 하나만 출력.</rules>\n"
    "<output_format>\n"
    "{\n"
    '  "caption": "사람·차량이 무엇을 하는지 + 특이행동을 한국어 1~2문장으로 구체적으로",\n'
    '  "activity": true              // 사람이나 움직임이 있으면 true, 아무도 없는 정적 배경이면 false\n'
    "}\n"
    "</output_format>"
)
