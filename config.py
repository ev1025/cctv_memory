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
VLM_BACKEND = os.environ.get("VLM_BACKEND", "qwen2-vl")
T2I_BACKEND = os.environ.get("T2I_BACKEND", "sdxl")

# ── Image→Text 분류 평가/운영 프롬프트 (메인) ─────────────────────────────────
#   채점용(4종 모델 비교): 정오가 명확하도록 '예/아니오'·단어로만 답하게 한다.
#   reports/eval_i2t.py 가 이 둘을 그대로 사용. 운영 상세 분석은 VLM_ANOMALY_PROMPT.
I2T_RISK_PROMPT = ("이 이미지에 화재·연기·사람 낙상·기계 전도 같은 위험하거나 이상한 상황이 있습니까? "
                   "'예' 또는 '아니오'로만 답하세요.")
I2T_TYPE_PROMPT = "상황 유형을 fire / smoke / fall / machine / normal 중 하나의 영어 단어로만 답하세요."

# ── [보류] 도형 PoC 용 프롬프트 (t2i/ 격리, 현재 i2t 평가엔 미사용) ───────────
VLM_PROMPT = (
    "이 이미지에 있는 도형들의 앞뒤(겹침) 순서와 위치를 설명하세요. "
    "맨 앞(가장 위에 겹쳐 보이는)에 있는 도형부터 순서대로, "
    "'맨 앞에 원, 그 뒤에 삼각형, 맨 뒤에 사각형이 있습니다' 처럼 한국어 한 문장으로 답하세요. "
    "원, 삼각형, 사각형 중 실제로 보이는 것만 포함하세요."
)
# 여러 프레임(영상)을 한 프롬프트에 넣어 시계열 행동을 추론할 때 쓰는 프롬프트 (#3 multi-image)
VLM_MULTIFRAME_PROMPT = (
    "다음은 시간 순서대로 나열된 연속된 프레임들입니다(앞쪽이 먼저). "
    "프레임 간 변화를 바탕으로 장면 속 사람·사물의 행동 흐름을 시간 순서대로 한국어로 간단히 설명하세요."
)
# #4 화재 위험 이상행동 분석 — XML 구조화 + CoT(reasoning) + Unknown 클래스로 고도화.
# 행동 묘사와 이상행동 판단을 1회 추론으로 통합(reasoning 필드가 행동 묘사 역할).
VLM_ANOMALY_PROMPT = (
    "<system_role>\n"
    "당신은 자동차 운송 선박(PCTC) 내부의 화재 위험 및 이상행동을 감지하는 '해양 안전 감시 AI 전문가'입니다.\n"
    "</system_role>\n\n"
    "<objective>\n"
    "제공된 연속 프레임(시간 순서)을 분석하여 작업자의 이상행동 여부를 판단하고, 지정된 JSON 형식으로만 출력하세요.\n"
    "</objective>\n\n"
    "<context>\n"
    "- 이곳은 차량이 밀집된 선박 내부이므로 조도가 낮거나 화질이 떨어질 수 있습니다.\n"
    "</context>\n\n"
    # ASK-HINT(2025): 추상적 "이상한가?" 대신 세밀한 상호작용·판별 단서를 명시적으로 질의 → 환각↓·정확도↑
    "<detection_cues>\n"
    "추상적으로 '이상한가?'를 묻지 말고, 아래 세밀한 시각 단서를 각각 구체적으로 확인하십시오:\n"
    "- 흡연: 손이 입 근처로 반복적으로 이동하는가? 손끝이나 입에서 연기나 작은 불빛이 보이는가?\n"
    "- 화기: 밝은 불꽃·스파크·용접 섬광이 있는가? 그것이 단순 조명 반사광이나 손전등 불빛과 명확히 구분되는가?\n"
    "- 비인가 출입: 사람이 정해진 통로를 벗어나 차량 사이나 제한 구역으로 들어갔는가?\n"
    "- 사람-사물 상호작용: 사람의 손이 어떤 사물(라이터·공구·전선·차량)과 접촉하는가? 무엇을 들고 있는가?\n"
    "</detection_cues>\n\n"
    "<rules>\n"
    "1. 환각 방지: 오직 화면에 명확히 픽셀로 존재하는 팩트만 서술하십시오. 불확실한 픽셀, 조명 반사광, 그림자를 화기나 연기로 임의 추측하지 마십시오.\n"
    "2. 시각적 단계 추론(Visual CoT): 'reasoning' 필드에 먼저 위 detection_cues 를 기준으로 프레임 간 객체의 움직임과 상태 변화를 시간 순서대로(프레임1→2→3) 관찰하여 적은 후, 그 관찰을 근거로 최종 결론을 도출하십시오.\n"
    "3. 예외 처리: 모션 블러나 조명 부족으로 판단이 불가능한 경우, 억지로 판단하지 말고 risk_level을 \"Unknown\"으로 설정하십시오.\n"
    "4. 출력 제약: 마크다운 코드블록이나 기타 텍스트를 대 포함하지 말고, 오직 파싱 가능한 순수 JSON 객체 단 하나만 출력하십시오.\n"
    "</rules>\n\n"
    "<output_format>\n"
    "{\n"
    "  \"reasoning\": \"프레임 1~3의 관찰 결과 요약 및 위험 요소 판단의 논리적 근거\",\n"
    "  \"risk_level\": \"High\" | \"Low\" | \"Unknown\",\n"
    "  \"type\": \"smoking\" | \"fire\" | \"intrusion\" | \"none\" | \"unidentified\"\n"
    "}\n"
    "</output_format>"
)
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


# ── 6) T2I 공통 설정 (모델별 steps/guidance/variant 는 models.T2I_REGISTRY) ───
T2I_DTYPE = torch.float16   # SD/SDXL 모두 fp16 배포본
T2I_NEGATIVE = ("pattern, seamless pattern, tiled, mosaic, repeated shapes, kaleidoscope, fabric, texture, "
                "busy, cluttered, many small shapes, "
                "photo, realistic, 3d render, shadow, blurry, text, watermark")
SEED = 42  # 재현성

# 한국어 도형 단어 → SD 가 잘 알아듣는 영어 단어 ('직사각형'이 '사각형' 부분문자열 충돌 주의)
SHAPE_KO2EN = {
    "원": "circle", "동그라미": "circle",
    "삼각형": "triangle", "세모": "triangle",
    "사각형": "square", "네모": "square", "정사각형": "square",
    "직사각형": "rectangle",
}

# ── 7) video-memory (영상 이력 RAG) ──────────────────────────────────────────
#    오프라인 mp4 를 구간별 캡션으로 색인(ChromaDB) → 자연어로 검색+답변.
MEMORY_DIR = os.path.join(OUTPUT_DIR, "vmem")        # chroma/(인덱스) + thumbs/ — memory/ 코드 폴더와 이름 분리
SEGMENT_SECONDS = float(os.environ.get("SEGMENT_SECONDS", "5"))   # 고정 그리드 윈도우(초)
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "bge-m3")         # models.EMBED_REGISTRY 키

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
