"""reports/eval_i2t.py — Image→Text VLM 분류 평가 (위험/이상 데이터셋) → 결과 json 저장.

파일명 prefix 로 정답 라벨(fire/smoke/fall/machine/normal)을 부여하고, 4종 VLM 의
위험 인지·환각·유형 분류를 채점해 모델별 results/eval_parts/<model>.json 에 저장한다.
(병렬 분할 실행 + 통합 build 용. 통합 HTML: python -m reports.build_eval_i2t)

실행:  python -m reports.eval_i2t   (VLM_MODELS env 로 모델 subset 지정 가능)
"""
import config

import os
import glob
import json
from PIL import Image

from image_to_text import VLMCaptioner

VLM_MODELS = os.environ.get("VLM_MODELS", "qwen2-vl,qwen2.5-vl,qwen3-vl,internvl3").split(",")
DATASET = os.environ.get(
    "DATASET_DIR", r"C:\Users\eg287\OneDrive\바탕 화면\project\전시회\vlm\dataset\images")

# 프롬프트는 config 에서 단일 관리(운영/평가 일관) — 도형 PoC 프롬프트와 혼동 방지
RISK_Q = config.I2T_RISK_PROMPT
TYPE_Q = config.I2T_TYPE_PROMPT


def label(fname):
    n = fname.lower()
    if n.startswith("smoke_fire") or n.startswith("fire"):
        return "fire", True
    if n.startswith("smoke"):
        return "smoke", True
    if n.startswith("person_fall"):
        return "fall", True
    if n.startswith("machine_tipover"):
        return "machine", True
    if n.startswith("normal"):
        return "normal", False
    return "unknown", None


def main():
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        paths += glob.glob(os.path.join(DATASET, ext))
    items = []
    for p in sorted(paths):
        fn = os.path.basename(p)
        cat, risk = label(fn)
        if risk is None:
            continue
        items.append((p, fn, cat, risk))
    print(f"데이터셋 {len(items)}장", flush=True)

    parts = os.path.join(config.REPORT_DIR, "eval_parts")
    os.makedirs(parts, exist_ok=True)
    # 메타(이미지 목록·정답·프롬프트) — build 가 이미지별 표를 그릴 때 사용
    with open(os.path.join(parts, "_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"items": [[fn, cat, risk] for _, fn, cat, risk in items],
                   "risk_q": RISK_Q, "type_q": TYPE_Q}, f, ensure_ascii=False)

    for m in VLM_MODELS:
        vlm = VLMCaptioner(m).load()
        rows = []
        for p, fn, cat, risk in items:
            img = Image.open(p).convert("RGB")
            ra = vlm.caption(img, RISK_Q)
            ta = vlm.caption(img, TYPE_Q)
            said_risk = any(k in ra for k in ["예", "네", "있", "위험", "이상"]) and "아니" not in ra
            rows.append([fn, cat, risk, ra.strip(), said_risk == risk, ta.strip(), cat in ta.lower()])
        vlm.unload()
        with open(os.path.join(parts, f"{m}.json"), "w", encoding="utf-8") as f:
            json.dump({"model": m, "rows": rows}, f, ensure_ascii=False)
        print(f"[eval] {m}: 위험 {sum(r[4] for r in rows)}/{len(rows)} · "
              f"유형 {sum(r[6] for r in rows)}/{len(rows)} → json", flush=True)
    print("EVAL_PARTS_DONE")


if __name__ == "__main__":
    main()
