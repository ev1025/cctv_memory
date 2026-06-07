"""anomaly_test.py — #4 이상행동 분석 JSON 출력 검증.
같은 사람 영상에 VLM_ANOMALY_PROMPT 를 주고, VLM 이 JSON 형식을 지키는지 + 파싱되는지 확인.
(이 영상은 평범한 보행이라 결과는 'none/Low' 예상 — 핵심은 JSON 형식·파싱 성공 여부)"""
import config

import os
import re
import json
import time

from yolo_trigger import YoloVlmPipeline

VIDEO = os.environ.get("TEST_VIDEO", "people-detection.mp4")
BACKEND = os.environ.get("SMOKE_VLM", "qwen2.5-vl")


def _extract_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None, "JSON 블록 없음"
    try:
        return json.loads(m.group()), None
    except Exception as e:
        return None, f"파싱 실패: {e}"


def main():
    if not os.path.exists(VIDEO):
        print(f"영상 없음: {VIDEO}")
        return
    print(f"[#4] 영상={VIDEO} VLM={BACKEND}  (이상행동 JSON)")
    t0 = time.time()
    pipe = YoloVlmPipeline(vlm_backend=BACKEND)
    results = pipe.process_video(VIDEO, sample_fps=1.0, trigger_frames=3,
                                 prompt=config.VLM_ANOMALY_PROMPT, max_triggers=2)
    sec = round(time.time() - t0, 1)

    print(f"\n[#4] 트리거 {len(results)}회 / {sec}s")
    ok = 0
    for r in results:
        parsed, err = _extract_json(r["text"])
        print(f"  · 프레임 {r['frames']} (사람 {r['persons']}명)")
        print(f"    원문: {r['text'][:160]}")
        if parsed is not None:
            ok += 1
            print(f"    [JSON OK] risk={parsed.get('risk_level')} type={parsed.get('type')}")
            print(f"    reasoning(CoT): {str(parsed.get('reasoning', parsed.get('reason', '')))[:140]}")
        else:
            print(f"    [JSON 실패] {err}")
    print(f"\n[#4] JSON 파싱 성공 {ok}/{len(results)}")
    print("ANOMALY_JSON_DONE")


if __name__ == "__main__":
    main()
