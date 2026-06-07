"""integration_test.py — #1 YOLO 트리거 → VLM 융합 전체 파이프라인 통합 테스트.
실제 사람 영상(people-detection.mp4)을 넣어 1차(YOLO 사람감지) → 2차(VLM 시계열 분석)가
끝까지 도는지 확인. max_triggers 로 2회만(시간 절약)."""
import config

import os
import time

from yolo_trigger import YoloVlmPipeline

VIDEO = os.environ.get("TEST_VIDEO", "people-detection.mp4")
BACKEND = os.environ.get("SMOKE_VLM", "qwen2.5-vl")


def main():
    if not os.path.exists(VIDEO):
        print(f"영상 없음: {VIDEO}")
        return
    print(f"[통합] 영상={VIDEO} VLM={BACKEND}  (YOLO 사람감지 → VLM 시계열)")
    t0 = time.time()
    pipe = YoloVlmPipeline(vlm_backend=BACKEND)
    results = pipe.process_video(VIDEO, sample_fps=1.0, trigger_frames=3, max_triggers=2)
    sec = round(time.time() - t0, 1)

    print(f"\n[통합] 트리거 {len(results)}회 / 총 {sec}s")
    for r in results:
        print(f"  · 프레임 {r['frames']} (사람 {r['persons']}명) → {r['text'][:140]}")
    print("INTEGRATION_DONE")


if __name__ == "__main__":
    main()
