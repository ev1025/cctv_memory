"""memory/segmenter.py — 하이브리드 구간 분할 (5초 그리드 + YOLO 보강).

고정 그리드(SEGMENT_SECONDS)로 영상 전체를 빠짐없이 커버하고,
YOLO 가 사람을 잡은 구간은 대표 프레임을 더 촘촘히 뽑는다(시계열 단서 강화).
VLM 은 호출하지 않는다 — 구간 목록(Segment)만 반환(색인은 video_memory 가 수행).

[비고] YOLO 가중치(yolov8n.pt)는 최초 1회 자동 다운로드. 실패 시 그리드만으로 폴백.
"""
import config  # ★ torch 보다 먼저

import os
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

PERSON_CLASS = 0   # COCO: person


@dataclass
class Segment:
    start_s: float
    end_s: float
    frames: list           # list[PIL.Image] — 시간순 대표 프레임
    trigger: str = "grid"  # "grid" | "yolo"
    person_count: int = 0


def _grab(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
    ok, fr = cap.read()
    return Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)) if ok else None


def _seg_frames(cap, start_s, end_s, fps, k):
    """구간 [start_s,end_s] 에서 균등하게 k 장(시간순) 추출."""
    idxs = [int((start_s + (end_s - start_s) * (i + 1) / (k + 1)) * fps) for i in range(k)]
    return [f for f in (_grab(cap, i) for i in idxs) if f]


def segment(video_path, seconds=None, frames_per_seg=None, use_yolo=True, yolo_weights="yolov8n.pt"):
    """mp4 → [Segment…]. 사람 잡힌 구간은 frames 를 더 촘촘히(+2장) 뽑고 trigger='yolo'."""
    seconds = seconds or config.SEGMENT_SECONDS
    frames_per_seg = frames_per_seg or config.SEGMENT_FRAMES   # 기본 = env SEGMENT_FRAMES (동작 인식 위해 늘림)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    dur = (total / fps) if fps else 0.0
    _maxdur = float(os.environ.get("MAX_DURATION", "0"))
    if _maxdur > 0 and dur > _maxdur:
        dur = _maxdur                            # 긴 영상은 앞부분만 색인(데모 시간 제한)

    yolo = None
    if use_yolo:
        try:
            from ultralytics import YOLO
            yolo = YOLO(yolo_weights)
        except Exception as e:
            print(f"[segmenter] YOLO 미사용(그리드만): {e}", flush=True)

    segs = []
    t = 0.0
    while t < dur:
        e = min(t + seconds, dur)
        frames = _seg_frames(cap, t, e, fps, frames_per_seg)
        if not frames:
            t += seconds
            continue
        seg = Segment(round(t, 2), round(e, 2), frames, "grid")
        if yolo is not None:
            mid = frames[len(frames) // 2]
            bgr = cv2.cvtColor(np.array(mid), cv2.COLOR_RGB2BGR)
            res = yolo(bgr, conf=0.4, classes=[PERSON_CLASS], verbose=False)[0]
            seg.person_count = len(res.boxes)
            if seg.person_count >= 1:                          # ── YOLO 보강 ──
                seg.trigger = "yolo"
                seg.frames = _seg_frames(cap, t, e, fps, frames_per_seg + 2)  # 더 촘촘히
        segs.append(seg)
        t += seconds
    cap.release()
    print(f"[segmenter] {len(segs)} 구간 (grid {sum(s.trigger=='grid' for s in segs)} / "
          f"yolo {sum(s.trigger=='yolo' for s in segs)}), {dur:.1f}s", flush=True)
    return segs
