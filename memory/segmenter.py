"""memory/segmenter.py — 고정 그리드 구간 분할.

영상 전체를 SEGMENT_SECONDS 그리드로 빠짐없이 잘라, 각 구간에서 SEGMENT_FRAMES 장(시간순)을 뽑는다.
VLM 은 호출하지 않는다 — 구간 목록(Segment)만 반환(색인은 video_memory 가 수행).

[설계] 사람·차량 등장/체류는 tracker(YOLO+ByteTrack)가 전체 영상에서 이미 판정한다(tracks_in_window).
       그래서 구간 분할 단계에서 YOLO 를 또 돌리지 않는다(중복 제거 — 예전엔 구간 중앙 1프레임에
       YOLO 를 재실행해 +2 프레임을 보강했으나, 프레임 수를 키운 지금은 실익이 없고 tracker 와 중복).
       활동(activity) 판정은 VLM 캡션(parse_event)이, person_count 는 tracker 가 담당한다.
"""
import config  # ★ torch 보다 먼저

import os
from dataclasses import dataclass

import cv2
from PIL import Image


@dataclass
class Segment:
    start_s: float
    end_s: float
    frames: list           # list[PIL.Image] — 시간순 대표 프레임
    trigger: str = "grid"  # 항상 grid (YOLO 보강 제거). 필드는 하위호환 위해 유지.
    person_count: int = 0  # 구간 단위는 미사용 — 이벤트 person_count 는 tracker 가 채움.


def _grab(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx))
    ok, fr = cap.read()
    return Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)) if ok else None


def _seg_frames(cap, start_s, end_s, fps, k):
    """구간 [start_s,end_s] 에서 균등하게 k 장(시간순) 추출."""
    idxs = [int((start_s + (end_s - start_s) * (i + 1) / (k + 1)) * fps) for i in range(k)]
    return [f for f in (_grab(cap, i) for i in idxs) if f]


def segment(video_path, seconds=None, frames_per_seg=None):
    """mp4 → [Segment…]. SEGMENT_SECONDS 그리드로 자르고 구간당 SEGMENT_FRAMES 장(균등) 추출."""
    seconds = seconds or config.SEGMENT_SECONDS
    frames_per_seg = frames_per_seg or config.SEGMENT_FRAMES   # 구간당 VLM 입력 프레임 수
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    dur = (total / fps) if fps else 0.0
    _maxdur = float(os.environ.get("MAX_DURATION", "0"))
    if _maxdur > 0 and dur > _maxdur:
        dur = _maxdur                            # 긴 영상은 앞부분만 색인(데모 시간 제한)

    segs = []
    t = 0.0
    while t < dur:
        e = min(t + seconds, dur)
        frames = _seg_frames(cap, t, e, fps, frames_per_seg)
        if frames:
            segs.append(Segment(round(t, 2), round(e, 2), frames, "grid"))
        t += seconds
    cap.release()
    print(f"[segmenter] {len(segs)} 구간 (그리드 {frames_per_seg}프레임/구간), {dur:.1f}s", flush=True)
    return segs
