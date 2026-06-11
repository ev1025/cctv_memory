"""memory/tracker.py — YOLO+ByteTrack 추적 패스 (사람·차량 ID 타임라인).

[왜] '행동 이력'의 핵심은 같은 사람이 언제 들어와 언제 나갔는지·얼마나 머물렀는지다.
     VLM 캡션 전에 가벼운 추적 패스로 track 별 등장/퇴장/체류시간을 '결정적으로' 산출한다.
     → 배회(loitering)는 VLM 판단이 아니라 여기서 나온 dwell_s 로 판정(환각 없음).

[흐름] mp4 → TRACK_FPS 로 균일 샘플 → YOLO.track(persist=True, bytetrack)
       → track_id 별 누적(시각·중심·bbox) → Track(등장~퇴장·체류·이동량)

[비고] 추적은 VLM 과 '별도 패스'다. 8GB 에서는 이 패스(YOLOv8n ~1GB) 를 먼저 돌려 타임라인만
       뽑고 YOLO 를 내린 뒤 VLM 캡션을 올린다(동시 상주 회피).

CLI: python -m memory.tracker <mp4> [--fps 5]
"""
import config  # ★ torch 보다 먼저 (CUDA_VISIBLE_DEVICES/HF_HOME 고정)

import os
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

from video.yolo_trigger import PERSON_CLASS

# COCO 차량류 — '차량 상호작용' 이벤트 판단용
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
TRACK_CLASSES = [PERSON_CLASS] + list(VEHICLE_CLASSES)

TRACK_FPS = float(os.environ.get("TRACK_FPS", "5"))       # 추적 샘플링(초당 N프레임). 5면 YOLOv8n 가벼움
TRACK_CONF = float(os.environ.get("TRACK_CONF", "0.35"))  # YOLO confidence 임계


@dataclass
class Track:
    track_id: int
    cls: int                       # COCO class id
    label: str                     # "person" | "car" | ...
    first_seen_s: float
    last_seen_s: float
    n_obs: int = 0                 # 샘플된 관측 수
    centers: list = field(default_factory=list)   # [(t_s, cx, cy)] 중심 궤적
    boxes: list = field(default_factory=list)     # [(t_s, x1, y1, x2, y2)]

    @property
    def dwell_s(self):
        """등장~퇴장 체류 시간(초). 배회 판정의 핵심 신호."""
        return round(self.last_seen_s - self.first_seen_s, 2)

    @property
    def is_person(self):
        return self.cls == PERSON_CLASS

    def displacement(self):
        """체류 동안 중심 이동 범위(px) — 정지(배회) vs 이동 구분용."""
        if len(self.centers) < 2:
            return 0.0
        xs = [c[1] for c in self.centers]
        ys = [c[2] for c in self.centers]
        return float(np.hypot(max(xs) - min(xs), max(ys) - min(ys)))


def track_video(video_path, fps=None, conf=None, weights="yolov8n.pt"):
    """mp4 → {track_id: Track}. 사람+차량을 ByteTrack 로 추적.

    persist=True 로 프레임 간 ID 를 유지하므로 반드시 시간순 프레임에 순차 호출한다.
    MAX_DURATION(env) 으로 앞부분만 추적(데모/테스트 시간 제한).
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    fps_s = fps or TRACK_FPS
    conf = conf or TRACK_CONF
    from ultralytics import YOLO
    model = YOLO(weights)

    cap = cv2.VideoCapture(video_path)
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(vfps / fps_s)))           # vfps → 목표 샘플링 fps
    _maxdur = float(os.environ.get("MAX_DURATION", "0"))

    tracks = {}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / vfps
        if _maxdur > 0 and t > _maxdur:
            break
        if idx % step == 0:
            res = model.track(frame, persist=True, conf=conf, classes=TRACK_CLASSES,
                              tracker="bytetrack.yaml", verbose=False)[0]
            if res.boxes is not None and res.boxes.id is not None:
                ids = res.boxes.id.int().tolist()
                clss = res.boxes.cls.int().tolist()
                xyxy = res.boxes.xyxy.tolist()
                for tid, c, box in zip(ids, clss, xyxy):
                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    tr = tracks.get(tid)
                    if tr is None:
                        label = "person" if c == PERSON_CLASS else VEHICLE_CLASSES.get(c, str(c))
                        tr = tracks[tid] = Track(tid, c, label, round(t, 2), round(t, 2))
                    tr.last_seen_s = round(t, 2)
                    tr.n_obs += 1
                    tr.centers.append((round(t, 2), round(cx, 1), round(cy, 1)))
                    tr.boxes.append((round(t, 2), round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)))
        idx += 1
    cap.release()
    del model                                    # 8GB: VLM 캡션 전에 YOLO GPU 메모리 반환
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    n_person = sum(t.is_person for t in tracks.values())
    print(f"[tracker] {len(tracks)} track (사람 {n_person} / 차량 {len(tracks) - n_person}), "
          f"{idx / vfps:.1f}s @ {fps_s}fps 샘플", flush=True)
    return tracks


def tracks_in_window(tracks, start_s, end_s):
    """[start_s, end_s] 구간에 활동(겹침)한 track 목록 — segmenter/event_builder 연동용."""
    return [t for t in tracks.values()
            if t.last_seen_s >= start_s and t.first_seen_s <= end_s]


def _main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    path = sys.argv[1]
    fps = None
    if "--fps" in sys.argv:
        fps = float(sys.argv[sys.argv.index("--fps") + 1])
    tracks = track_video(path, fps=fps)
    print(f"\n=== track 타임라인 ({len(tracks)}) ===")
    for t in sorted(tracks.values(), key=lambda x: x.first_seen_s):
        print(f"  #{t.track_id:>3} {t.label:9} "
              f"[{t.first_seen_s:6.1f}s ~ {t.last_seen_s:6.1f}s] "
              f"체류 {t.dwell_s:5.1f}s · 관측 {t.n_obs:3d} · 이동범위 {t.displacement():6.1f}px")


if __name__ == "__main__":
    _main()
