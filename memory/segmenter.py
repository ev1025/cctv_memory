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


def _seg_times(start_s, end_s, k, peak=None, win=1.2):
    """k 개 샘플 시각. peak(모션 순간) 있으면 그 시각 ±win 창에 집중(동작 전·후), 없으면 균등."""
    if peak is None:
        return [start_s + (end_s - start_s) * (i + 1) / (k + 1) for i in range(k)]
    a, b = max(start_s, peak - win), min(end_s, peak + win)
    if b - a < 0.2:
        a, b = start_s, end_s
    return [a + (b - a) * (i + 0.5) / k for i in range(k)]


def _seg_frames(cap, start_s, end_s, fps, k, peak=None, actors=None):
    """시각별 프레임 추출. actors 주어지면 '동작 창 전체를 덮는 고정 박스'로 확대 크롭 —
    서있던 위치~쓰러진 위치를 모두 포함해, 추적 박스가 잠깐 무너져도 피사체가 화면에 남는다."""
    times = _seg_times(start_s, end_s, k, peak)
    box = _union_box_times(actors, times) if actors else None
    out = []
    for ts in times:
        im = _grab(cap, int(ts * fps))
        if im is None:
            continue
        if box is not None:
            im = _crop(im, box)
        out.append(im)
    return out


def _motion_peak(tracks, t0, t1):
    """구간 [t0,t1] 에서 움직임이 가장 큰 시각 — '중심 속도(스케일 불변) + 박스 종횡비 변화'.
    낙상=세로→가로 종횡비 급변, 싸움=빠른 중심 이동 을 함께 잡는다. 추적 단서 없으면 None(→균등 폴백)."""
    if not tracks:
        return None
    best_t, best_m = None, 0.0
    for tr in tracks.values():
        bxs = [b for b in tr.boxes if t0 <= b[0] <= t1]
        for (ta, ax1, ay1, ax2, ay2), (tb, bx1, by1, bx2, by2) in zip(bxs, bxs[1:]):
            dt = max(tb - ta, 1e-3)
            aw, ah = max(ax2 - ax1, 1.0), max(ay2 - ay1, 1.0)
            bw, bh = max(bx2 - bx1, 1.0), max(by2 - by1, 1.0)
            diag = ((aw + ah + bw + bh) / 2) or 1.0
            dcx, dcy = (bx1 + bx2 - ax1 - ax2) / 2, (by1 + by2 - ay1 - ay2) / 2
            speed = (dcx * dcx + dcy * dcy) ** 0.5 / diag / dt    # 박스크기로 정규화(원근 불변)
            shape = abs(bh / bw - ah / aw)                        # 종횡비 변화(낙상 핵심 신호)
            m = speed + shape
            if m > best_m:
                best_m, best_t = m, (ta + tb) / 2
    return best_t if best_m > 0 else None


def _action_actors(tracks, t0, t1):
    """구간 모션을 주도한 사람 track(피크 부근 고모션, 싸움 대비 최대 3명) — 크롭 대상."""
    scored = []
    for tr in (tracks or {}).values():
        if not tr.is_person:
            continue
        bxs = [b for b in tr.boxes if t0 <= b[0] <= t1]
        best = 0.0
        for (ta, ax1, ay1, ax2, ay2), (tb, bx1, by1, bx2, by2) in zip(bxs, bxs[1:]):
            dt = max(tb - ta, 1e-3)
            aw, ah = max(ax2 - ax1, 1.0), max(ay2 - ay1, 1.0)
            bw, bh = max(bx2 - bx1, 1.0), max(by2 - by1, 1.0)
            diag = ((aw + ah + bw + bh) / 2) or 1.0
            dcx, dcy = (bx1 + bx2 - ax1 - ax2) / 2, (by1 + by2 - ay1 - ay2) / 2
            best = max(best, (dcx * dcx + dcy * dcy) ** 0.5 / diag / dt + abs(bh / bw - ah / aw))
        if best > 0:
            scored.append((best, tr))
    if not scored:
        return []
    scored.sort(reverse=True, key=lambda x: x[0])
    top = scored[0][0]
    return [tr for s, tr in scored if s >= top * 0.5][:3]   # 피크 행위자 + 동급(싸움 상대)


def _union_box_times(actors, times):
    """행위자들의 times 전 시점 박스를 모두 덮는 합집합(고정 크롭 — 동작 전 위치~후 위치 포함)."""
    boxes = [min(tr.boxes, key=lambda bb: abs(bb[0] - ts))[1:]
             for tr in actors if tr.boxes for ts in times]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _crop(im, box, padf=1.8, max_frac=0.6):
    """행위자 박스로 확대 크롭. 박스가 이미 화면을 크게 차지하면(군집/근접) 전체 유지."""
    if box is None:
        return im
    W, H = im.size
    x1, y1, x2, y2 = box
    bw, bh = (x2 - x1) * padf, (y2 - y1) * padf
    if bw > W * max_frac or bh > H * max_frac:    # 행위자가 화면을 크게 차지 → 크롭 불필요
        return im
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = max(bw, bh)
    L, T, R, B = max(0, cx - s / 2), max(0, cy - s / 2), min(W, cx + s / 2), min(H, cy + s / 2)
    return im.crop((int(L), int(T), int(R), int(B)))


def segment(video_path, seconds=None, frames_per_seg=None, tracks=None):
    """mp4 → [Segment…]. tracks 주면 각 구간의 '움직임 큰 순간'에 프레임 집중(모션 가이드), 없으면 균등 그리드."""
    seconds = seconds or config.SEGMENT_SECONDS
    frames_per_seg = frames_per_seg or config.SEGMENT_FRAMES   # 구간당 VLM 입력 프레임 수
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    dur = (total / fps) if fps else 0.0
    _maxdur = float(os.environ.get("MAX_DURATION", "0"))
    if _maxdur > 0 and dur > _maxdur:
        dur = _maxdur                            # 긴 영상은 앞부분만 색인(데모 시간 제한)

    crop_on = config.CROP_ACTORS
    segs, n_motion, n_crop = [], 0, 0
    t = 0.0
    while t < dur:
        e = min(t + seconds, dur)
        peak = _motion_peak(tracks, t, e)        # tracks 없으면 None → 균등
        actors = _action_actors(tracks, t, e) if (crop_on and peak is not None) else None
        frames = _seg_frames(cap, t, e, fps, frames_per_seg, peak=peak, actors=actors)
        if frames:
            segs.append(Segment(round(t, 2), round(e, 2), frames, "motion" if peak is not None else "grid"))
            n_motion += peak is not None
            n_crop += bool(actors)
        t += seconds
    cap.release()
    print(f"[segmenter] {len(segs)} 구간 (모션 {n_motion} / 균등 {len(segs) - n_motion} / 크롭 {n_crop}), "
          f"{frames_per_seg}프레임/구간, {dur:.1f}s", flush=True)
    return segs
