"""build_timeline.py — 9개 카메라 '24시간 플레이리스트' 생성 (concat 없이).

[왜] 카메라마다 24시간 영상을 실제로 이어붙이면 파일이 수십 GB → 브라우저 사망.
     대신 원본 짧은 클립은 그대로 두고, '어느 클립이 그 카메라 24h 시계의 몇 초에 깔리는지'를
     표로 만든다(cctv_timeline.json). 프론트는 시각→그 클립만 재생, 끝나면 다음으로 자동 전환.

[밀도] 클립을 빈틈없이 이어 채워(continuous) 어느 시각이든 영상이 나오고 스크럽하면 계속 바뀐다.
       단, '사건(이력)'은 클립마다 카메라당 1곳만 event=True 로 표시(하루에 분산) → 이력이 안 터짐.

실행: python -m scripts.build_timeline
"""
import config

import os
import glob
import json
import random

import cv2

VID = os.path.join(config.MEMORY_DIR, "videos")
OUT = os.path.join(config.MEMORY_DIR, "cctv_timeline.json")

DAY = 86400
CAMERAS = 9
LABEL_DATES = ["2026-06-08", "2026-06-09", "2026-06-10", "2026-06-11", "2026-06-12"]
SEED = 20260612   # 고정 시드 → 빌드마다 동일 배치(날짜는 라벨만, 같은 플레이리스트)


def _duration(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    d = (n / fps) if fps else 0
    return round(d, 1) if d else 12.0


def build():
    clips = []
    for p in sorted(glob.glob(os.path.join(VID, "*.mp4"))):
        name = os.path.splitext(os.path.basename(p))[0]
        if "__thermal" in name:
            continue
        clips.append({"video_id": name, "duration": _duration(p)})
    if not clips:
        print("videos/ 에 클립이 없습니다.")
        return

    rng = random.Random(SEED)
    timeline = {"day_seconds": DAY, "loop_label_dates": LABEL_DATES, "cameras": []}
    for ci in range(CAMERAS):
        cam = f"CAM{ci + 1:02d}"
        playlist, off = [], 0
        while off < DAY:                                    # 빈틈없이 이어 채움(셔플 풀 반복)
            pool = clips[:]
            rng.shuffle(pool)
            for c in pool:
                if off >= DAY:
                    break
                playlist.append({"video_id": c["video_id"], "day_offset": off, "duration": c["duration"]})
                off += c["duration"]
        # 사건 표시: 클립마다 1개 배치를 event 로(하루에 분산) → 이력/알림은 이것만 fan-out
        by_clip = {}
        for i, it in enumerate(playlist):
            by_clip.setdefault(it["video_id"], []).append(i)
        for vid, idxs in by_clip.items():
            playlist[rng.choice(idxs)]["event"] = True
        timeline["cameras"].append(
            {"camera_id": cam, "camera_name": cam, "playlist": playlist})   # 이름 없이 CAM 번호만

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(timeline, f, ensure_ascii=False, separators=(",", ":"))
    tot = sum(len(c["playlist"]) for c in timeline["cameras"])
    ev = sum(sum(1 for it in c["playlist"] if it.get("event")) for c in timeline["cameras"])
    print(f"[timeline] {CAMERAS}대, 클립 풀 {len(clips)}개 → 배치 {tot}개(연속), 사건표시 {ev}개 → {OUT}", flush=True)
    for c in timeline["cameras"]:
        e = sum(1 for it in c["playlist"] if it.get("event"))
        print(f"  {c['camera_id']} {c['camera_name']:10} 배치 {len(c['playlist']):4d} / 사건 {e}")


if __name__ == "__main__":
    build()
