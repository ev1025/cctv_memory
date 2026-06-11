"""index_aihub.py — AI-Hub 07(지능형 관제 CCTV) VS 클립 배치 색인 + cctv_map 생성.

카테고리(파일명 코드 프리픽스) = 카메라로 매핑하고, 카메라 내에서 12분 간격으로 녹화시각 부여.
실행: CUDA_VISIBLE_DEVICES=0 LOAD_IN_4BIT=1 VLM_BACKEND=internvl3 MAX_SEGMENTS=10 MAX_DURATION=60 python index_aihub.py
"""
import config

import os
import glob
import json
import traceback
from datetime import datetime, timedelta

from memory import video_memory

VID = os.path.join(config.MEMORY_DIR, "videos")
CODE2CAM = {
    "fall":      ("CAM01", "낙상 감시"),
    "fight":     ("CAM02", "싸움/폭행 감시"),
    "intrusion": ("CAM03", "침입 감시"),
    "crowd":     ("CAM04", "군집 감시"),
    "density":   ("CAM05", "인파밀집 감시"),
    "flood":     ("CAM06", "침수 감시"),
}
BASE_DATE, BASE_HOUR, GAP_MIN = "2026-06-11", 9, 12   # 카메라 하루 09:00 시작, 클립 12분 간격

# ── 1) AI-Hub 클립 목록 → cctv_map.json (카테고리=카메라, 녹화시각 분산) ──
clips = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(VID, "*.mp4"))
               if os.path.basename(p).split("_")[0] in CODE2CAM)
cmap, per_cam = {}, {}
for vid in clips:
    cam, name = CODE2CAM[vid.split("_")[0]]
    i = per_cam.get(cam, 0); per_cam[cam] = i + 1
    t = datetime.fromisoformat(f"{BASE_DATE}T{BASE_HOUR:02d}:00:00") + timedelta(minutes=i * GAP_MIN)
    cmap[vid] = {"camera_id": cam, "camera_name": name, "recorded_at": t.isoformat()}
with open(os.path.join(config.MEMORY_DIR, "cctv_map.json"), "w", encoding="utf-8") as f:
    json.dump(cmap, f, ensure_ascii=False, indent=2)
print(f"[cctv_map] {len(cmap)} clips → 카메라 {sorted(per_cam)}", flush=True)

# ── 2) 배치 색인 (VLM 슬롯 캐시로 1회 로드 후 전 클립 순회) ──
ok, fail = [], []
for n, vid in enumerate(clips, 1):
    print(f"\n===== [{n}/{len(clips)}] {vid} ({cmap[vid]['camera_id']}) =====", flush=True)
    try:
        r = video_memory.index_video(os.path.join(VID, vid + ".mp4"), video_id=vid)
        print(f"[done] {r}", flush=True)
        ok.append(vid)
    except Exception as e:
        traceback.print_exc()
        print(f"[FAIL] {vid}: {e}", flush=True)
        fail.append(vid)
print(f"\n===== 색인 완료 — 성공 {len(ok)} / 실패 {len(fail)}: {fail} =====", flush=True)

# ── 3) 오픈셋 분류 (구간 캡션 → event_classes.json) — 색인과 분리된 단계지만 한 번에 실행 ──
if ok:
    print("\n===== 재분류(오픈셋) =====", flush=True)
    try:
        from scripts.reclassify import main as reclassify
        reclassify()
    except Exception:
        traceback.print_exc()
print("\n===== ALL DONE =====", flush=True)
