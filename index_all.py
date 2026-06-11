"""index_all.py — 데모용 다중 CCTV 색인. assets/embedding 의 영상을 outputs/vmem/videos 로
복사하고 ChromaDB 에 색인한다. 각 video_id 는 cctv_meta 가 카메라/날짜를 파생한다(3×3 그리드용).

실행: CUDA_VISIBLE_DEVICES=0 LOAD_IN_4BIT=1 VLM_BACKEND=internvl3 MAX_SEGMENTS=12 MAX_DURATION=90 python index_all.py
"""
import config

import os
import shutil
import traceback

from memory import video_memory

SRC = os.path.join(config.ASSETS_DIR, "embedding")
VID = os.path.join(config.MEMORY_DIR, "videos")
os.makedirs(VID, exist_ok=True)

# 전체 7개 재색인(parse_risk 강건화 반영). chroma 는 실행 전 비워 깨끗이 재생성. 없는 파일은 건너뜀.
TODO = ["fire6", "machine_tipover2", "machine_tipover3", "people-detection",
        "person-bicycle-car-detection", "person_fall3", "person_fall4"]

ok, fail = [], []
for vid in TODO:
    src = os.path.join(SRC, vid + ".mp4")
    if not os.path.exists(src):
        print(f"[skip] 없음: {vid}", flush=True)
        continue
    dst = os.path.join(VID, vid + ".mp4")
    if not os.path.exists(dst):
        shutil.copy(src, dst)
    print(f"\n===== 색인 시작: {vid} =====", flush=True)
    try:
        r = video_memory.index_video(dst, video_id=vid)
        print(f"[done] {vid}: {r}", flush=True)
        ok.append(vid)
    except Exception as e:
        traceback.print_exc()
        print(f"[FAIL] {vid}: {e}", flush=True)
        fail.append(vid)

print(f"\n===== ALL DONE — 성공 {len(ok)}: {ok} / 실패 {len(fail)}: {fail} =====", flush=True)
