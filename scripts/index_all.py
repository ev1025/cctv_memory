"""index_all.py — 데모용 다중 CCTV 색인. 통합 DATA/videos 의 영상을 ChromaDB 에 색인한다.
각 video_id 는 cctv_meta 가 카메라/날짜를 파생한다(그리드용). 영상은 DATA 에서 바로 읽는다(복사 없음).

실행: CUDA_VISIBLE_DEVICES=0 LOAD_IN_4BIT=1 VLM_BACKEND=internvl3 MAX_SEGMENTS=12 MAX_DURATION=90 python index_all.py
"""
import config

import glob
import os
import traceback

from memory import video_memory

DATA_VIDEOS = os.path.join(os.path.dirname(config.BASE_DIR), "DATA", "videos")

# 색인 대상 video_id(파일명, 확장자 제외). DATA/videos 하위 어디에 있든 재귀로 찾음.
TODO = ["fire6", "machine_tipover2", "machine_tipover3", "people-detection",
        "person-bicycle-car-detection", "person_fall3", "person_fall4"]

ok, fail = [], []
for vid in TODO:
    hits = glob.glob(os.path.join(DATA_VIDEOS, "**", vid + ".mp4"), recursive=True)
    if not hits:
        print(f"[skip] 없음: {vid}", flush=True)
        continue
    src = hits[0]
    print(f"\n===== 색인 시작: {vid} ({src}) =====", flush=True)
    try:
        r = video_memory.index_video(src, video_id=vid)
        print(f"[done] {vid}: {r}", flush=True)
        ok.append(vid)
    except Exception as e:
        traceback.print_exc()
        print(f"[FAIL] {vid}: {e}", flush=True)
        fail.append(vid)

print(f"\n===== ALL DONE — 성공 {len(ok)}: {ok} / 실패 {len(fail)}: {fail} =====", flush=True)
