"""bench_vlm.py — 영상 행동 색인용 VLM 후보 비교 (InternVL3 vs Qwen3-VL vs Qwen2.5-VL).

대표 클립(카테고리=정답행동)의 대표 구간에 SEGMENT_EVENT_PROMPT 를 돌려
캡션 품질·event_type·속도·VRAM 을 비교 → 영상 색인용 1종 확정 근거.
실행: CUDA_VISIBLE_DEVICES=0 LOAD_IN_4BIT=1 python bench_vlm.py
"""
import config

import os
import time
import torch

from memory import segmenter, video_memory
from image_to_text import VLMCaptioner

VID = os.path.join(config.MEMORY_DIR, "videos")
CANDIDATES = ["internvl3", "qwen3-vl", "qwen2.5-vl"]
CLIPS = {                       # video_id -> 정답(카테고리)
    "fall_E02_041":      "낙상",
    "fight_E03_029":     "싸움",
    "intrusion_E01_053": "침입",
    "crowd_E04_067":     "군집",
}
SEGS_PER_CLIP = 2               # 클립당 비교할 구간 수(가운데 활동 구간)


def pick_segs(vid):
    path = os.path.join(VID, vid + ".mp4")
    if not os.path.exists(path):
        return []
    segs = segmenter.segment(path)
    if not segs:
        return []
    mid = len(segs) // 2
    return (segs[mid:mid + SEGS_PER_CLIP] or segs[:SEGS_PER_CLIP])


def main():
    # 1) 구간 1회 추출(VLM 무관) — 모든 후보가 같은 프레임으로 공정 비교
    clip_segs = {v: pick_segs(v) for v in CLIPS}
    clip_segs = {v: s for v, s in clip_segs.items() if s}
    print(f"[clips] {[(v, len(s)) for v, s in clip_segs.items()]}", flush=True)

    rows = []      # (backend, vid, gt, seg_i, caption, sec)
    vram = {}      # backend -> (load_s, load_gb, peak_gb)
    for backend in CANDIDATES:
        print(f"\n{'='*60}\n[{backend}] 로딩...", flush=True)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            vlm = VLMCaptioner(backend).load()
        except Exception as e:
            print(f"  로드 실패: {str(e)[:120]}", flush=True)
            vram[backend] = ("FAIL", "-", "-")
            continue
        load_s, load_gb = round(time.time() - t0), round(torch.cuda.memory_allocated() / 1e9, 2)
        for vid, segs in clip_segs.items():
            for i, seg in enumerate(segs):
                t = time.time()
                raw = vlm.caption_frames(seg.frames, config.SEGMENT_EVENT_PROMPT)
                sec = round(time.time() - t, 2)
                cap = video_memory.parse_event(raw)
                rows.append((backend, vid, CLIPS[vid], i, cap, sec))
        vram[backend] = (load_s, load_gb, round(torch.cuda.max_memory_allocated() / 1e9, 2))
        vlm.unload()

    # 2) 클립별 비교(정답행동 vs 각 VLM 캡션/유형)
    print("\n\n" + "=" * 70 + "\n===== 클립별 VLM 비교 (정답=카테고리) =====")
    for vid, gt in CLIPS.items():
        if not any(r[1] == vid for r in rows):
            continue
        print(f"\n[{vid} = {gt}]")
        for backend in CANDIDATES:
            for r in [x for x in rows if x[0] == backend and x[1] == vid]:
                print(f"  {backend:11} seg{r[3]} {r[5]:>5}s | {r[4][:60]}")

    # 3) 자원/속도 요약
    print("\n===== 자원·속도 요약 =====")
    print(f"{'VLM':12} {'load_s':>7} {'load_GB':>8} {'peak_GB':>8} {'8GB':>5} {'평균추론':>8}")
    for b in CANDIDATES:
        ls, lg, pg = vram.get(b, ("-", "-", "-"))
        secs = [r[5] for r in rows if r[0] == b]
        avg = round(sum(secs) / len(secs), 2) if secs else "-"
        fit = "O" if isinstance(pg, float) and pg < 8 else ("X" if isinstance(pg, float) else "-")
        print(f"{b:12} {str(ls):>7} {str(lg):>8} {str(pg):>8} {fit:>5} {str(avg):>8}")
    print("\nBENCH_DONE")


if __name__ == "__main__":
    main()
