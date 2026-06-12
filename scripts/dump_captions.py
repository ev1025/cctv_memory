"""dump_captions.py — 16프레임 재색인 캡션 원문 3모델 비교 덤프.
각 클립의 구간별로 InternVL3 / Qwen3-VL / Qwen2.5-VL 캡션을 나란히 출력(후처리 없음, 원문 그대로)."""
import os
import chromadb

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "vmem")
MODELS = {"InternVL3": "chroma_internvl3_f16", "Qwen3-VL": "chroma_qwen3vl_f16", "Qwen2.5-VL": "chroma_qwen25vl_f16"}

data, clips = {}, set()
for name, sub in MODELS.items():
    path = os.path.join(BASE, sub)
    if not os.path.isdir(path):
        print(f"[skip] {name}: {path} 없음"); continue
    col = chromadb.PersistentClient(path=path).get_collection("cargo_cctv")
    r = col.get(include=["documents", "metadatas"])
    d = {}
    for doc, m in zip(r["documents"], r["metadatas"]):
        key = (m.get("video_id"), int(m.get("start_s") or 0))
        d[key] = (doc, m.get("event_type"))
        clips.add(m.get("video_id"))
    data[name] = d

GT = {"fall": "낙상", "fight": "싸움", "intrusion": "침입", "crowd": "군집", "density": "인파밀집", "flood": "침수"}
for vid in sorted(clips):
    gt = GT.get(vid.split("_")[0], "?")
    print(f"\n{'='*70}\n  {vid}   (실제라벨: {gt})\n{'='*70}")
    starts = sorted(set(k[1] for name in data for k in data[name] if k[0] == vid))
    for st in starts:
        print(f"  ── {st:>3}s ──────────────────────────────────────")
        for name in MODELS:
            if name not in data:
                continue
            cap, et = data[name].get((vid, st), ("—", ""))
            print(f"   {name:11} [{et}]: {cap}")
