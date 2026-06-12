"""dump_captions_json.py — 16프레임 재색인 캡션을 비교 페이지용 JSON 으로.
clips[].segments[] 에 구간(start_s)별 3모델 캡션/유형을 정렬해 담는다. outputs/vmem/captions_f16.json"""
import os
import json
import chromadb

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "vmem")
MODELS = {"internvl3": "chroma_internvl3_f16", "qwen3vl": "chroma_qwen3vl_f16", "qwen25vl": "chroma_qwen25vl_f16"}
LABELS = {"internvl3": "InternVL3-8B", "qwen3vl": "Qwen3-VL-8B", "qwen25vl": "Qwen2.5-VL-7B"}
GT = {"fall": "낙상", "fight": "싸움", "intrusion": "침입", "crowd": "군집", "density": "인파밀집", "flood": "침수"}

clips = {}
for key, sub in MODELS.items():
    path = os.path.join(BASE, sub)
    if not os.path.isdir(path):
        print(f"[skip] {key}: {path}"); continue
    col = chromadb.PersistentClient(path=path).get_collection("cargo_cctv")
    r = col.get(include=["documents", "metadatas"])
    for doc, m in zip(r["documents"], r["metadatas"]):
        vid = m.get("video_id")
        st = int(m.get("start_s") or 0)
        c = clips.setdefault(vid, {"video_id": vid, "gt": GT.get(vid.split("_")[0], "?"), "_segs": {}})
        seg = c["_segs"].setdefault(st, {"start_s": st, "end_s": int(m.get("end_s") or st + 5)})
        seg[key] = {"cap": doc, "type": m.get("event_type")}

out = []
for vid in sorted(clips):
    c = clips[vid]
    c["segments"] = [c["_segs"][k] for k in sorted(c["_segs"])]
    del c["_segs"]
    out.append(c)

dst = os.path.join(BASE, "captions_f16.json")
json.dump({"models": MODELS and [{"key": k, "label": LABELS[k]} for k in MODELS], "clips": out},
          open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"wrote {dst} — {len(out)} clips")
