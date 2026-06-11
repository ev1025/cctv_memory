"""eval_aihub.py — 분류 3-way 비교: A(VLM 직접) vs B(임베딩 오픈셋) vs GT(실제 라벨).

  A  : VLM 이 클립 대표 프레임을 보고 유형을 '직접' 분류 (고정 목록, 클립 단위)
  B  : 구간 캡션 임베딩을 event_classes.json 에 비교 (오픈셋, 구간 단위)
  GT : AI-Hub 클립 카테고리(파일명 프리픽스 = 라벨 출처)

A 는 클립을 한 번에 보므로 짧은 사건(싸움 몇 초)이 묻히기 쉽고(의미 희석),
B 는 구간 단위라 그 순간을 잡는다 → 두 방식의 차이를 정량으로 보여준다.
실행: CUDA_VISIBLE_DEVICES=0 LOAD_IN_4BIT=1 python -m scripts.eval_aihub   (A 가 VLM 써서 GPU 필요)
"""
import config

import os
from collections import defaultdict

from PIL import Image

from memory.vector_store import default_store, _is_special
from memory.classifier import EventClassifier

CODE2GT = {
    "fall": "falldown", "fight": "fight", "intrusion": "invasion",
    "crowd": "gathering", "density": "crowd", "flood": "flood",
}
A_TYPES = ["falldown", "fight", "invasion", "gathering", "crowd", "flood", "normal"]
A_PROMPT = ("다음 CCTV 장면을 아래 유형 중 하나로만 분류하라. 영어 키 하나만 답하라.\n"
            "falldown(쓰러짐) / fight(싸움) / invasion(침입) / gathering(군집) / "
            "crowd(인파밀집) / flood(침수) / normal(특이사항 없음)")


def _resolve_thumb(p):
    if not p:
        return None
    q = p.replace("\\", "/")
    rp = os.path.join(config.MEMORY_DIR, "thumbs", q.split("thumbs/", 1)[1]) if "thumbs/" in q else p
    return rp if os.path.exists(rp) else None


def clip_pred_B(items):
    """B: 구간 분류 [(event_type, severity)] 중 가장 특이·심각한 유형(없으면 normal)."""
    special = [(s, et) for (et, s) in items if _is_special(et)]
    return max(special)[1] if special else "normal"


def vlm_classify_A(vlm, thumbs):
    """A: 클립 대표 프레임을 VLM 이 직접 보고 유형 1개로 분류."""
    frames = [Image.open(t).convert("RGB") for t in thumbs[:6] if t]
    if not frames:
        return "normal"
    raw = (vlm.caption_frames(frames, A_PROMPT) or "").lower()
    return next((t for t in A_TYPES if t in raw), "normal")


def main():
    store = default_store()
    res = store.col.get(include=["metadatas", "embeddings"])
    if not res["ids"]:
        print("색인된 구간이 없습니다.")
        return
    clf = EventClassifier()

    clips = defaultdict(lambda: {"B": [], "thumbs": []})
    for meta, emb in zip(res["metadatas"], res["embeddings"]):
        vid = meta.get("video_id")
        if meta.get("by_tracker"):
            clips[vid]["B"].append((meta.get("event_type"), int(meta.get("severity") or 0)))
            continue
        r = clf.classify_vec(list(emb))                      # B: 구간 임베딩 분류
        clips[vid]["B"].append((r["event_type"], r["severity"]))
        th = _resolve_thumb(meta.get("thumb"))
        if th:
            clips[vid]["thumbs"].append((meta.get("start_s", 0), th))
    clips = {v: d for v, d in clips.items() if v.split("_")[0] in CODE2GT}

    from image_to_text import VLMCaptioner
    vlm = VLMCaptioner(config.VLM_BACKEND).load()            # A: VLM 직접분류용

    A_ok = B_ok = 0
    per = defaultdict(lambda: [0, 0, 0])                     # gt -> [n, A_ok, B_ok]
    print(f"{'클립':22} {'GT':10} {'A(VLM직접)':16} {'B(임베딩)':12}")
    print("-" * 66)
    for vid, d in sorted(clips.items()):
        gt = CODE2GT[vid.split("_")[0]]
        thumbs = [t for _, t in sorted(d["thumbs"])]
        a = vlm_classify_A(vlm, thumbs)
        b = clip_pred_B(d["B"])
        aok, bok = (a == gt), (b == gt)
        A_ok += aok
        B_ok += bok
        per[gt][0] += 1
        per[gt][1] += aok
        per[gt][2] += bok
        print(f"{vid:22} {gt:10} {a:12}{'O' if aok else 'X':>3} {b:10}{'O' if bok else 'X':>2}")
    vlm.unload()

    n = len(clips)
    print(f"\n===== 정확도 (클립 {n}개) =====")
    print(f"  A. VLM 직접분류  : {A_ok}/{n} = {A_ok / n:.0%}")
    print(f"  B. 임베딩 오픈셋 : {B_ok}/{n} = {B_ok / n:.0%}")
    print("\n카테고리별 (n / A맞음 / B맞음):")
    for gt, (cn, a, b) in sorted(per.items()):
        print(f"  {gt:10} n={cn:2}  A={a:2}  B={b:2}")
    print("EVAL_DONE")


if __name__ == "__main__":
    main()
