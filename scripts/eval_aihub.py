"""eval_aihub.py — 분류 방식 3-way 비교 평가 (클립 단위).

  A. VLM 직접분류  : 색인 때 VLM 이 SEGMENT_EVENT_PROMPT 로 정한 event_type (고정 taxonomy)
  B. 임베딩 유사도 : 캡션을 event_classes.json 에 임베딩 비교(오픈셋, classifier)
  GT. 실제 라벨     : AI-Hub 클립 카테고리(파일명 프리픽스 = 라벨 출처)

클립별로 A/B 의 '가장 특이한 유형' 예측을 GT 와 비교 → 정확도. (재색인·VLM 불필요)
실행: python eval_aihub.py
"""
import config

from collections import defaultdict

from memory.vector_store import default_store
from memory.classifier import EventClassifier

# AI-Hub 파일명 프리픽스 → GT event_class key (event_classes.json 과 일치)
#   crowd(군집 zip)=Gathering, density(인파밀집 zip)=Crowd, intrusion(침입)=Invasion, fall(쓰러짐)=Falldown
CODE2GT = {
    "fall": "falldown", "fight": "fight", "intrusion": "invasion",
    "crowd": "gathering", "density": "crowd", "flood": "flood",
}


def dominant_A(events):
    """VLM 직접 event_type 중 가장 심각/특이한 것(없으면 normal)."""
    special = [(s, et) for (_, et, s, _) in events if et not in ("normal", "unknown", None)]
    return max(special)[1] if special else "normal"


def dominant_B(clf, events):
    """캡션 임베딩 분류 중 가장 점수 높은 특이 유형(없으면 normal)."""
    best = ("normal", 0.0)
    for (_, _, _, emb) in events:
        r = clf.classify_vec(list(emb))
        if r["event_type"] != "normal" and r["score"] > best[1]:
            best = (r["event_type"], r["score"])
    return best[0]


def main():
    store = default_store()
    res = store.col.get(include=["documents", "metadatas", "embeddings"])
    if not res["ids"]:
        print("색인된 사건이 없습니다.")
        return
    clf = EventClassifier()

    clips = defaultdict(list)   # video_id -> [(caption, A_event_type, severity, emb)]
    for doc, m, emb in zip(res["documents"], res["metadatas"], res["embeddings"]):
        clips[m["video_id"]].append((doc, m.get("event_type"), m.get("severity", 0), emb))
    # AI-Hub 클립만(프리픽스가 카테고리)
    clips = {v: e for v, e in clips.items() if v.split("_")[0] in CODE2GT}

    A_ok = B_ok = 0
    per = defaultdict(lambda: [0, 0, 0])   # gt -> [n, A_ok, B_ok]
    print(f"{'클립':24} {'GT':10} {'A(VLM)':14} {'B(임베딩)':14}")
    print("-" * 70)
    for vid, events in sorted(clips.items()):
        gt = CODE2GT[vid.split("_")[0]]
        a, b = dominant_A(events), dominant_B(clf, events)
        aok, bok = (a == gt), (b == gt)
        A_ok += aok; B_ok += bok
        per[gt][0] += 1; per[gt][1] += aok; per[gt][2] += bok
        print(f"{vid:24} {gt:10} {a:12}{'O' if aok else 'X':>2} {b:12}{'O' if bok else 'X':>2}")

    n = len(clips)
    print(f"\n===== 정확도 (클립 {n}개) =====")
    print(f"  A. VLM 직접   : {A_ok}/{n} = {A_ok/n:.0%}")
    print(f"  B. 임베딩분류 : {B_ok}/{n} = {B_ok/n:.0%}")
    print("\n카테고리별 (n / A맞음 / B맞음):")
    for gt, (cn, a, b) in sorted(per.items()):
        print(f"  {gt:10} n={cn:2}  A={a:2}  B={b:2}")
    print("EVAL_DONE")


if __name__ == "__main__":
    main()
