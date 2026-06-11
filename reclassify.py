"""reclassify.py — 색인된 사건을 event_classes.json 기준으로 재분류(event_type·severity 갱신).

VLM 안 씀(저장된 캡션 임베딩 ↔ 클래스 설명 임베딩 비교만) → 재색인 불필요.
클래스 목록(event_classes.json)을 바꾼 뒤 이 스크립트만 돌리면 콘솔에 즉시 반영.
실행: python reclassify.py
"""
import config

from memory.vector_store import default_store
from memory.classifier import EventClassifier


def main():
    store = default_store()
    res = store.col.get(include=["documents", "metadatas", "embeddings"])
    ids = res["ids"]
    if not ids:
        print("색인된 사건이 없습니다.")
        return

    clf = EventClassifier()
    new_metas, counts = [], {}
    for doc, meta, emb in zip(res["documents"], res["metadatas"], res["embeddings"]):
        r = clf.classify_vec(list(emb))            # 저장된 임베딩 재사용(재임베딩 X)
        m = dict(meta)
        m["event_type"] = r["event_type"]
        m["severity"] = r["severity"]
        new_metas.append(m)
        counts[r["event_type"]] = counts.get(r["event_type"], 0) + 1

    store.col.update(ids=ids, metadatas=new_metas)
    print(f"재분류 {len(ids)}건 → {dict(sorted(counts.items(), key=lambda x: -x[1]))}", flush=True)
    print("RECLASSIFY_DONE")


if __name__ == "__main__":
    main()
