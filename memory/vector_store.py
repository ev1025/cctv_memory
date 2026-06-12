"""memory/vector_store.py — ChromaDB 래퍼 (Parent-Child / 구간 단위 저장).

[구조] 저장 단위 = 5초 '구간'(child). 같은 사건은 metadata.event_id 로 묶인다(parent).
       검색·분류는 구간 단위로(의미 희석 없음), 조회(이력/알림/타임라인)는 event_id 로 묶어
       '사건' 단위로 돌려준다 → 다운스트림(app·cctv_meta·프론트)은 그대로.

PersistentClient(HNSW, cosine). 임베딩은 ChromaEmbedder(text_embedder)로 문서·질의에 동일 모델.
"""
import config  # ★ torch 보다 먼저

import os
import threading


_DEFAULT = None
_DEFAULT_LOCK = threading.Lock()


def default_store():
    """프로세스 공유 단일 VectorStore(+임베더). 매 요청 재로드 방지 → 검색/조회를 빠르게."""
    global _DEFAULT
    with _DEFAULT_LOCK:                       # warmup 스레드와 첫 요청의 동시 생성 방지
        if _DEFAULT is None:
            from memory.text_embedder import TextEmbedder
            _DEFAULT = VectorStore(TextEmbedder())
    return _DEFAULT


def _is_special(et):
    return et not in ("normal", "unknown", None)


def group_events(rows):
    """구간 레코드 [(document, metadata)…] → 사건 단위로 묶기.

    사건 = 같은 event_id 구간들. 대표(캡션·유형·썸네일) = 가장 특이·심각한 구간.
    반환 dict 는 기존 '사건' 인터페이스와 동일(키: video_id·start_s·end_s·caption·event_type·
    severity·person_count·dwell_s·has_vehicle·thumb·indexed_at) + seek_s(대표 구간 시각).
    """
    groups = {}
    for d, m in rows:
        eid = m.get("event_id") or m.get("id") or m.get("video_id")
        groups.setdefault(eid, []).append((d, m))
    events = []
    for eid, segs in groups.items():
        best_d, best_m = max(segs, key=lambda dm: (int(dm[1].get("severity") or 0),
                                                   _is_special(dm[1].get("event_type"))))
        m0 = segs[0][1]
        events.append({
            "event_id": eid, "video_id": m0.get("video_id"),
            "start_s": min(s[1].get("start_s", 0) for s in segs),
            "end_s": max(s[1].get("end_s", 0) for s in segs),
            "caption": best_d,
            "event_type": best_m.get("event_type"),
            "severity": max(int(s[1].get("severity") or 0) for s in segs),
            "person_count": m0.get("person_count"), "dwell_s": m0.get("dwell_s"),
            "has_vehicle": m0.get("has_vehicle"), "thumb": best_m.get("thumb"),
            "indexed_at": m0.get("indexed_at"), "seg_count": len(segs),
            "seek_s": best_m.get("start_s"),          # 대표(가장 특이) 구간 시각 = 정확 점프
        })
    return events


class VectorStore:
    def __init__(self, embedder, collection="cargo_cctv"):
        import chromadb
        path = os.path.join(config.MEMORY_DIR, config.CHROMA_SUBDIR)
        os.makedirs(path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=path)
        from memory.text_embedder import ChromaEmbedder
        self.ef = ChromaEmbedder(embedder)
        self.col = self.client.get_or_create_collection(
            collection, embedding_function=self.ef, metadata={"hnsw:space": "cosine"})

    def add(self, records):
        """records: [{id, document, metadata}] → upsert(같은 id 면 덮어씀=멱등)."""
        if not records:
            return 0
        self.col.upsert(
            ids=[r["id"] for r in records],
            documents=[r["document"] for r in records],
            metadatas=[r["metadata"] for r in records])
        return len(records)

    def search(self, query, k=5, where=None):
        """질의 → top-k '구간'(+메타필터). 반환: [{id, document, metadata, score}].

        구간 단위 매칭(의미 희석 없음). 사건 묶기는 호출부(video_memory.query)가 event_id 로.
        """
        n = self.col.count()
        if n == 0:
            return []
        res = self.col.query(query_texts=[query], n_results=min(k, n), where=where or None)
        out = []
        for i in range(len(res["ids"][0])):
            out.append({
                "id": res["ids"][0][i],
                "document": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "score": round(1.0 - res["distances"][0][i], 4),   # cosine 거리 → 유사도
            })
        return out

    def count(self):
        return self.col.count()

    def _rows(self, where=None):
        res = self.col.get(where=where, include=["documents", "metadatas"])
        return list(zip(res["documents"], res["metadatas"]))

    def list_videos(self):
        """색인된 영상별 요약 — [{video_id, segments(=사건 수), types}] (라이브러리 목록용)."""
        if self.col.count() == 0:
            return []
        vids = {}
        for ev in group_events(self._rows()):
            v = vids.setdefault(ev["video_id"],
                                {"video_id": ev["video_id"], "segments": 0, "types": {},
                                 "indexed_at": ev.get("indexed_at")})
            v["segments"] += 1
            et = ev.get("event_type") or "normal"
            v["types"][et] = v["types"].get(et, 0) + 1
        return sorted(vids.values(), key=lambda x: x["video_id"])

    def get_segments(self, video_id):
        """영상의 전체 사건(시간순) — 타임라인용."""
        return sorted(group_events(self._rows(where={"video_id": video_id})),
                      key=lambda s: s["start_s"])

    def all_segments(self):
        """전체 영상의 모든 사건(사건 단위 묶음 — 알림·요약용)."""
        if self.col.count() == 0:
            return []
        return group_events(self._rows())

    def raw_segments(self):
        """모든 활동 '구간'(사건으로 안 묶음) — 이력을 구간 단위로 전부 노출.
        검색도 구간 단위라, 검색 결과가 이력에 그대로 나타나게 하려고 사용."""
        if self.col.count() == 0:
            return []
        out = []
        for d, m in self._rows():
            out.append({
                "event_id": m.get("event_id"), "video_id": m.get("video_id"),
                "start_s": m.get("start_s", 0), "end_s": m.get("end_s", 0),
                "caption": d, "event_type": m.get("event_type") or "normal",
                "severity": int(m.get("severity") or 0),
                "person_count": m.get("person_count"), "dwell_s": m.get("dwell_s"),
                "has_vehicle": m.get("has_vehicle"), "thumb": m.get("thumb"),
                "indexed_at": m.get("indexed_at"),
            })
        return out
