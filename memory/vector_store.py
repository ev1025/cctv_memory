"""memory/vector_store.py — ChromaDB 래퍼 (구간 단위 저장).

[구조] 저장·검색·조회 단위 = 5초 '구간'. 구간 캡션 원문을 그대로 색인하고(의미 희석 없음),
       유형(event_type)은 분리된 임베딩 분류기(classifier/reclassify)가 메타에 붙인다.

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


def _seg(d, m):
    """구간 레코드 (document, metadata) → 다운스트림 공통 구간 dict."""
    return {
        "video_id": m.get("video_id"),
        "start_s": m.get("start_s", 0), "end_s": m.get("end_s", 0),
        "caption": d, "event_type": m.get("event_type") or "normal",
        "label": m.get("label"), "thumb": m.get("thumb"),
        "indexed_at": m.get("indexed_at"),
    }


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
        """질의 → top-k '구간'(+메타필터). 반환: [{id, document, metadata, score}]."""
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
        """색인된 영상별 요약 — [{video_id, segments(=구간 수), types, indexed_at}] (라이브러리 목록용)."""
        if self.col.count() == 0:
            return []
        vids = {}
        for d, m in self._rows():
            vid = m.get("video_id")
            v = vids.setdefault(vid, {"video_id": vid, "segments": 0, "types": {},
                                      "indexed_at": m.get("indexed_at")})
            v["segments"] += 1
            et = m.get("event_type") or "normal"
            v["types"][et] = v["types"].get(et, 0) + 1
        return sorted(vids.values(), key=lambda x: x["video_id"])

    def get_segments(self, video_id):
        """영상의 전체 구간(시간순) — 타임라인용."""
        return sorted((_seg(d, m) for d, m in self._rows(where={"video_id": video_id})),
                      key=lambda s: s["start_s"])

    def all_segments(self):
        """전체 영상의 모든 구간 — 알림·이력·요약용."""
        if self.col.count() == 0:
            return []
        return [_seg(d, m) for d, m in self._rows()]
