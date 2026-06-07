"""memory/vector_store.py — ChromaDB 래퍼.

구간 이력 저장(upsert 멱등) + 메타필터+벡터 검색을 한 쿼리로. PersistentClient(HNSW, cosine).
임베딩은 ChromaEmbedder(text_embedder)로 문서·질의에 동일 모델을 쓴다.
"""
import config  # ★ torch 보다 먼저

import os


class VectorStore:
    def __init__(self, embedder, collection="cargo_cctv"):
        import chromadb
        path = os.path.join(config.MEMORY_DIR, "chroma")
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
        """질의 → top-k 구간(+메타필터). 반환: [{id, document, metadata, score}]."""
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

    def list_videos(self):
        """색인된 영상별 요약 — [{video_id, segments, risks}] (라이브러리 목록용)."""
        if self.col.count() == 0:
            return []
        res = self.col.get(include=["metadatas"])
        vids = {}
        for m in res["metadatas"]:
            v = vids.setdefault(m["video_id"],
                                {"video_id": m["video_id"], "segments": 0, "risks": {}})
            v["segments"] += 1
            rt = m.get("risk_type", "none")
            v["risks"][rt] = v["risks"].get(rt, 0) + 1
        return sorted(vids.values(), key=lambda x: x["video_id"])

    def get_segments(self, video_id):
        """영상의 전체 구간(시간순) — 타임라인용. [{start_s, end_s, caption, risk_type, severity, thumb}]."""
        res = self.col.get(where={"video_id": video_id}, include=["documents", "metadatas"])
        segs = [{"start_s": m["start_s"], "end_s": m["end_s"], "caption": d,
                 "risk_type": m.get("risk_type"), "severity": m.get("severity"),
                 "thumb": m.get("thumb")}
                for d, m in zip(res["documents"], res["metadatas"])]
        return sorted(segs, key=lambda s: s["start_s"])
