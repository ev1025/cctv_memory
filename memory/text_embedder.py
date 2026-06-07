"""memory/text_embedder.py — 다국어 텍스트 임베딩 래퍼.

VLMCaptioner 와 동형(load/encode/unload)으로 슬롯 관리. CPU 로드 기본(VLM 8GB VRAM 경쟁 회피).
ChromaDB 의 EmbeddingFunction(ChromaEmbedder)으로 감싸 문서·질의에 '동일' 모델을 쓴다.
"""
import config  # ★ torch 보다 먼저 (GPU 격리/HF 캐시)

import os
import gc

import models


class TextEmbedder:
    """sentence-transformers 다국어 임베딩 (load/encode/unload)."""

    def __init__(self, backend=None):
        self.backend = backend or config.EMBED_BACKEND
        if self.backend not in models.EMBED_REGISTRY:
            raise ValueError(
                f"unknown embed backend '{self.backend}'. available: {list(models.EMBED_REGISTRY)}")
        self.model_id = models.EMBED_REGISTRY[self.backend]["id"]
        self.model = None

    def load(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            device = os.environ.get("EMBED_DEVICE", "cpu")   # CPU 기본(VLM 과 VRAM 경쟁 회피)
            print(f"[EMBED] 로딩: {self.backend} ({self.model_id}) device={device}", flush=True)
            self.model = SentenceTransformer(self.model_id, device=device)
        return self

    def encode(self, texts):
        """str → list[float] / list[str] → list[list[float]] (정규화 벡터)."""
        if self.model is None:
            self.load()
        single = isinstance(texts, str)
        arr = self.model.encode([texts] if single else list(texts), normalize_embeddings=True)
        return arr[0].tolist() if single else arr.tolist()

    def unload(self):
        self.model = None
        gc.collect()


class ChromaEmbedder:
    """ChromaDB EmbeddingFunction 어댑터 — 문서/질의에 동일 TextEmbedder 사용.

    chromadb 가 add/query 시 자동 호출(__call__). name() 은 컬렉션 메타에 박혀,
    재오픈 시 임베딩 함수 불일치를 감지하는 키가 된다.
    """

    def __init__(self, embedder: TextEmbedder):
        self._e = embedder

    def __call__(self, input):          # chromadb: Documents -> Embeddings (legacy)
        return self._e.encode(list(input))

    def embed_documents(self, input):   # chromadb 1.x: 문서 색인
        return self._e.encode(list(input))

    def embed_query(self, input):       # chromadb 1.x: 질의 검색
        return self._e.encode(list(input))

    def name(self):
        return f"textembedder-{self._e.backend}"
