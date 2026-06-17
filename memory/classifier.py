"""memory/classifier.py — 오픈셋 이벤트 분류 (캡션 임베딩 vs 클래스 설명 임베딩).

색인(VLM)과 분리: 영상 캡션을 '설정 가능한 클래스 목록(event_classes.json)'에 임베딩 유사도로 분류.
→ 클래스만 바꾸면 재색인(VLM) 없이 재분류(reclassify.py)만 하면 반영됨. (Twelve Labs 식 오픈셋)

특이 유형 어느 것에도 임계 이상 안 붙으면 'normal'(특이사항 없음).
"""
import config

import os
import json

from memory.text_embedder import TextEmbedder

CLASSES_PATH = os.path.join(config.BASE_DIR, "event_classes.json")
MIN_SCORE = float(os.environ.get("CLASSIFY_MIN_SCORE", "0.6"))   # 이 미만이면 normal. 실제 캡션은 노이즈 많아 0.7보다 낮춤. CLASSIFY_MIN_SCORE 로 튜닝


def load_classes(path=CLASSES_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _cos(a, b):
    return sum(x * y for x, y in zip(a, b))   # 둘 다 정규화 벡터 → 내적 = 코사인


class EventClassifier:
    """캡션(또는 저장된 캡션 임베딩)을 클래스 목록에 임베딩 유사도로 분류."""

    def __init__(self, embedder=None, min_score=MIN_SCORE):
        self.classes = load_classes()
        self.min_score = min_score
        self.emb = (embedder or TextEmbedder()).load()
        self.class_vecs = self.emb.encode([c["desc"] for c in self.classes])

    def classify_vec(self, vec):
        """정규화된 캡션 임베딩 → 최적 클래스 dict(저장 임베딩 재사용 시 재임베딩 X)."""
        sims = [_cos(vec, cv) for cv in self.class_vecs]
        i = max(range(len(sims)), key=lambda j: sims[j])
        if sims[i] < self.min_score:
            return {"event_type": "normal", "label": "정상", "score": round(sims[i], 3)}
        c = self.classes[i]
        return {"event_type": c["key"], "label": c["label"], "score": round(sims[i], 3)}
