"""法衡 · 重排层：BGE-Reranker 精排 + 无模型降级打分。

混合召回（top~50）后交给 cross-encoder 精排取 top~8，
是引用准确率提升的关键一环：向量相似度只保证「相关」，
reranker 才保证「能回答」。
"""
from __future__ import annotations

import os
import re
from typing import List, Optional

from corpus.splitters import cn_to_int

from .store import Hit


class Reranker:
    """重排器接口。"""

    def rerank(self, query: str, hits: List[Hit], top_n: int = 8) -> List[Hit]:
        raise NotImplementedError


class BgeReranker(Reranker):
    """BGE cross-encoder 精排（如 BAAI/bge-reranker-v2-m3）。"""

    def __init__(self, model_path: str = "BAAI/bge-reranker-v2-m3", device: Optional[str] = None):
        from FlagEmbedding import FlagReranker  # 延迟导入

        self._model = FlagReranker(model_path, use_fp16=True, device=device)

    def rerank(self, query: str, hits: List[Hit], top_n: int = 8) -> List[Hit]:
        pairs = [[query, h.chunk.text] for h in hits]
        scores = self._model.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            scores = [scores]
        for h, s in zip(hits, scores):
            h.score = float(s)
        return sorted(hits, key=lambda h: h.score, reverse=True)[:top_n]


class LexicalReranker(Reranker):
    """无模型降级精排：词面重合 + 引头命中 + 效力/类型先验。

    打分构成（0~1）：
    - 0.45 查询词与块文本的字符二元组重合率（包含度，非对称 Jaccard）
    - 0.25 查询中出现「《X》第N条」且块引头一致 → 直接满额
    - 0.20 块类型先验（条父块 > 款子块 > 附表行 > 咨询轮次）
    - 0.10 法名命中查询
    """

    _KIND_PRIOR = {"article": 1.0, "table": 0.8, "clause": 0.7,
                   "case_passage": 0.6, "table_row": 0.5, "consult_turn": 0.4}

    @staticmethod
    def _bigrams(text: str) -> set:
        t = re.sub(r"\s+", "", text)
        return {t[i:i + 2] for i in range(max(len(t) - 1, 0))}

    def rerank(self, query: str, hits: List[Hit], top_n: int = 8) -> List[Hit]:
        q_grams = self._bigrams(query)
        q_article_nos = {cn_to_int(a) for a in re.findall(r"第([零一二三四五六七八九十百千0-9]+)条", query)}
        for h in hits:
            c = h.chunk
            grams = self._bigrams(c.text)
            contain = len(q_grams & grams) / len(q_grams) if q_grams else 0.0
            heading_bonus = 1.0 if (c.article_no is not None and c.article_no in q_article_nos) else 0.0
            kind_bonus = self._KIND_PRIOR.get(c.kind.value, 0.3)
            law_bonus = 1.0 if c.law_name and c.law_name in query else 0.0
            h.score = 0.45 * contain + 0.25 * heading_bonus + 0.20 * kind_bonus + 0.10 * law_bonus
        return sorted(hits, key=lambda h: h.score, reverse=True)[:top_n]


def get_reranker(model_path: Optional[str] = None) -> Reranker:
    """按环境选择精排器；模型不可用则降级词面打分。"""
    path = model_path or os.environ.get("LEXANSWER_RERANKER_PATH", "")
    if path:
        try:
            return BgeReranker(path)
        except Exception as exc:
            print(f"[rerank] Reranker 不可用（{exc}），降级为 LexicalReranker")
    return LexicalReranker()
