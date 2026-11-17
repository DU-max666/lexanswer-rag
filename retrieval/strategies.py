"""法衡 · 多策略检索编排：直接 / HyDE / 子查询 / 回溯。

四种策略的选择逻辑（对应简历项目中的「多策略改写」）：

- ``DIRECT``    首轮召回质量足够：直接精排 + 父块注入
- ``HYDE``      首轮召回为空或低分：生成假设性法条文档再检索，
                把「口语问题」拉回「法条语言」空间
- ``SUBQUERY``  问题含多个子问（并列连词/多问号）：拆分为子查询
                分别检索后 RRF 合并，避免单向量表示稀释多意图
- ``BACKTRACK`` 命中集中在款/项子块：扩大父块回溯窗口并按条号
                二次精确检索，保证整条语境完整

``RetrievalPipeline`` 把意图闸门、FAQ 短路、混合检索、精排、
父块回溯串成一条可离线跑通（全 mock）的链路。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from corpus.schema import ChunkKind

from .embeddings import EmbeddingProvider
from .faq_shortcircuit import FaqMatch, FaqShortCircuit
from .intent_gate import Intent, IntentGate
from .parent_child import Evidence, expand_to_parents, render_context
from .rerank import Reranker
from .store import Hit, QueryFilter, VectorStore, rrf_fuse


class Strategy(str, Enum):
    DIRECT = "direct"
    HYDE = "hyde"
    SUBQUERY = "subquery"
    BACKTRACK = "backtrack"


@dataclass
class RetrievalOutcome:
    """一次检索编排的完整产物，供生成与校验层消费。"""

    query: str
    intent: Intent
    strategy: Strategy
    evidences: List[Evidence] = field(default_factory=list)
    faq_hit: Optional[FaqMatch] = None
    raw_hits: List[Hit] = field(default_factory=list)

    def context(self) -> str:
        if self.faq_hit is not None:
            return f"【FAQ 审定答案】{self.faq_hit.entry.answer}"
        return render_context(self.evidences)


# ---------------------------------------------------------------------------
# 策略选择器
# ---------------------------------------------------------------------------

_MULTI_QUESTION_RE = re.compile(r"[？?；;]|以及|并且|(，)?和(?!解)|另外|同时|还要")
_CHILD_KINDS = {ChunkKind.CLAUSE.value, ChunkKind.TABLE_ROW.value}


def select_strategy(query: str, first_round: List[Hit],
                    low_score_threshold: float = 0.018) -> Strategy:
    """依据首轮召回质量与问题形态选择改写策略。"""
    top = first_round[0].score if first_round else 0.0
    # 多子问优先拆分：拆分后仍可能触发 HyDE，但顺序上先保证覆盖面
    if len(_MULTI_QUESTION_RE.findall(query)) >= 2 or len(query) > 45:
        return Strategy.SUBQUERY
    if not first_round or top < low_score_threshold:
        return Strategy.HYDE
    child_ratio = (
        sum(1 for h in first_round[:8] if h.chunk.kind.value in _CHILD_KINDS)
        / max(len(first_round[:8]), 1)
    )
    if child_ratio >= 0.6:
        return Strategy.BACKTRACK
    return Strategy.DIRECT


def build_hypothetical_doc(query: str) -> str:
    """HyDE 假设文档构造（mock 版）。

    真实实现由本地 7B 模型生成一段「假想法条」；离线链路用
    模板把口语问题改写为法条语体，保持接口一致：
    「处理……的，依照……规定，处……」
    """
    core = re.sub(r"^(请问|想问一下|咨询一下)[，,：:]?\s*", "", query.strip())
    core = core.rstrip("？?。.！!")
    return (
        f"关于{core}的规定。处理{core}情形的，应当依照有关法律法规执行；"
        f"违反规定的，由主管部门责令改正，并可处以相应罚款。{core}"
    )


def decompose_subqueries(query: str, min_len: int = 5) -> List[str]:
    """按问号/分号/并列连词拆分子查询；拆不出两个则原样返回。"""
    parts = re.split(r"[？?；;]|以及|并且|另外|同时", query)
    parts = [p.strip("，, 。") for p in parts if p and len(p.strip("，, 。")) >= min_len]
    return parts if len(parts) >= 2 else [query.strip()]


# ---------------------------------------------------------------------------
# 编排管线
# ---------------------------------------------------------------------------

class RetrievalPipeline:
    """意图闸门 → FAQ 短路 → 混合检索 → 策略改写 → 精排 → 父块回溯。"""

    def __init__(self, embedder: EmbeddingProvider, store: VectorStore,
                 reranker: Reranker, gate: IntentGate,
                 faq: Optional[FaqShortCircuit] = None):
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self.gate = gate
        self.faq = faq

    def run(self, query: str, flt: Optional[QueryFilter] = None,
            top_k: int = 20, max_parents: int = 6) -> RetrievalOutcome:
        flt = flt or QueryFilter()
        intent_result = self.gate.classify(query)
        outcome = RetrievalOutcome(query=query, intent=intent_result.intent,
                                   strategy=Strategy.DIRECT)
        if not intent_result.need_retrieval:
            return outcome
        if intent_result.intent == Intent.FAQ and self.faq is not None:
            hit = self.faq.match(query)
            if hit is not None:
                outcome.faq_hit = hit
                return outcome
        # 首轮混合检索
        first = self._search(query, flt, top_k)
        strategy = select_strategy(query, first)
        outcome.strategy = strategy
        merged: List[Hit] = first
        if strategy == Strategy.HYDE:
            hypo = self._search(build_hypothetical_doc(query), flt, top_k)
            merged = self._merge_rrf(first, hypo, top_k)
        elif strategy == Strategy.SUBQUERY:
            rounds = [first]
            for sub in decompose_subqueries(query)[1:]:
                rounds.append(self._search(sub, flt, top_k))
            merged = self._merge_rrf(*rounds, top_k=top_k)
        elif strategy == Strategy.BACKTRACK:
            # 按命中条号做整条精确二次检索，扩充父块候选
            article_ids = {h.chunk.parent_id or h.chunk.chunk_id for h in first[:8]}
            extra: List[Hit] = []
            for cid in article_ids:
                c = self.store.get_chunk(cid)
                if c is not None and flt.allow(c):
                    extra.extend(self._search(c.heading, flt, top_k=3))
            merged = self._merge_rrf(first, extra, top_k)
        reranked = self.reranker.rerank(query, merged, top_n=max_parents * 2)
        outcome.raw_hits = reranked
        outcome.evidences = expand_to_parents(reranked, self.store, max_parents=max_parents)
        return outcome

    def _search(self, text: str, flt: QueryFilter, top_k: int) -> List[Hit]:
        res = self.embedder.encode_query(text)
        return self.store.search(res.dense[0], res.sparse[0], flt, top_k=top_k)

    @staticmethod
    def _merge_rrf(*rounds: List[Hit], top_k: int) -> List[Hit]:
        """多轮召回按 chunk_id 做 RRF 合并，保留最高分块的引用。"""
        by_id = {}
        for r in rounds:
            for h in r:
                by_id.setdefault(h.chunk.chunk_id, h)
        ranked_lists = [[h.chunk.chunk_id for h in r] for r in rounds if r]
        fused = rrf_fuse(ranked_lists)
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        merged: List[Hit] = []
        for cid, score in ordered:
            h = by_id[cid]
            merged.append(Hit(chunk=h.chunk, score=score))
        return merged
