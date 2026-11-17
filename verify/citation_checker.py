"""法衡 · 引用回查与拒答判定。

把 citation_extractor 抽出的引用逐条与**本次检索召回的法条**比对：

- ``EXACT``             法名+条号命中，且引用上下文与条文内容重合度达标
- ``WRONG_NUMBER``      法名命中但条号不在召回集合（编号错引）
- ``CONTENT_MISMATCH``  法名+条号命中但引用内容与条文对不上（张冠李戴）
- ``UNKNOWN``           法名/条号在知识库检索不到（幻觉或废止）

判定策略（policy）：
- 全部 EXACT → ``pass``，正常返回
- 存在 UNKNOWN 且置信不足 → ``refuse``，返回标准拒答文案
- 其他问题 → ``clarify``，返回澄清提示并附可引用的正确条文
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from retrieval.parent_child import Evidence

from .citation_extractor import Citation, extract_citations

REFUSAL_ANSWER = (
    "抱歉，本次回答涉及的法条无法在现行有效知识库中回查确认，"
    "为避免误导，我不提供未经核实的法律依据。"
    "建议：1）核对法名与条号；2）联系承办律师确认适用规范。"
)

CLARIFY_TEMPLATE = (
    "您引用的条文与检索结果存在出入（{issues}）。"
    "知识库中可确认的现行条文为：{candidates}。请确认后重新提问。"
)

#: 模型「自我拒答」标志词（区别于回查系统触发的 clarify/refuse）
MODEL_REFUSAL_MARKERS = (
    "未找到可引用条文", "无法给出有依据的结论",
    "无法依据该条文回答", "无法在现行有效知识库中回查确认",
)


def is_model_refusal(answer: str) -> bool:
    """判断回答文本是否为模型主动拒答（微调拒答能力的直接体现）。"""
    return any(m in answer for m in MODEL_REFUSAL_MARKERS)


class CitationStatus(str, Enum):
    EXACT = "exact"
    WRONG_NUMBER = "wrong_number"
    CONTENT_MISMATCH = "content_mismatch"
    UNKNOWN = "unknown"


@dataclass
class CitationCheck:
    citation: Citation
    status: CitationStatus
    confidence: float
    matched_evidence: Optional[str] = None  # 命中的证据引头


@dataclass
class VerifyVerdict:
    """整体校验结论。"""

    action: str                       # pass | clarify | refuse
    checks: List[CitationCheck] = field(default_factory=list)
    final_answer: Optional[str] = None  # 非 pass 时给出的替代文案

    @property
    def citation_accuracy(self) -> float:
        if not self.checks:
            return 0.0
        ok = sum(1 for c in self.checks if c.status == CitationStatus.EXACT)
        return ok / len(self.checks)


def _bigrams(text: str) -> set:
    t = re.sub(r"\s+", "", text)
    return {t[i:i + 2] for i in range(max(len(t) - 1, 0))}


def _context_overlap(citation: Citation, answer: str, evidence_text: str) -> float:
    """引用后 60 字与条文正文的二元组包含度（模型复述原文的保真度）。"""
    tail = answer[citation.end:citation.end + 60]
    if not tail:
        return 1.0  # 引用即结尾，不惩罚
    g1, g2 = _bigrams(tail), _bigrams(evidence_text)
    if not g1:
        return 0.0
    return len(g1 & g2) / len(g1)


def build_evidence_index(evidences: List[Evidence]) -> Dict[Tuple[str, Optional[int]], Evidence]:
    """(法名, 条号) → 证据 的索引；无条号证据不参与回查。"""
    index: Dict[Tuple[str, Optional[int]], Evidence] = {}
    for ev in evidences:
        if ev.parent.article_no is not None:
            index[(ev.parent.law_name, ev.parent.article_no)] = ev
    return index


def check_answer(answer: str, evidences: List[Evidence],
                 overlap_threshold: float = 0.35,
                 refuse_confidence: float = 0.3) -> VerifyVerdict:
    """回答级校验：抽取→回查→判定。"""
    citations = extract_citations(answer)
    if not citations:
        # 无引用回答（闲聊/FAQ/拒答）不在本层拦截
        return VerifyVerdict(action="pass", checks=[])
    index = build_evidence_index(evidences)
    laws_in_kb = {law for (law, _) in index}
    checks: List[CitationCheck] = []
    for c in citations:
        ev = index.get(c.key())
        if ev is not None:
            overlap = _context_overlap(c, answer, ev.parent.text)
            status = (CitationStatus.EXACT if overlap >= overlap_threshold
                      else CitationStatus.CONTENT_MISMATCH)
            checks.append(CitationCheck(c, status, overlap, ev.parent.heading))
        elif c.law_name in laws_in_kb:
            checks.append(CitationCheck(c, CitationStatus.WRONG_NUMBER, 0.0))
        else:
            checks.append(CitationCheck(c, CitationStatus.UNKNOWN, 0.0))

    # 同一 (法名,条号) 多处出现时取最优状态：
    # 只要有一处忠实复述即视为引用正确，避免「结论段再次提及」被误判
    severity = {CitationStatus.EXACT: 0, CitationStatus.CONTENT_MISMATCH: 1,
                CitationStatus.WRONG_NUMBER: 2, CitationStatus.UNKNOWN: 3}
    best: Dict[Tuple, CitationCheck] = {}
    for ck in checks:
        k = ck.citation.key()
        if k not in best or severity[ck.status] < severity[best[k].status]:
            best[k] = ck
    checks = list(best.values())

    bad = [ck for ck in checks if ck.status != CitationStatus.EXACT]
    if not bad:
        return VerifyVerdict(action="pass", checks=checks)
    unknown = [ck for ck in bad if ck.status == CitationStatus.UNKNOWN]
    if unknown and all(ck.confidence < refuse_confidence for ck in unknown):
        return VerifyVerdict(action="refuse", checks=checks, final_answer=REFUSAL_ANSWER)
    issues = "、".join(f"{ck.citation.raw}（{ck.status.value}）" for ck in bad)
    candidates = "；".join(sorted({ev.parent.heading for ev in evidences})) or "（无）"
    return VerifyVerdict(
        action="clarify", checks=checks,
        final_answer=CLARIFY_TEMPLATE.format(issues=issues, candidates=candidates),
    )
