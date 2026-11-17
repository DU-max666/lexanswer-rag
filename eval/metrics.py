"""法衡 · 评测指标与错误分型。

两个核心指标（口径与微调报告一致）：

- **法条引用准确率**：回答中的引用与召回条文回查 EXACT 的比例
  （编号错引 / 内容不匹配 / 幻觉引用均计为不准确）
- **回答准确率**：行为正确（该答答、该拒拒）且引用准确、
  适用场景关键词命中、四段式结构合格的样本占比

错误分型（用于定位改进方向，而非只看总分）：

====================  ==========================================
分型                  判定
====================  ==========================================
编号错引              法名对、条号与召回不符（WRONG_NUMBER）
内容不匹配            条号对、复述内容与条文重合度低（CONTENT_MISMATCH）
适用场景错误          引用对但适用条件未命中金标关键词
法条幻觉              引用了知识库不存在的条文（UNKNOWN）
该拒未拒              金标为拒答但模型给出实体回答
不该拒却拒            金标为可答但模型拒答
格式违规              四段式结构不达标
====================  ==========================================
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

from verify.answer_format import FormatReport, split_sections
from verify.citation_checker import CitationStatus, VerifyVerdict, is_model_refusal

ERROR_NONE = "正确"
ERROR_WRONG_NUMBER = "编号错引"
ERROR_CONTENT_MISMATCH = "内容不匹配"
ERROR_WRONG_SCENARIO = "适用场景错误"
ERROR_HALLUCINATION = "法条幻觉"
ERROR_MISSED_REFUSAL = "该拒未拒"
ERROR_OVER_REFUSAL = "不该拒却拒"
ERROR_FORMAT = "格式违规"

_STATUS_TO_ERROR = {
    CitationStatus.WRONG_NUMBER: ERROR_WRONG_NUMBER,
    CitationStatus.CONTENT_MISMATCH: ERROR_CONTENT_MISMATCH,
    CitationStatus.UNKNOWN: ERROR_HALLUCINATION,
}


@dataclass
class GoldSample:
    """评测集单条金标。"""

    id: str
    question: str
    gold_law: str
    gold_article: int
    expected_behavior: str          # "answer" | "refuse"
    gold_keywords: List[str]        # 适用场景关键词（任一命中即场景正确）
    note: str = ""

    @classmethod
    def from_dict(cls, d: Dict) -> "GoldSample":
        return cls(
            id=d["id"], question=d["question"],
            gold_law=d["gold_law"], gold_article=int(d["gold_article"]),
            expected_behavior=d["expected_behavior"],
            gold_keywords=d.get("gold_keywords", []),
            note=d.get("note", ""),
        )


@dataclass
class SampleScore:
    """单样本评分结果。"""

    id: str
    model: str
    behavior_correct: bool
    citation_correct: bool
    scenario_correct: bool
    format_ok: bool
    error_type: str


def score_prediction(gold: GoldSample, model: str, answer: str,
                     verdict: VerifyVerdict, fmt: FormatReport) -> SampleScore:
    """按「行为→引用→场景→格式」四层判分，首个不达标层记为错误分型。

    注意：拒答与否以**模型自身输出**为准（verify.is_model_refusal），
    引用回查触发的 clarify/refuse 是系统安全网，不计入模型拒答能力。
    """
    model_refused = is_model_refusal(answer)

    if gold.expected_behavior == "refuse":
        behavior_correct = model_refused
        citation_correct = model_refused  # 拒答不产生错误引用
        return SampleScore(gold.id, model, behavior_correct, citation_correct,
                           True, True,
                           ERROR_NONE if behavior_correct else ERROR_MISSED_REFUSAL)

    # 期望回答
    behavior_correct = bool(answer) and not model_refused
    citation_correct = _has_gold_citation(verdict, gold)
    sections = split_sections(answer)
    scenario_text = sections.get("适用条件", "") + sections.get("结论", "") + answer
    scenario_correct = (not gold.gold_keywords) or any(k in scenario_text for k in gold.gold_keywords)
    format_ok = fmt.ok

    error = ERROR_NONE
    if not behavior_correct:
        error = ERROR_OVER_REFUSAL
    elif not citation_correct:
        statuses = {ck.status for ck in verdict.checks}
        error = next((_STATUS_TO_ERROR[s] for s in
                      (CitationStatus.UNKNOWN, CitationStatus.WRONG_NUMBER,
                       CitationStatus.CONTENT_MISMATCH) if s in statuses),
                     ERROR_HALLUCINATION)
    elif not scenario_correct:
        error = ERROR_WRONG_SCENARIO
    elif not format_ok:
        error = ERROR_FORMAT
    return SampleScore(gold.id, model, behavior_correct, citation_correct,
                       scenario_correct, format_ok, error)


def _has_gold_citation(verdict: VerifyVerdict, gold: GoldSample) -> bool:
    """回答中是否存在与金标 (法名, 条号) 一致且 EXACT 的引用。"""
    for ck in verdict.checks:
        c = ck.citation
        if (c.law_name == gold.gold_law and c.article_no == gold.gold_article
                and ck.status == CitationStatus.EXACT):
            return True
    return False


def aggregate(scores: List[SampleScore]) -> Dict[str, float]:
    """汇总：两个准确率 + 拒答正确率 + 错误分型计数。"""
    n = len(scores) or 1
    out: Dict[str, float] = {
        "样本数": len(scores),
        "回答准确率": 100.0 * sum(1 for s in scores if s.behavior_correct
                                  and s.citation_correct and s.scenario_correct
                                  and s.format_ok) / n,
        "法条引用准确率": 100.0 * sum(1 for s in scores if s.citation_correct) / n,
        "行为正确率": 100.0 * sum(1 for s in scores if s.behavior_correct) / n,
    }
    errors = Counter(s.error_type for s in scores if s.error_type != ERROR_NONE)
    out["错误分型"] = dict(errors)  # type: ignore[assignment]
    return out
