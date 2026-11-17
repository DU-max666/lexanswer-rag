"""法衡 · 回答结构校验。

微调数据与系统提示词都强制了四段式契约：
**【原文】→【适用条件】→【结论】→【建议】**。
本模块在返回用户前做结构校验，保证下游（律师复核、评测）
可按固定段落解析：

- 四段齐全且顺序正确
- 每段非空、长度在合理区间（原文段允许较长）
- 「建议」段必须包含免责提示词（不构成正式法律意见）

给出 0~1 的 format_score 与可修复问题清单，
供 API 层决定「直接返回 / 追加重生成提示 / 截断告警」。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

SECTIONS = ["原文", "适用条件", "结论", "建议"]
_SECTION_RE = re.compile(r"【(" + "|".join(SECTIONS) + r")】")

#: 各段长度下限/上限（字符）
_LEN_LIMITS = {"原文": (10, 1200), "适用条件": (4, 200), "结论": (4, 200), "建议": (4, 200)}

DISCLAIMER_HINT = "不构成正式法律意见"


@dataclass
class FormatReport:
    ok: bool
    score: float
    missing: List[str] = field(default_factory=list)
    out_of_order: bool = False
    empty_sections: List[str] = field(default_factory=list)
    too_long: List[str] = field(default_factory=list)
    no_disclaimer: bool = False

    def as_dict(self) -> dict:
        return {
            "ok": self.ok, "score": round(self.score, 3),
            "missing": self.missing, "out_of_order": self.out_of_order,
            "empty_sections": self.empty_sections, "too_long": self.too_long,
            "no_disclaimer": self.no_disclaimer,
        }


def split_sections(answer: str) -> "dict[str, str]":
    """按【段名】切分回答为有序字典；段外文本忽略。"""
    parts: "dict[str, str]" = {}
    matches = list(_SECTION_RE.finditer(answer))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(answer)
        parts[m.group(1)] = answer[m.end():end].strip()
    return parts


def check_answer_format(answer: str) -> FormatReport:
    """执行四段式结构校验。"""
    parts = split_sections(answer)
    missing = [s for s in SECTIONS if s not in parts]
    present = [s for s in SECTIONS if s in parts]
    # 顺序检查：按出现位置重排后应与 SECTIONS 一致
    order = [m.group(1) for m in _SECTION_RE.finditer(answer)]
    out_of_order = order != sorted(order, key=SECTIONS.index) if len(set(order)) == len(order) else False
    empty = [s for s in present if not parts[s]]
    too_long = [s for s in present
                if len(parts[s]) > _LEN_LIMITS[s][1]]
    no_disclaimer = ("建议" in parts and DISCLAIMER_HINT not in parts["建议"])

    score = 1.0
    score -= 0.25 * len(missing)
    score -= 0.15 * len(empty)
    score -= 0.1 if out_of_order else 0.0
    score -= 0.05 * len(too_long)
    score -= 0.1 if no_disclaimer else 0.0
    score = max(score, 0.0)
    ok = not missing and not empty and not out_of_order
    return FormatReport(ok=ok, score=score, missing=missing,
                        out_of_order=out_of_order, empty_sections=empty,
                        too_long=too_long, no_disclaimer=no_disclaimer)


REPAIR_PROMPT = (
    "你的上一轮回答不符合格式要求：{issues}。"
    "请严格按【原文】【适用条件】【结论】【建议】四段重新输出，"
    "「建议」段末尾注明“本回复不构成正式法律意见”。"
)


def repair_prompt_for(report: FormatReport) -> str:
    """把结构问题转成可回灌模型的重生成提示。"""
    issues: List[str] = []
    if report.missing:
        issues.append("缺少段落：" + "、".join(f"【{s}】" for s in report.missing))
    if report.empty_sections:
        issues.append("空段落：" + "、".join(report.empty_sections))
    if report.out_of_order:
        issues.append("段落顺序错误")
    if report.no_disclaimer:
        issues.append("【建议】段缺少免责说明")
    return REPAIR_PROMPT.format(issues="；".join(issues) or "无")
