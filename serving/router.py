"""法衡 · 数据敏感级路由。

律所核心红线：**案卷不出内网**。路由层在生成请求下发前判定：

1. 查询含涉案情信号（案号、当事人、委托、判决书内容等）
   → 一律本地 7B（vLLM），绝不外发；
2. 检索命中的知识块含 internal/case 敏感级 → 同上；
3. 纯公开法条咨询 → 允许走云端大模型（先 PII 脱敏），
   未配置云端时自然回落本地。

PII 脱敏：手机号、身份证、银行卡、固话按掩码替换，
脱敏后的查询才允许进入云端通道；审计日志记录原始长度而非原文。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

#: 涉案情信号：命中任一即强制本地
CASE_SIGNAL_PATTERNS = [
    r"案号|民初|民终|刑初|行初|执行字",
    r"当事人|委托人|原告|被告|上诉人|被申请人",
    r"判决书|起诉状|答辩状|代理词|仲裁裁决",
    r"我方客户|本所承办|经办法官",
    r"\d{3,}万元?(合同|标的|赔偿)",
]

#: PII 掩码规则（名称: 正则）
PII_PATTERNS = {
    "mobile": re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    "id_card": re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"),
    "bank_card": re.compile(r"(?<!\d)\d{16,19}(?!\d)"),
    "landline": re.compile(r"(?<!\d)0\d{2,3}-?\d{7,8}(?!\d)"),
}


@dataclass
class RouteDecision:
    """路由结论。"""

    target: str            # "local-7b" | "cloud"
    reason: str
    masked_query: str
    pii_hits: List[str]


def detect_case_signals(query: str) -> List[str]:
    """返回命中的涉案情信号模式描述列表。"""
    hits: List[str] = []
    for pat in CASE_SIGNAL_PATTERNS:
        if re.search(pat, query):
            hits.append(pat)
    return hits


def mask_pii(text: str) -> "tuple[str, List[str]]":
    """PII 掩码替换，返回 (脱敏文本, 命中类型列表)。"""
    hits: List[str] = []
    out = text
    for name, pat in PII_PATTERNS.items():
        out, n = pat.subn(lambda m: m.group(0)[:2] + "*" * (len(m.group(0)) - 4) + m.group(0)[-2:], out)
        if n:
            hits.extend([name] * n)
    return out, hits


def route(query: str, retrieved_sensitivities: Optional[Iterable[str]] = None,
          allow_cloud: Optional[bool] = None) -> RouteDecision:
    """核心路由函数。

    参数:
        retrieved_sensitivities: 本次检索命中块的敏感级集合
        allow_cloud: 显式覆盖（测试/降级开关），默认读环境变量
    """
    allow_cloud = allow_cloud if allow_cloud is not None else bool(
        os.environ.get("LEXANSWER_CLOUD_BASE_URL", "")
    )
    case_signals = detect_case_signals(query)
    sensitive_chunks = [
        s for s in (retrieved_sensitivities or []) if s in ("internal", "case")
    ]
    masked, pii_hits = mask_pii(query)
    if case_signals:
        return RouteDecision("local-7b",
                             f"查询含 {len(case_signals)} 类涉案情信号，强制本地模型",
                             masked, pii_hits)
    if sensitive_chunks:
        return RouteDecision("local-7b",
                             f"检索命中 {len(sensitive_chunks)} 个非公开敏感块，强制本地模型",
                             masked, pii_hits)
    if allow_cloud:
        return RouteDecision("cloud", "公开法条咨询且云端通道已配置，脱敏后走云端", masked, pii_hits)
    return RouteDecision("local-7b", "云端通道未配置，默认本地模型", masked, pii_hits)
