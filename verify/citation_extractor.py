"""法衡 · 法条引用抽取。

从模型回答中抽取所有「《法名》第N条（第M款/项）」形式的引用，
供 citation_checker 与召回法条回查比对。

要点：
- 中文数字与阿拉伯数字条号统一归一为 int（复用 corpus.splitters.cn_to_int）
- 记录原文偏移量，便于在回答中高亮定位问题引用
- 只抽「条」级引用；款/项作为附加字段，避免过度解析
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from corpus.splitters import cn_to_int

#: 《法名》第N条[第M款|项]，条号支持中文/阿拉伯数字
_CITE_RE = re.compile(
    r"《([^》]{2,30})》\s*第([零一二三四五六七八九十百千两0-9]+)条"
    r"(?:\s*第([零一二三四五六七八九十两0-9]+)(款|项))?"
)


@dataclass
class Citation:
    """一条法条引用。"""

    law_name: str
    article_no: Optional[int]      # 解析失败为 None（如「第若干条」）
    raw: str                       # 原文匹配串
    start: int                     # 在回答文本中的起始偏移
    end: int
    sub_no: Optional[int] = None   # 款/项号
    sub_kind: Optional[str] = None # "款" | "项"

    def key(self) -> "tuple[str, Optional[int]]":
        return (self.law_name, self.article_no)

    def __str__(self) -> str:
        base = f"《{self.law_name}》第{self.article_no}条"
        if self.sub_no is not None:
            base += f"第{self.sub_no}{self.sub_kind}"
        return base


def extract_citations(text: str) -> List[Citation]:
    """抽取回答全文中的法条引用（按出现顺序）。"""
    citations: List[Citation] = []
    for m in _CITE_RE.finditer(text):
        citations.append(Citation(
            law_name=m.group(1).strip(),
            article_no=cn_to_int(m.group(2)),
            raw=m.group(0),
            start=m.start(),
            end=m.end(),
            sub_no=cn_to_int(m.group(3)) if m.group(3) else None,
            sub_kind=m.group(4),
        ))
    return citations
