"""法衡 · 父子块回溯：子块命中，父块整条注入。

向量检索命中「《X法》第五条第（二）项」这类细粒度子块时，
若直接把子块塞进提示词，模型会缺失同条其余款项的语境，
导致「内容不匹配」型引用错误。本模块负责：

1. 将命中块沿 ``parent_id`` 回溯到父块（整条/整表）
2. 同一父块的多个子块命中合并，避免重复注入
3. 拼装「引头 + 整条原文 + 款项列表」的证据文本
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from corpus.schema import Chunk, ChunkKind

from .store import Hit, VectorStore


@dataclass
class Evidence:
    """一条注入模型的证据（父块粒度）。"""

    parent: Chunk
    hit_children: List[Chunk] = field(default_factory=list)
    score: float = 0.0

    def context_text(self) -> str:
        """拼装证据文本：引头 + 父块全文 + 命中款项标注。

        父块正文本身已含《法名》第N条引头，仅在缺失时补一行，避免重复。
        """
        lines = []
        head = self.parent.heading or f"《{self.parent.law_name}》"
        status = ""
        if self.parent.effect_status.value != "effective":
            status = f"（效力状态：{self.parent.effect_status.value}）"
        if not self.parent.text.startswith(head):
            lines.append(f"{head}{status}")
        lines.append(self.parent.text)
        if self.hit_children:
            marks = "、".join(
                f"第{c.clause_no}项" if c.clause_no is not None else c.chunk_id[:6]
                for c in self.hit_children
            )
            lines.append(f"[用户问题命中的款项：{marks}]")
        return "\n".join(lines)


def expand_to_parents(hits: List[Hit], store: VectorStore,
                      max_parents: int = 6) -> List[Evidence]:
    """命中块回溯父块并去重合并。

    - 命中父块自身（整条/整表）：直接作为证据
    - 命中子块：回溯父块，子块记入 ``hit_children``
    - 父块缺失（数据异常）：退化为使用子块文本自证
    """
    bucket: Dict[str, Evidence] = {}
    order: List[str] = []
    for hit in hits:
        c = hit.chunk
        parent_id = c.parent_id if c.parent_id else c.chunk_id
        if parent_id not in bucket:
            parent = store.get_chunk(parent_id) or c
            bucket[parent_id] = Evidence(parent=parent, score=hit.score)
            order.append(parent_id)
        ev = bucket[parent_id]
        ev.score = max(ev.score, hit.score)
        if c.chunk_id != parent_id and c not in ev.hit_children:
            ev.hit_children.append(c)
    ordered = sorted((bucket[p] for p in order), key=lambda e: e.score, reverse=True)
    return ordered[:max_parents]


def render_context(evidences: List[Evidence]) -> str:
    """证据列表 → 提示词「检索依据」段落。"""
    blocks = []
    for i, ev in enumerate(evidences, 1):
        blocks.append(f"【依据{i}】\n{ev.context_text()}")
    return "\n\n".join(blocks)
