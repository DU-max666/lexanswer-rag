"""法衡 · 法条版本去重。

律所知识库多年累积的法规文本常见问题：同一部《X法》第N条存在
多个版本（历次修订）、同一附表被重复粘贴。检索层若不去重，
会把废止条文与现行条文同时注入提示词，直接拉低引用准确率。

去重键：``(规范化法名, 条号, 块类型, 款/项号)``。

保留策略（按优先级）：
1. 效力状态：现行有效 > 已被修订 > 草案 > 已废止
2. 版本号字典序（约定版本号为公布日期，如 ``2024-06``）
3. 文本更完整者（长度）

已废止块**不删除**而是移入 ``repealed`` 列表，供微调构造拒答负样本。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .schema import Chunk, ChunkKind, EffectStatus

_STATUS_RANK = {
    EffectStatus.EFFECTIVE: 0,
    EffectStatus.AMENDED: 1,
    EffectStatus.DRAFT: 2,
    EffectStatus.REPEALED: 3,
}


@dataclass
class DedupResult:
    """去重输出：保留块、废止块、被丢弃块及原因。"""

    kept: List[Chunk] = field(default_factory=list)
    repealed: List[Chunk] = field(default_factory=list)
    dropped: List[Tuple[Chunk, str]] = field(default_factory=list)

    def stats(self) -> Dict[str, int]:
        return {
            "kept": len(self.kept),
            "repealed": len(self.repealed),
            "dropped": len(self.dropped),
            "total_input": len(self.kept) + len(self.dropped) + len(self.repealed),
        }


def dedup_key(chunk: Chunk) -> Tuple:
    """同一法条不同版本的对齐键；无条号块按文本哈希对齐。"""
    import hashlib
    if chunk.article_no is not None:
        return (chunk.law_name, chunk.kind.value, chunk.article_no, chunk.clause_no)
    digest = hashlib.sha1(chunk.text.encode("utf-8")).hexdigest()[:12]
    return (chunk.law_name, chunk.kind.value, None, digest)


def _version_sort_key(chunk: Chunk) -> str:
    """版本号约定为日期串（2024-06 形式），空版本排最后。"""
    return chunk.extra.get("version", "") or "0000"


def dedup_chunks(chunks: List[Chunk]) -> DedupResult:
    """执行版本去重。

    父块（ARTICLE/TABLE）与子块（CLAUSE/TABLE_ROW）分别对齐；
    若父块被替换，其 ``child_ids`` 以新父块下实际存在的子块重算。
    """
    groups: Dict[Tuple, List[Chunk]] = defaultdict(list)
    for c in chunks:
        groups[dedup_key(c)].append(c)

    result = DedupResult()
    for key, group in groups.items():
        group.sort(
            key=lambda c: (_STATUS_RANK[c.effect_status], _version_sort_key(c), len(c.text)),
            reverse=False,
        )
        # 同键下若存在现行有效版本，则其余版本视为过期
        best = group[0]
        if best.effect_status == EffectStatus.REPEALED:
            result.repealed.append(best)
            for other in group[1:]:
                result.dropped.append((other, "同条号存在更新版本"))
        else:
            result.kept.append(best)
            for other in group[1:]:
                reason = ("同条号存在更新版本" if other.effect_status != EffectStatus.REPEALED
                          else "同条号存在现行版本，废止版转入负样本池")
                if other.effect_status == EffectStatus.REPEALED:
                    result.repealed.append(other)
                else:
                    result.dropped.append((other, reason))

    # 重算父块 child_ids：只保留实际入库的子块
    kept_ids = {c.chunk_id for c in result.kept}
    for c in result.kept:
        if c.is_parent() and c.child_ids:
            c.child_ids = [cid for cid in c.child_ids if cid in kept_ids]
    return result


def find_conflicts(chunks: List[Chunk]) -> List[Tuple[Chunk, Chunk]]:
    """找出同键但效力状态不同的可疑对，供人工复核报告使用。"""
    groups: Dict[Tuple, List[Chunk]] = defaultdict(list)
    for c in chunks:
        groups[dedup_key(c)].append(c)
    conflicts: List[Tuple[Chunk, Chunk]] = []
    for group in groups.values():
        statuses = {c.effect_status for c in group}
        if len(statuses) > 1:
            group.sort(key=lambda c: _STATUS_RANK[c.effect_status])
            conflicts.append((group[0], group[-1]))
    return conflicts
