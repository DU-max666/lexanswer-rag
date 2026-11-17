"""法衡 · 语料元数据契约。

本模块定义语料治理全链路共享的数据结构：

- ``DocMeta``  文档级元数据（法名、效力状态、敏感级、版本）
- ``Chunk``    切块级元数据（父子指针、引头、块类型）

检索、微调、校验、评测各模块一律以本文件为唯一数据契约，
避免散落在各处的字典键名漂移。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class EffectStatus(str, Enum):
    """法规效力状态，用于检索过滤与拒答判定。"""

    EFFECTIVE = "effective"   # 现行有效
    AMENDED = "amended"       # 已被修订（存在更新版本，本块仅作历史参考）
    REPEALED = "repealed"     # 已废止（检索默认过滤，微调用作拒答负样本）
    DRAFT = "draft"           # 草案 / 未生效


class Sensitivity(str, Enum):
    """数据敏感级，驱动服务侧的模型路由与脱敏策略。"""

    PUBLIC = "public"         # 公开法规，可走云端模型
    INTERNAL = "internal"     # 律所内部知识，仅限内网
    CASE = "case"             # 涉案情材料，案卷不出内网，强制本地模型


class ChunkKind(str, Enum):
    """切块类型：不同文档类型的切分器产出不同的块。"""

    CHAPTER = "chapter"           # 章（仅作父级上下文，一般不单独入库）
    SECTION = "section"           # 节
    ARTICLE = "article"           # 条（父块，带《法名》第N条引头）
    CLAUSE = "clause"             # 款 / 项（子块，命中后回溯父块）
    TABLE = "table"               # 附表整体（父块）
    TABLE_ROW = "table_row"       # 附表行（子块，携带表头语义）
    CASE_PASSAGE = "case_passage" # 类案段落（语义断崖切分产物）
    CONSULT_TURN = "consult_turn" # 咨询记录轮次（问答对）


#: 各块类型在拼装提示词时的展示权重（父块整条注入时用于排序）
KIND_PRIORITY: Dict[ChunkKind, int] = {
    ChunkKind.ARTICLE: 0,
    ChunkKind.TABLE: 1,
    ChunkKind.CASE_PASSAGE: 2,
    ChunkKind.CLAUSE: 3,
    ChunkKind.TABLE_ROW: 4,
    ChunkKind.CONSULT_TURN: 5,
    ChunkKind.CHAPTER: 6,
    ChunkKind.SECTION: 7,
}


def _stable_id(*parts: str) -> str:
    """由若干语义片段生成稳定的短哈希 id，保证重复构造幂等。"""
    joined = "\x1f".join(parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


@dataclass
class DocMeta:
    """文档级元数据。"""

    doc_id: str
    law_name: str                                  # 规范化法名（不含书名号）
    source_path: str = ""                          # 原始文件路径，便于回查
    effect_status: EffectStatus = EffectStatus.EFFECTIVE
    sensitivity: Sensitivity = Sensitivity.PUBLIC
    version: str = ""                              # 版本号 / 公布日期
    issuer: str = ""                               # 制定机关（虚构或公开信息）
    extra: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def normalize_law_name(name: str) -> str:
        """去掉书名号与首尾空白，作为跨文档对齐法名的键。"""
        return name.strip().strip("《》").replace(" ", "")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["effect_status"] = self.effect_status.value
        data["sensitivity"] = self.sensitivity.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocMeta":
        return cls(
            doc_id=data["doc_id"],
            law_name=data["law_name"],
            source_path=data.get("source_path", ""),
            effect_status=EffectStatus(data.get("effect_status", "effective")),
            sensitivity=Sensitivity(data.get("sensitivity", "public")),
            version=data.get("version", ""),
            issuer=data.get("issuer", ""),
            extra=data.get("extra", {}),
        )


@dataclass
class Chunk:
    """切块级元数据；父子指针使「子块命中、父块注入」成为可能。"""

    chunk_id: str
    doc_id: str
    law_name: str
    kind: ChunkKind
    text: str                                       # 块正文（含引头）
    heading: str = ""                               # 引头，如《示例数据保护法》第十条
    article_no: Optional[int] = None                # 条号（整数，便于精确回查）
    clause_no: Optional[int] = None                 # 款/项号
    parent_id: Optional[str] = None                 # 父块指针；None 表示自身即父块
    child_ids: List[str] = field(default_factory=list)
    effect_status: EffectStatus = EffectStatus.EFFECTIVE
    sensitivity: Sensitivity = Sensitivity.PUBLIC
    page: Optional[int] = None                      # 原文页码（PDF 溯源用）
    extra: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def make_id(doc_id: str, kind: ChunkKind, article_no: Optional[int],
                clause_no: Optional[int] = None) -> str:
        """按 文档+类型+条+款 生成稳定 chunk_id。"""
        return _stable_id(doc_id, kind.value, str(article_no), str(clause_no))

    def is_parent(self) -> bool:
        return self.parent_id is None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        data["effect_status"] = self.effect_status.value
        data["sensitivity"] = self.sensitivity.value
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=data["chunk_id"],
            doc_id=data["doc_id"],
            law_name=data["law_name"],
            kind=ChunkKind(data["kind"]),
            text=data["text"],
            heading=data.get("heading", ""),
            article_no=data.get("article_no"),
            clause_no=data.get("clause_no"),
            parent_id=data.get("parent_id"),
            child_ids=data.get("child_ids", []),
            effect_status=EffectStatus(data.get("effect_status", "effective")),
            sensitivity=Sensitivity(data.get("sensitivity", "public")),
            page=data.get("page"),
            extra=data.get("extra", {}),
        )


@dataclass
class RawDoc:
    """加载器产出的原始文档：纯文本 + 元数据，交给切分器消费。"""

    meta: DocMeta
    text: str
    tables: List[str] = field(default_factory=list)  # 抽取出的附表原文块
