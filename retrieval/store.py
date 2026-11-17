"""法衡 · 向量存储：Milvus collection 设计与本地内存降级实现。

Milvus collection 设计（单 collection，稠密+稀疏双索引混合检索）：

=================  ====================  ==========================
字段               类型                  说明
=================  ====================  ==========================
chunk_id           VARCHAR 主键          稳定哈希 id
parent_id          VARCHAR               父块指针（子块命中回溯用）
doc_id             VARCHAR               所属文档
law_name           VARCHAR               法名（标量过滤）
article_no         INT64                 条号（-1 表示无）
kind               VARCHAR               块类型
effect_status      VARCHAR               效力状态（默认过滤已废止）
sensitivity        VARCHAR               敏感级（越权过滤）
dense_vector       FLOAT_VECTOR(1024)    BGE-M3 稠密，HNSW/COSINE
sparse_vector      SPARSE_FLOAT_VECTOR   BGE-M3 稀疏，倒排索引
text               VARCHAR(8192)         块正文
=================  ====================  ==========================

未安装 pymilvus / 无 Milvus 服务时，``InMemoryStore`` 以暴力检索
提供完全相同的查询接口（含 RRF 融合），支撑本地单测。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from corpus.schema import Chunk

#: 默认过滤：只召回现行有效条文（废止条文留给拒答判定链路）
DEFAULT_FILTER: Callable[[Chunk], bool] = lambda c: c.effect_status.value == "effective"


@dataclass
class Hit:
    """一次混合检索命中。"""

    chunk: Chunk
    score: float
    dense_rank: int = -1
    sparse_rank: int = -1


@dataclass
class QueryFilter:
    """检索过滤条件，Milvus 表达式与内存谓词共用同一语义。"""

    effect_statuses: List[str] = field(default_factory=lambda: ["effective"])
    kinds: Optional[List[str]] = None
    law_names: Optional[List[str]] = None
    max_sensitivity: str = "case"  # 内存中全放行；由服务层按角色收紧

    def allow(self, chunk: Chunk) -> bool:
        if chunk.effect_status.value not in self.effect_statuses:
            return False
        if self.kinds and chunk.kind.value not in self.kinds:
            return False
        if self.law_names and chunk.law_name not in self.law_names:
            return False
        return True

    def to_milvus_expr(self) -> str:
        status_in = ", ".join(f'"{s}"' for s in self.effect_statuses)
        expr = f"effect_status in [{status_in}]"
        if self.kinds:
            kinds_in = ", ".join(f'"{k}"' for k in self.kinds)
            expr += f" and kind in [{kinds_in}]"
        if self.law_names:
            names_in = ", ".join(f'"{n}"' for n in self.law_names)
            expr += f" and law_name in [{names_in}]"
        return expr


def rrf_fuse(rank_lists: List[List[str]], k: int = 60) -> Dict[str, float]:
    """Reciprocal Rank Fusion：融合多路召回排序，无需调权重。"""
    scores: Dict[str, float] = {}
    for ranks in rank_lists:
        for i, cid in enumerate(ranks):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + i + 1)
    return scores


class VectorStore:
    """存储接口：下游只依赖本接口。"""

    def upsert(self, chunks: List[Chunk], vectors_dense: List[List[float]],
               vectors_sparse: List[Dict[int, float]]) -> None:
        raise NotImplementedError

    def search(self, dense: List[float], sparse: Dict[int, float],
               flt: QueryFilter, top_k: int = 20) -> List[Hit]:
        raise NotImplementedError

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        raise NotImplementedError

    def all_chunks(self) -> List[Chunk]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 本地内存实现（暴力检索，接口与 Milvus 版一致）
# ---------------------------------------------------------------------------

class InMemoryStore(VectorStore):
    """内存向量库：稠密余弦 + 稀疏点积，RRF 融合两路排序。"""

    def __init__(self) -> None:
        self._chunks: Dict[str, Chunk] = {}
        self._dense: Dict[str, List[float]] = {}
        self._sparse: Dict[str, Dict[int, float]] = {}

    def upsert(self, chunks, vectors_dense, vectors_sparse) -> None:
        for c, d, s in zip(chunks, vectors_dense, vectors_sparse):
            self._chunks[c.chunk_id] = c
            self._dense[c.chunk_id] = d
            self._sparse[c.chunk_id] = s

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))  # 向量已归一化

    @staticmethod
    def _dot_sparse(a: Dict[int, float], b: Dict[int, float]) -> float:
        small, big = (a, b) if len(a) <= len(b) else (b, a)
        return sum(w * big.get(k, 0.0) for k, w in small.items())

    def search(self, dense, sparse, flt, top_k=20) -> List[Hit]:
        allowed = [cid for cid, c in self._chunks.items() if flt.allow(c)]
        dense_ranked = sorted(
            allowed, key=lambda cid: self._cosine(dense, self._dense[cid]), reverse=True
        )
        sparse_ranked = sorted(
            allowed, key=lambda cid: self._dot_sparse(sparse, self._sparse[cid]), reverse=True
        )
        fused = rrf_fuse([dense_ranked[: top_k * 3], sparse_ranked[: top_k * 3]])
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        hits: List[Hit] = []
        for cid, score in ordered:
            hits.append(Hit(
                chunk=self._chunks[cid],
                score=score,
                dense_rank=dense_ranked.index(cid) if cid in dense_ranked else -1,
                sparse_rank=sparse_ranked.index(cid) if cid in sparse_ranked else -1,
            ))
        return hits

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self._chunks.get(chunk_id)

    def all_chunks(self) -> List[Chunk]:
        return list(self._chunks.values())


# ---------------------------------------------------------------------------
# Milvus 实现
# ---------------------------------------------------------------------------

class MilvusStore(VectorStore):
    """Milvus 混合检索封装（稠密 HNSW + 稀疏倒排，RRFRanker 融合）。"""

    COLLECTION = "lexanswer_chunks"

    def __init__(self, uri: str = "http://localhost:19530", token: str = "",
                 dim: int = 1024):
        from pymilvus import (  # 延迟导入
            MilvusClient, AnnSearchRequest, RRFRanker, DataType, Function,
        )
        self._AnnSearchRequest = AnnSearchRequest
        self._RRFRanker = RRFRanker
        self._client = MilvusClient(uri=uri, token=token)
        self._dim = dim
        self._ensure_collection(DataType)

    def _ensure_collection(self, DataType) -> None:
        """按文件头注释的 schema 建 collection（幂等）。"""
        if self._client.has_collection(self.COLLECTION):
            return
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=64)
        schema.add_field("parent_id", DataType.VARCHAR, max_length=64)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=128)
        schema.add_field("law_name", DataType.VARCHAR, max_length=256)
        schema.add_field("article_no", DataType.INT64)
        schema.add_field("kind", DataType.VARCHAR, max_length=32)
        schema.add_field("effect_status", DataType.VARCHAR, max_length=16)
        schema.add_field("sensitivity", DataType.VARCHAR, max_length=16)
        schema.add_field("text", DataType.VARCHAR, max_length=8192)
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=self._dim)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
        index = self._client.prepare_index_params()
        index.add_index(field_name="dense_vector", index_type="HNSW", metric_type="COSINE",
                        params={"M": 16, "efConstruction": 200})
        index.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX",
                        metric_type="IP")
        self._client.create_collection(self.COLLECTION, schema=schema, index_params=index)

    def upsert(self, chunks, vectors_dense, vectors_sparse) -> None:
        rows = []
        for c, d, s in zip(chunks, vectors_dense, vectors_sparse):
            rows.append({
                "chunk_id": c.chunk_id,
                "parent_id": c.parent_id or "",
                "doc_id": c.doc_id,
                "law_name": c.law_name,
                "article_no": c.article_no if c.article_no is not None else -1,
                "kind": c.kind.value,
                "effect_status": c.effect_status.value,
                "sensitivity": c.sensitivity.value,
                "text": c.text[:8000],
                "dense_vector": d,
                "sparse_vector": {str(k): v for k, v in s.items()},
            })
        if rows:
            self._client.upsert(collection_name=self.COLLECTION, data=rows)

    def search(self, dense, sparse, flt, top_k=20) -> List[Hit]:
        reqs = [
            self._AnnSearchRequest(
                data=[dense], anns_field="dense_vector",
                param={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=top_k * 2, expr=flt.to_milvus_expr()),
            self._AnnSearchRequest(
                data=[{str(k): v for k, v in sparse.items()}], anns_field="sparse_vector",
                param={"metric_type": "IP"}, limit=top_k * 2, expr=flt.to_milvus_expr()),
        ]
        raw = self._client.hybrid_search(
            collection_name=self.COLLECTION, reqs=reqs,
            ranker=self._RRFRanker(60), limit=top_k,
            output_fields=["chunk_id"],
        )
        hits: List[Hit] = []
        for rank, group in enumerate(raw):
            for doc in group:
                cid = doc["entity"]["chunk_id"] if "entity" in doc else doc["id"]
                chunk = self.get_chunk(cid)
                if chunk is not None:
                    hits.append(Hit(chunk=chunk, score=float(doc.get("distance", 0.0))))
        return hits

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        rows = self._client.get(collection_name=self.COLLECTION, ids=[chunk_id])
        if not rows:
            return None
        r = rows[0]
        return Chunk.from_dict({
            "chunk_id": r["chunk_id"], "doc_id": r["doc_id"], "law_name": r["law_name"],
            "kind": r["kind"], "text": r["text"], "parent_id": r.get("parent_id") or None,
            "article_no": r.get("article_no", -1) if r.get("article_no", -1) >= 0 else None,
            "effect_status": r.get("effect_status", "effective"),
            "sensitivity": r.get("sensitivity", "public"),
        })

    def all_chunks(self) -> List[Chunk]:  # pragma: no cover - 需要在线服务
        raise NotImplementedError("Milvus 全量导出请使用 query_iterator，本地链路无需实现")


def get_store(dim: int = 256) -> VectorStore:
    """按环境选择存储：``LEXANSWER_MILVUS_URI`` 存在则用 Milvus，否则内存。"""
    uri = os.environ.get("LEXANSWER_MILVUS_URI", "")
    if uri:
        try:
            return MilvusStore(uri=uri, token=os.environ.get("LEXANSWER_MILVUS_TOKEN", ""), dim=dim)
        except Exception as exc:
            print(f"[store] Milvus 不可用（{exc}），降级为 InMemoryStore")
    return InMemoryStore()
