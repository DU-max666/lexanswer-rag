"""法衡 · BGE-M3 稠密+稀疏混合向量封装。

生产路径加载 ``BAAI/bge-m3``（FlagEmbedding），一次编码同时得到：

- dense  稠密向量（语义召回）
- sparse 稀疏词权重（词面召回，Milvus ``SPARSE_INVERTED_INDEX``）

裸环境（无 GPU、未下载模型）自动降级为 ``MockHashEmbedder``：
字符二元组哈希向量，接口与真实模型完全一致，保证单测与
数据构造脚本在任意机器可跑通。通过 ``get_embedder()`` 统一获取。
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Dict, List, Optional


class DenseSparseResult:
    """一次编码的稠密+稀疏输出。"""

    def __init__(self, dense: List[List[float]], sparse: List[Dict[int, float]]):
        self.dense = dense
        self.sparse = sparse

    def __len__(self) -> int:
        return len(self.dense)


class EmbeddingProvider:
    """嵌入提供者接口：下游只依赖本接口，不绑定具体模型。"""

    dim: int = 0

    def encode(self, texts: List[str]) -> DenseSparseResult:
        raise NotImplementedError

    def encode_query(self, query: str) -> "DenseSparseResult":
        return self.encode([query])


# ---------------------------------------------------------------------------
# 真实模型：BGE-M3
# ---------------------------------------------------------------------------

class BgeM3Embedder(EmbeddingProvider):
    """BGE-M3 封装；仅在显式配置模型路径且已安装 FlagEmbedding 时启用。"""

    def __init__(self, model_path: str = "BAAI/bge-m3", device: Optional[str] = None):
        from FlagEmbedding import BGEM3FlagModel  # 延迟导入：裸环境不触碰

        kwargs = {"use_fp16": True}
        if device:
            kwargs["device"] = device
        self._model = BGEM3FlagModel(model_path, **kwargs)
        self.dim = 1024

    def encode(self, texts: List[str]) -> DenseSparseResult:
        out = self._model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = [list(map(float, v)) for v in out["dense_vecs"]]
        sparse = [
            {int(k): float(v) for k, v in weights.items()}
            for weights in out["lexical_weights"]
        ]
        return DenseSparseResult(dense, sparse)


# ---------------------------------------------------------------------------
# 本地降级：哈希模拟嵌入
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


class MockHashEmbedder(EmbeddingProvider):
    """确定性模拟嵌入：字符二元组 + 拉丁词哈希分桶。

    不是近似语义模型，仅用于在无模型环境下验证链路正确性；
    向量分布经 L2 归一化，余弦相似度与真实接口行为一致。
    """

    def __init__(self, dim: int = 256, sparse_buckets: int = 65536):
        self.dim = dim
        self.sparse_buckets = sparse_buckets

    @staticmethod
    def _tokens(text: str) -> List[str]:
        """切出字符二元组与拉丁词，作为模拟特征。"""
        cleaned = re.sub(r"\s+", "", text)
        grams = [cleaned[i:i + 2] for i in range(max(len(cleaned) - 1, 0))]
        grams += _TOKEN_RE.findall(text.lower())
        return grams

    def _bucket(self, token: str, mod: int) -> int:
        return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % mod

    def encode(self, texts: List[str]) -> DenseSparseResult:
        dense_rows: List[List[float]] = []
        sparse_rows: List[Dict[int, float]] = []
        for t in texts:
            vec = [0.0] * self.dim
            sparse: Dict[int, float] = {}
            for tok in self._tokens(t):
                vec[self._bucket(tok, self.dim)] += 1.0
                sid = self._bucket(tok, self.sparse_buckets)
                sparse[sid] = sparse.get(sid, 0.0) + 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            dense_rows.append([v / norm for v in vec])
            sparse_rows.append(sparse)
        return DenseSparseResult(dense_rows, sparse_rows)


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------

def get_embedder(model_path: Optional[str] = None) -> EmbeddingProvider:
    """按环境选择嵌入提供者。

    优先级：显式参数 > 环境变量 ``LEXANSWER_BGE_M3_PATH`` > 本地 mock。
    """
    path = model_path or os.environ.get("LEXANSWER_BGE_M3_PATH", "")
    if path:
        try:
            return BgeM3Embedder(path)
        except Exception as exc:  # 依赖缺失或加载失败都降级，不中断链路
            print(f"[embeddings] BGE-M3 不可用（{exc}），降级为 MockHashEmbedder")
    return MockHashEmbedder()
