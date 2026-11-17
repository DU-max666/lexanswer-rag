"""法衡 · 语料治理包：加载、切分、去重、元数据契约。"""
from .schema import Chunk, ChunkKind, DocMeta, EffectStatus, RawDoc, Sensitivity
from .loaders import LoaderUnavailable, load_directory, load_document
from .splitters import split_document
from .dedup import DedupResult, dedup_chunks

__all__ = [
    "Chunk", "ChunkKind", "DocMeta", "EffectStatus", "RawDoc", "Sensitivity",
    "LoaderUnavailable", "load_document", "load_directory",
    "split_document", "dedup_chunks", "DedupResult",
]
