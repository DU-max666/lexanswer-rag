"""法衡 · 检索链路包：混合向量、存储、精排、父子回溯、意图闸门、多策略、FAQ 短路。"""
from .embeddings import BgeM3Embedder, EmbeddingProvider, MockHashEmbedder, get_embedder
from .store import Hit, InMemoryStore, MilvusStore, QueryFilter, VectorStore, get_store, rrf_fuse
from .rerank import BgeReranker, LexicalReranker, Reranker, get_reranker
from .parent_child import Evidence, expand_to_parents, render_context
from .intent_gate import Intent, IntentGate, IntentResult, get_intent_gate
from .faq_shortcircuit import BM25Index, FaqMatch, FaqShortCircuit
from .strategies import RetrievalOutcome, RetrievalPipeline, Strategy, select_strategy

__all__ = [
    "BgeM3Embedder", "EmbeddingProvider", "MockHashEmbedder", "get_embedder",
    "Hit", "InMemoryStore", "MilvusStore", "QueryFilter", "VectorStore", "get_store", "rrf_fuse",
    "BgeReranker", "LexicalReranker", "Reranker", "get_reranker",
    "Evidence", "expand_to_parents", "render_context",
    "Intent", "IntentGate", "IntentResult", "get_intent_gate",
    "BM25Index", "FaqMatch", "FaqShortCircuit",
    "RetrievalOutcome", "RetrievalPipeline", "Strategy", "select_strategy",
]
