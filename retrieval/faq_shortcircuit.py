"""法衡 · FAQ BM25 阈值短路。

律所 80% 的高频提问其实是事务性问题（委托流程、收费标准、
材料清单）。这类问题答案稳定、措辞要求低，用 BM25 在 FAQ
小库上直接命中并返回人工审定答案，不进向量链路：

- 命中条件一：top1 得分 >= ``min_score``
- 命中条件二：top1 与 top2 分差 >= ``min_margin``（防歧义）

BM25 为纯 Python 实现（Okapi BM25，k1=1.5, b=0.75），
中文按字符二元组 + 拉丁词切分，无任何第三方依赖。
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def tokenize(text: str) -> List[str]:
    """CJK 字符二元组 + 拉丁词/数字，兼顾中文与编号类查询。"""
    cleaned = re.sub(r"\s+", "", text)
    grams = [cleaned[i:i + 2] for i in range(max(len(cleaned) - 1, 0))]
    latin = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return grams + latin


class BM25Index:
    """Okapi BM25 内存索引。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self._doc_len: List[int] = []
        self._docs: List[str] = []
        self._avg_len = 0.0

    def add(self, docs: List[str]) -> None:
        for text in docs:
            tokens = tokenize(text)
            tf = Counter(tokens)
            idx = len(self._docs)
            self._docs.append(text)
            self._doc_len.append(len(tokens))
            for term, freq in tf.items():
                self._postings[term].append((idx, freq))
        n = len(self._doc_len)
        self._avg_len = sum(self._doc_len) / n if n else 0.0

    def search(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        n = len(self._docs)
        if n == 0:
            return []
        scores: Dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            plist = self._postings.get(term)
            if not plist:
                continue
            idf = math.log(1.0 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
            for doc_i, freq in plist:
                tf_norm = freq * (self.k1 + 1) / (
                    freq + self.k1 * (1 - self.b + self.b * self._doc_len[doc_i] / (self._avg_len or 1))
                )
                scores[doc_i] += idf * tf_norm
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[:top_k]


@dataclass
class FaqEntry:
    question: str
    answer: str
    aliases: List[str]


@dataclass
class FaqMatch:
    entry: FaqEntry
    score: float
    margin: float


class FaqShortCircuit:
    """FAQ 短路器：加载 jsonl FAQ 库，命中即返回人工答案。

    FAQ 库格式（每行）::

        {"question": "...", "answer": "...", "aliases": ["...", "..."]}
    """

    def __init__(self, min_score: float = 4.5, min_margin: float = 1.0):
        self.min_score = min_score
        self.min_margin = min_margin
        self.entries: List[FaqEntry] = []
        self._index = BM25Index()

    def load_jsonl(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                self.entries.append(FaqEntry(
                    question=data["question"],
                    answer=data["answer"],
                    aliases=data.get("aliases", []),
                ))
        corpus = [
            " ".join([e.question] + e.aliases) for e in self.entries
        ]
        self._index.add(corpus)

    def match(self, query: str) -> Optional[FaqMatch]:
        """返回 None 表示未命中（或存在歧义），应继续走向量链路。"""
        if not self.entries:
            return None
        ranked = self._index.search(query, top_k=2)
        if not ranked:
            return None
        top_i, top_score = ranked[0]
        margin = top_score - ranked[1][1] if len(ranked) > 1 else math.inf
        if top_score < self.min_score or margin < self.min_margin:
            return None
        return FaqMatch(entry=self.entries[top_i], score=top_score, margin=margin)
