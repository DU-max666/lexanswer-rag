"""法衡 · 前后对照 A/B 评测。

对照实验设计的核心：**同一检索结果集，只换模型**。

对每个评测样本：
1. 检索链路只跑一次（意图闸门→混合检索→精排→父块回溯），
   得到固定上下文；
2. 基座模型与微调模型分别在同一上下文上生成；
3. 统一过引用回查 + 结构校验，按 metrics.py 口径打分。

这样回答准确率/引用准确率的变化只能来自模型本身，
排除了检索波动的干扰——与简历项目中「52.3%→81.5%」的
对照口径一致。

裸环境用 MockGateway 的 base/finetuned 两种风格演示全链路；
配置 ``--gateway openai`` 后走 OpenAI 兼容端点（vLLM），
``--model-a`` / ``--model-b`` 分别指定基座与微调模型名。

用法::

    python -m eval.run_ab_eval --eval-set eval/eval_set.jsonl
    python -m eval.run_ab_eval --gateway openai \
        --model-a Qwen2.5-7B-Instruct --model-b lexanswer-7b
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

from corpus import load_directory, split_document
from corpus.dedup import dedup_chunks
from retrieval.embeddings import get_embedder
from retrieval.intent_gate import get_intent_gate
from retrieval.rerank import get_reranker
from retrieval.store import get_store
from retrieval.strategies import RetrievalPipeline
from serving.gateway import (
    Message, MockGateway, OpenAICompatGateway, get_gateway_from_env,
)
from verify.answer_format import check_answer_format
from verify.citation_checker import check_answer

from .metrics import GoldSample, SampleScore, aggregate, score_prediction

SYSTEM_PROMPT = (
    "你是律所内部法律知识助手。仅依据【检索依据】作答，"
    "按【原文】【适用条件】【结论】【建议】四段输出；"
    "检索依据不足时必须拒答，禁止编造法条。"
)


def load_eval_set(path: str) -> List[GoldSample]:
    samples: List[GoldSample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(GoldSample.from_dict(json.loads(line)))
    return samples


def build_pipeline(corpus_dir: str) -> RetrievalPipeline:
    """从样例语料构建可离线运行的检索管线。"""
    embedder = get_embedder()
    store = get_store(dim=getattr(embedder, "dim", 256))
    docs = load_directory(corpus_dir)
    chunks = [c for d in docs for c in split_document(d)]
    kept = dedup_chunks(chunks).kept
    vectors = embedder.encode([c.text for c in kept])
    store.upsert(kept, vectors.dense, vectors.sparse)
    return RetrievalPipeline(embedder, store, get_reranker(), get_intent_gate())


def evaluate(pipeline: RetrievalPipeline, gateways: Dict[str, object],
             samples: List[GoldSample]) -> Dict[str, List[SampleScore]]:
    """固定检索结果集，逐模型生成并打分。"""
    results: Dict[str, List[SampleScore]] = {name: [] for name in gateways}
    for gold in samples:
        outcome = pipeline.run(gold.question)
        context = outcome.context()
        messages: List[Message] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"【检索依据】\n{context}\n\n【问题】\n{gold.question}"},
        ]
        for name, gw in gateways.items():
            answer = gw.chat(messages)  # type: ignore[attr-defined]
            verdict = check_answer(answer, outcome.evidences)
            fmt = check_answer_format(answer)
            results[name].append(score_prediction(gold, name, answer, verdict, fmt))
    return results


def print_report(results: Dict[str, List[SampleScore]],
                 order: List[str]) -> None:
    """打印前后对照表。"""
    summaries = {name: aggregate(scores) for name, scores in results.items()}
    print("\n" + "=" * 66)
    print(f"{'模型':<22}{'回答准确率':>10}{'引用准确率':>10}{'行为正确率':>10}")
    print("-" * 66)
    for name in order:
        s = summaries[name]
        print(f"{name:<22}{s['回答准确率']:>9.1f}%{s['法条引用准确率']:>9.1f}%"
              f"{s['行为正确率']:>9.1f}%")
    print("-" * 66)
    for name in order:
        print(f"[{name}] 错误分型: {summaries[name]['错误分型']}")
    print("=" * 66)


def main() -> None:
    p = argparse.ArgumentParser(description="同一检索结果集、只换模型的 A/B 评测")
    p.add_argument("--eval-set", default="eval/eval_set.jsonl")
    p.add_argument("--corpus-dir", default="corpus/sample_data")
    p.add_argument("--gateway", choices=["mock", "openai"], default="mock")
    p.add_argument("--model-a", default="Qwen2.5-7B-Instruct", help="对照模型（基座）")
    p.add_argument("--model-b", default="lexanswer-7b", help="实验模型（微调）")
    p.add_argument("--detail", action="store_true", help="打印逐样本得分")
    args = p.parse_args()

    pipeline = build_pipeline(args.corpus_dir)
    samples = load_eval_set(args.eval_set)

    if args.gateway == "mock":
        gateways = {"base(未微调)": MockGateway("base"),
                    "finetuned(微调后)": MockGateway("finetuned")}
    else:
        env_gw = get_gateway_from_env()
        if not isinstance(env_gw, OpenAICompatGateway):
            raise SystemExit("openai 模式需设置 LEXANSWER_MODEL_BASE_URL（指向 vLLM 端点）")
        gateways = {args.model_a: env_gw, args.model_b: env_gw}

    results = evaluate(pipeline, gateways, samples)
    if args.detail:
        for name, scores in results.items():
            for s in scores:
                print(f"  [{name}] {s.id}: behavior={s.behavior_correct} "
                      f"citation={s.citation_correct} scenario={s.scenario_correct} "
                      f"format={s.format_ok} error={s.error_type}")
    print_report(results, list(gateways.keys()))


if __name__ == "__main__":
    main()
