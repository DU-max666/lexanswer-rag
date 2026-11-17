"""法衡 · ShareGPT 指令集构造脚本。

从 ``corpus/sample_data`` 的小样例法规出发，构造可训练的
指令微调数据（真实项目里替换为律师复核过的案卷脱敏语料）：

正样本
    现行有效条文 → 「原文 + 适用条件 + 结论 + 建议」四段式回答，
    与 verify/answer_format.py 的输出契约完全一致，让模型
    在微调阶段就学会可校验的回答结构。

负样本（拒答能力的关键，占比由 --neg-ratio 控制）
    1. 已废止法规条文：回答必须声明废止并拒绝引用
    2. 不存在的条号：回答必须声明检索不到并拒绝编造

输出 ShareGPT 对话格式（jsonl，每行）::

    {"id": "...", "conversations": [
        {"from": "system", "value": "..."},
        {"from": "human", "value": "..."},
        {"from": "gpt", "value": "..."}]}

用法::

    python -m finetune.build_dataset --corpus-dir corpus/sample_data \
        --out finetune/output/demo_dataset.jsonl --seed 7 --neg-ratio 0.3
"""
from __future__ import annotations

import argparse
import json
import random
import re
from typing import Dict, List

from corpus import load_directory, split_document
from corpus.dedup import dedup_chunks
from corpus.schema import Chunk, ChunkKind, EffectStatus

SYSTEM_PROMPT = (
    "你是律所内部法律知识助手。回答必须依据检索到的现行有效法条，"
    "按【原文】【适用条件】【结论】【建议】四段输出；"
    "若所询条文已废止或检索不到，必须明确拒答并说明原因，禁止编造法条。"
)

REFUSE_REPEALED = (
    "无法依据该条文回答。《{law}》（{version}）已被宣告废止，"
    "其条文不再具有法律效力，引用可能导致错误法律意见。"
    "请提供现行有效的法律依据，或咨询值班律师确认替代规范。"
)

REFUSE_MISSING = (
    "无法回答。经在知识库中检索，《{law}》不存在第{article}条，"
    "或该条文未收录。为避免编造法条，本助手拒绝给出推测性回答，"
    "建议核对法名与条号后重新提问。"
)


def _topic_of(chunk: Chunk) -> str:
    """从条文正文提取简短主题短语，用于生成多样化问法。"""
    body = re.sub(r"^《[^》]+》第[零一二三四五六七八九十百千0-9]+条\s*", "", chunk.text)
    body = re.split(r"[，。；：]", body)
    phrase = next((seg.strip() for seg in body if len(seg.strip()) >= 6), "相关规定")
    return phrase[:18]


def positive_samples(articles: List[Chunk], rng: random.Random) -> List[Dict]:
    """有效条文 → 四段式正样本（每条文 1~2 条问法变体）。"""
    samples: List[Dict] = []
    question_templates = [
        "根据《{law}》，关于{topic}有什么规定？",
        "《{law}》第{cn}条的内容是什么？请给出适用条件与建议。",
    ]
    for a in articles:
        topic = _topic_of(a)
        variants = [question_templates[0].format(law=a.law_name, topic=topic)]
        if rng.random() < 0.5:
            variants.append(question_templates[1].format(law=a.law_name, cn=_cn(a.article_no)))
        answer = build_structured_answer(a)
        for i, q in enumerate(variants):
            samples.append(make_record(f"pos-{a.chunk_id}-{i}", q, answer))
    return samples


def _cn(num) -> str:
    from corpus.splitters import _int_to_cn
    return _int_to_cn(int(num)) if num is not None else "?"


def build_structured_answer(a: Chunk) -> str:
    """按四段式契约组装标准答案（原文/适用条件/结论/建议）。"""
    body = re.sub(r"^《[^》]+》第[零一二三四五六七八九十百千0-9]+条\s*", "", a.text)
    items = re.split(r"（[一二三四五六七八九十]+）", body)
    if len(items) > 1:
        # 款项列举型条文：各列举情形即适用条件
        conds = [seg.strip("；;，。 ") for seg in items[1:] if len(seg.strip("；;，。 ")) >= 4]
        conditions = "；".join(conds)[:120]
    else:
        m = re.match(r"^(.{4,30}?)的，", body)
        conditions = m.group(1) + "的情形" if m else body[:24] + "等情形"
    return (
        f"【原文】{a.heading}：{body}\n"
        f"【适用条件】{conditions}\n"
        f"【结论】该条为现行有效规定，处理{a.law_name}项下相关事务时应严格遵照执行。\n"
        f"【建议】涉及具体个案时，请结合事实材料咨询承办律师，本回复不构成正式法律意见。"
    )


def negative_samples(articles: List[Chunk], all_laws: Dict[str, List[int]],
                     rng: random.Random, quota: int) -> List[Dict]:
    """构造废止条文拒答 + 不存在条号拒答两类负样本。"""
    samples: List[Dict] = []
    # 1) 已废止法规
    for a in articles:
        q = f"《{a.law_name}》第{_cn(a.article_no)}条规定的内容现在还适用吗？我想引用它写法律意见。"
        ans = REFUSE_REPEALED.format(law=a.law_name, version=a.extra.get("version", "旧版本"))
        samples.append(make_record(f"neg-r-{a.chunk_id}", q, ans))
    # 2) 不存在的条号：挑一部法，编一个超出其最大条号的编号
    while len(samples) < quota and all_laws:
        law = rng.choice(list(all_laws.keys()))
        fake_no = max(all_laws[law]) + rng.randint(2, 50)
        q = f"请解释《{law}》第{_cn(fake_no)}条关于保证金的规定，并给出适用建议。"
        ans = REFUSE_MISSING.format(law=law, article=_cn(fake_no))
        samples.append(make_record(f"neg-m-{law}-{fake_no}", q, ans))
    return samples


def make_record(rid: str, question: str, answer: str) -> Dict:
    """ShareGPT 记录封装。"""
    return {
        "id": rid,
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer},
        ],
    }


def build_dataset(corpus_dir: str, seed: int = 7, neg_ratio: float = 0.3,
                  max_positive: int = 200) -> List[Dict]:
    """主流程：加载→切分→去重→正负样本→打乱。"""
    rng = random.Random(seed)
    docs = load_directory(corpus_dir)
    chunks = [c for d in docs for c in split_document(d)]
    result = dedup_chunks(chunks)

    effective_articles = [
        c for c in result.kept
        if c.kind == ChunkKind.ARTICLE and c.effect_status == EffectStatus.EFFECTIVE
    ]
    repealed_articles = [
        c for c in result.repealed if c.kind == ChunkKind.ARTICLE
    ][:8]
    all_laws: Dict[str, List[int]] = {}
    for c in result.kept:
        if c.article_no is not None:
            all_laws.setdefault(c.law_name, []).append(c.article_no)

    positives = positive_samples(effective_articles[:max_positive], rng)
    neg_quota = max(int(len(positives) * neg_ratio), len(repealed_articles))
    negatives = negative_samples(repealed_articles, all_laws, rng, neg_quota)
    dataset = positives + negatives
    rng.shuffle(dataset)
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="构造 ShareGPT 指令微调数据集")
    parser.add_argument("--corpus-dir", default="corpus/sample_data")
    parser.add_argument("--out", default="finetune/output/demo_dataset.jsonl")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--neg-ratio", type=float, default=0.3)
    parser.add_argument("--max-positive", type=int, default=200)
    args = parser.parse_args()

    dataset = build_dataset(args.corpus_dir, args.seed, args.neg_ratio, args.max_positive)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in dataset:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    pos = sum(1 for r in dataset if r["id"].startswith("pos"))
    print(f"[build_dataset] 写入 {len(dataset)} 条（正 {pos} / 拒答负 {len(dataset) - pos}）→ {args.out}")


if __name__ == "__main__":
    main()
