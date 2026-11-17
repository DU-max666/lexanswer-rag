# 法衡 · QLoRA 微调

在 Qwen2.5-7B-Instruct 上做法律问答指令微调，单卡 4090（24GB）即可完成。

## 流程

```bash
# 1. 构造 ShareGPT 指令集（正样本 + 拒答负样本）
python -m finetune.build_dataset --corpus-dir corpus/sample_data \
    --out finetune/output/demo_dataset.jsonl --seed 7 --neg-ratio 0.3

# 2. 训练（无 GPU 环境自动进入 mock 计划模式，可先 --mock 校验配置）
python -m finetune.train_qlora --config finetune/configs/peft_lora.yaml

# 3. 合并适配器回基座（供 vLLM 部署）
python -m finetune.merge_adapter \
    --base Qwen/Qwen2.5-7B-Instruct \
    --adapter finetune/output/lexanswer-qlora \
    --out finetune/output/lexanswer-merged
```

## 关键配置（configs/peft_lora.yaml）

| 项 | 值 | 说明 |
|---|---|---|
| 量化 | NF4 4-bit + 双重量化 | 基座权重 ≈3.9GB，bf16 计算 |
| 适配器 | r=64 / α=128 / dropout 0.05 | 挂 q/k/v/o + gate/up/down |
| 可训参数 | ≈0.57% | 7.61B 基座上约 43M |
| 批量 | 1 × 累积 16 = 有效 16 | 显存受限下的稳定梯度 |
| 序列 | 2048 | 覆盖「法条上下文 + 四段式回答」 |
| 轮次 | 3 | 约 4.5h（真实语料规模） |

## 数据设计要点

- 正样本答案强制 **原文/适用条件/结论/建议** 四段结构，与
  `verify/answer_format.py` 的校验契约一致；
- 负样本覆盖两类拒答场景：**已废止法规**、**不存在的条号**，
  教会模型「查不到就说查不到」，而不是幻觉编造；
- 演示数据集由 `corpus/sample_data` 自动生成，真实训练需替换为
  律师复核的脱敏案卷语料（格式相同即可）。

## 依赖说明

真实训练需要 GPU + `transformers / peft / bitsandbytes / trl / accelerate`。
裸环境运行 `train_qlora.py` 会自动输出训练计划（mock 模式），
用于验证配置正确性与 CI 冒烟。
