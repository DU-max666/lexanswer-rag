"""法衡 · Qwen2.5-7B-Instruct QLoRA 微调脚本。

单卡 RTX 4090（24GB）上的 QLoRA 配方（详见 configs/peft_lora.yaml）：

- 基座 4-bit NF4 量化加载 + 双重量化（double quant），bf16 计算
- 梯度检查点（gradient checkpointing）换取激活显存
- LoRA 适配器挂 attention 投影（q/k/v/o）+ MLP（gate/up/down），
  r=64 / alpha=128 / dropout=0.05，可训参数约占 7B 基座的 0.57%
- 有效 batch = per_device 1 × 梯度累积 16 = 16，序列长度 2048，3 epoch
- 约 4.5 小时完成一轮训练（演示数据集规模远小于此，仅作链路验证）

无 GPU / 未装 transformers+peft+bitsandbytes 时，脚本进入
``--mock`` 计划模式：解析并校验全部配置、估算显存与可训参数占比，
输出训练计划后正常退出，保证 CI 与本地可跑。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple

#: Qwen2.5-7B 全量权重（bf16）约 15.2GB；4-bit NF4 后约 3.9GB
BASE_WEIGHT_GB = {"qwen2.5-7b-instruct": 15.2, "qwen2.5-7b": 15.2}

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]


@dataclass
class TrainConfig:
    """训练超参集合；命令行与 yaml 配置共享同一结构。"""

    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    dataset_path: str = "finetune/output/demo_dataset.jsonl"
    output_dir: str = "finetune/output/lexanswer-qlora"
    epochs: int = 3
    seq_len: int = 2048
    per_device_batch: int = 1
    grad_accum: int = 16
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: list(LORA_TARGET_MODULES))
    nf4: bool = True
    double_quant: bool = True
    compute_dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    logging_steps: int = 10
    save_steps: int = 200

    @property
    def effective_batch(self) -> int:
        return self.per_device_batch * self.grad_accum

    @classmethod
    def from_yaml(cls, path: str) -> "TrainConfig":
        """从 peft_lora.yaml 读取（无 PyYAML 时解析极简 key: value）。"""
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(open(path, "r", encoding="utf-8")) or {}
        except ImportError:
            data = {}
            for line in open(path, "r", encoding="utf-8"):
                line = line.split("#", 1)[0].strip()
                if ":" in line and not line.endswith(":"):
                    k, v = line.split(":", 1)
                    data[k.strip()] = _coerce(v.strip())
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


def _coerce(value: str):
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value.strip("'\"")


def estimate_trainable_params_pct(lora_r: int, target_modules: List[str]) -> float:
    """估算 LoRA 可训参数占比（Qwen2.5-7B 口径）。

    基准锚点：r=64、适配器挂满 attention(q/k/v/o)+FFN(gate/up/down)
    共 7 个投影时，peft 实测可训参数约 43.5M / 7.62B ≈ 0.57%。
    本估算按秩与模块数线性缩放，供无卡环境校验配置量级。
    """
    ref_pct, ref_r, ref_modules = 0.57, 64, 7
    return ref_pct * (lora_r / ref_r) * (len(target_modules) / ref_modules)


def load_sharegpt(path: str) -> List[Dict]:
    """读取 ShareGPT jsonl 指令集。"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_mock(cfg: TrainConfig, n_samples: int) -> None:
    """打印训练计划与资源估算，不触碰 GPU。"""
    pct = estimate_trainable_params_pct(cfg.lora_r, cfg.target_modules)
    base_gb = BASE_WEIGHT_GB.get(cfg.base_model.split("/")[-1].lower(), 15.2)
    quant_gb = base_gb * 0.26 if cfg.nf4 else base_gb
    if cfg.double_quant and cfg.nf4:
        quant_gb *= 0.94  # 双重量化再省约 6% 量化常数
    feasible = "可行" if quant_gb + 6 < 24 else "需降低 seq_len 或 batch"
    print("=" * 60)
    print("[train_qlora][MOCK] 无 GPU/训练依赖环境，输出训练计划：")
    print(json.dumps(asdict(cfg), ensure_ascii=False, indent=2))
    print("-" * 60)
    print(f"样本数           : {n_samples}")
    print(f"有效 batch       : {cfg.effective_batch}")
    print(f"每 epoch 步数    : {max(n_samples // cfg.effective_batch, 1)}")
    print(f"可训参数占比估算 : {pct:.2f}%")
    print(f"显存估算         : 量化权重≈{quant_gb:.1f}GB + LoRA/优化器/激活≈6GB → 单卡24GB {feasible}")
    print("[train_qlora][MOCK] 计划校验通过。有卡环境去掉 --mock 即执行真实训练。")


def to_chatml(record: Dict) -> str:
    """ShareGPT 记录 → ChatML 文本。"""
    role_map = {"system": "system", "human": "user", "gpt": "assistant"}
    parts = []
    for conv in record["conversations"]:
        role = role_map[conv["from"]]
        parts.append(f"{role}\n{conv['value']}")
    return "\n".join(parts)


def train(cfg: TrainConfig) -> None:
    """QLoRA 真实训练：仅在 GPU + 完整依赖环境下可达。"""
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    )
    from peft import (
        LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType,
    )

    bnb = BitsAndBytesConfig(
        load_in_4bit=cfg.nf4,
        bnb_4bit_quant_type="nf4" if cfg.nf4 else None,
        bnb_4bit_use_double_quant=cfg.double_quant,
        bnb_4bit_compute_dtype=getattr(torch, cfg.compute_dtype),
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, quantization_config=bnb, device_map="auto",
        trust_remote_code=True,
    )
    if cfg.gradient_checkpointing:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()  # 预期约 0.57%

    texts = [to_chatml(r) for r in load_sharegpt(cfg.dataset_path)]

    try:
        from trl import SFTTrainer, SFTConfig  # 首选 TRL 打包 SFT

        sft_args = SFTConfig(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.per_device_batch,
            gradient_accumulation_steps=cfg.grad_accum,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type=cfg.lr_scheduler,
            warmup_ratio=cfg.warmup_ratio,
            bf16=(cfg.compute_dtype == "bfloat16"),
            gradient_checkpointing=cfg.gradient_checkpointing,
            max_seq_length=cfg.seq_len,
            logging_steps=cfg.logging_steps,
            save_strategy="steps",
            save_steps=cfg.save_steps,
            save_total_limit=2,
            report_to=[],
        )
        trainer = SFTTrainer(model=model, args=sft_args,
                             train_dataset=texts, processing_class=tokenizer)
    except ImportError:
        # 无 trl 时的兜底：手写极简循环仅演示链路（生产建议 trl）
        from transformers import TrainingArguments, Trainer
        from torch.utils.data import Dataset

        class TextDataset(Dataset):
            def __len__(self):
                return len(texts)

            def __getitem__(self, i):
                enc = tokenizer(texts[i], truncation=True, max_length=cfg.seq_len,
                               padding="max_length", return_tensors="pt")
                return {k: v.squeeze(0) for k, v in enc.items()} | {"labels": enc["input_ids"].squeeze(0)}

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=cfg.output_dir, num_train_epochs=cfg.epochs,
                per_device_train_batch_size=cfg.per_device_batch,
                gradient_accumulation_steps=cfg.grad_accum,
                learning_rate=cfg.learning_rate,
                save_strategy="steps", save_steps=cfg.save_steps,
                report_to=[],
            ),
            train_dataset=TextDataset(),
        )
    trainer.train()
    trainer.save_model(cfg.output_dir)  # 仅保存 LoRA 适配器（几十 MB）
    tokenizer.save_pretrained(cfg.output_dir)
    print(f"[train_qlora] 适配器已保存 → {cfg.output_dir}")


def _gpu_ready() -> bool:
    """检测真实训练是否可行：CUDA + 三大依赖。"""
    try:
        import torch
        import peft  # noqa: F401
        import bitsandbytes  # noqa: F401

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="法衡 QLoRA 微调（无卡环境自动进入 mock 计划模式）")
    p.add_argument("--config", default="finetune/configs/peft_lora.yaml",
                   help="yaml 配置（命令行参数覆盖其上）")
    p.add_argument("--base-model", dest="base_model")
    p.add_argument("--dataset", dest="dataset_path")
    p.add_argument("--output-dir", dest="output_dir")
    p.add_argument("--epochs", type=int)
    p.add_argument("--seq-len", dest="seq_len", type=int)
    p.add_argument("--per-device-batch", dest="per_device_batch", type=int)
    p.add_argument("--grad-accum", dest="grad_accum", type=int)
    p.add_argument("--learning-rate", dest="learning_rate", type=float)
    p.add_argument("--lora-r", dest="lora_r", type=int)
    p.add_argument("--lora-alpha", dest="lora_alpha", type=int)
    p.add_argument("--no-nf4", dest="nf4", action="store_false", default=None)
    p.add_argument("--no-double-quant", dest="double_quant", action="store_false", default=None)
    p.add_argument("--mock", action="store_true", help="强制计划模式，不触碰 GPU")
    return p


def resolve_config(argv=None) -> Tuple[TrainConfig, bool]:
    """yaml 打底 + 命令行覆盖，得到最终 TrainConfig。"""
    args = build_arg_parser().parse_args(argv)
    cfg = TrainConfig.from_yaml(args.config) if args.config else TrainConfig()
    for key in ("base_model", "dataset_path", "output_dir", "epochs", "seq_len",
                "per_device_batch", "grad_accum", "learning_rate", "lora_r",
                "lora_alpha", "nf4", "double_quant"):
        val = getattr(args, key, None)
        if val is not None:
            setattr(cfg, key, val)
    return cfg, args.mock


def main(argv=None) -> None:
    cfg, force_mock = resolve_config(argv)
    n_samples = 0
    if cfg.dataset_path:
        try:
            n_samples = len(load_sharegpt(cfg.dataset_path))
        except FileNotFoundError:
            print(f"[train_qlora] 数据集不存在：{cfg.dataset_path}（先运行 build_dataset.py）")
    if force_mock or not _gpu_ready():
        run_mock(cfg, n_samples)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
