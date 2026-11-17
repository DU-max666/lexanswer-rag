"""法衡 · LoRA 适配器合并。

vLLM 对「基座 + 运行时挂载 adapter」的支持有限且吞吐差，
生产部署采用：训练产出 adapter → 离线合并回基座 → 以合并权重起服务。

- 合并时基座以 bf16 加载（不量化），保证精度不二次受损
- 输出目录可直接喂给 serving/vllm_launch.py
- 无 GPU/依赖环境提供 ``--check`` 干跑：校验 adapter 目录结构完整性

用法::

    python -m finetune.merge_adapter \
        --base Qwen/Qwen2.5-7B-Instruct \
        --adapter finetune/output/lexanswer-qlora \
        --out finetune/output/lexanswer-merged
"""
from __future__ import annotations

import argparse
import os

ADAPTER_REQUIRED_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
]


def check_adapter_dir(adapter_dir: str) -> bool:
    """校验 adapter 目录：PEFT 保存的最小文件集是否齐全。"""
    missing = [f for f in ADAPTER_REQUIRED_FILES
               if not os.path.exists(os.path.join(adapter_dir, f))]
    if missing:
        print(f"[merge_adapter] adapter 目录缺少: {missing}")
        return False
    print(f"[merge_adapter] adapter 目录校验通过: {adapter_dir}")
    return True


def merge(base_model: str, adapter_dir: str, out_dir: str,
         dtype: str = "bfloat16") -> None:
    """真实合并路径：PeftModel.from_pretrained → merge_and_unload → 保存。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=getattr(torch, dtype),
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    merged = model.merge_and_unload()
    os.makedirs(out_dir, exist_ok=True)
    merged.save_pretrained(out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model, trust_remote_code=True).save_pretrained(out_dir)
    print(f"[merge_adapter] 合并权重已保存 → {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="LoRA 适配器合并回基座")
    p.add_argument("--base", required=True, help="基座模型路径或 HF 名称")
    p.add_argument("--adapter", required=True, help="train_qlora 输出目录")
    p.add_argument("--out", required=True, help="合并权重输出目录")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--check", action="store_true", help="仅校验 adapter 目录，不加载模型")
    args = p.parse_args()

    if not check_adapter_dir(args.adapter):
        raise SystemExit(1)
    if args.check:
        print("[merge_adapter] --check 模式完成（真实合并需 GPU/依赖环境）")
        return
    try:
        merge(args.base, args.adapter, args.out, args.dtype)
    except ImportError as exc:
        print(f"[merge_adapter] 缺少训练依赖（{exc}）。裸环境请用 --check 干跑，"
              f"或在有卡机器上安装 requirements-finetune 后重试。")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
