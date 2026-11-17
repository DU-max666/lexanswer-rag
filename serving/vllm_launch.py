"""法衡 · vLLM 启动器。

把合并后的权重（finetune/merge_adapter.py 产物）以 OpenAI 兼容
服务拉起，单卡 4090 实测口径：首 Token ≈0.5s，稳定 35-40 token/s，
并发 4-8 路（法律问答场景下质量/延迟平衡点）。

本脚本职责：
1. 校验合并权重目录完整性（config.json + safetensors）
2. 组装 vllm serve 命令（量化、显存占用、并发上限均可调）
3. ``--print-only`` 打印命令供人工审阅；默认尝试 exec 启动

用法::

    python -m serving.vllm_launch --model /models/lexanswer-merged --print-only
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import Dict, List

#: 单卡 24GB 推荐参数（与 deploy/README.md 保持一致）
DEFAULTS: Dict[str, object] = {
    "port": 8000,
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.85,
    "max_num_seqs": 8,
    "dtype": "bfloat16",
}


def check_model_dir(model_dir: str) -> List[str]:
    """返回缺失的关键权重文件列表（空列表 = 通过）。"""
    required_any = ("config.json",)
    problems: List[str] = []
    if not os.path.isdir(model_dir):
        return [f"目录不存在: {model_dir}"]
    for f in required_any:
        if not os.path.exists(os.path.join(model_dir, f)):
            problems.append(f)
    weights = [f for f in os.listdir(model_dir) if f.endswith((".safetensors", ".bin"))]
    if not weights:
        problems.append("*.safetensors / *.bin 权重分片")
    return problems


def build_command(model_dir: str, served_name: str, quantization: str = "",
                  extra: List[str] = None) -> List[str]:
    """组装 vllm serve 命令。"""
    cmd = [
        "vllm", "serve", model_dir,
        "--served-model-name", served_name,
        "--port", str(DEFAULTS["port"]),
        "--max-model-len", str(DEFAULTS["max_model_len"]),
        "--gpu-memory-utilization", str(DEFAULTS["gpu_memory_utilization"]),
        "--max-num-seqs", str(DEFAULTS["max_num_seqs"]),
        "--dtype", str(DEFAULTS["dtype"]),
    ]
    if quantization:
        cmd += ["--quantization", quantization]  # 如 awq/gptq 进一步压显存
    if extra:
        cmd += extra
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(description="以 vLLM 拉起合并权重的 OpenAI 兼容服务")
    p.add_argument("--model", required=True, help="merge_adapter 输出目录")
    p.add_argument("--served-name", default="lexanswer-7b")
    p.add_argument("--quantization", default="", help="可选 awq/gptq 等，空=不量化")
    p.add_argument("--max-num-seqs", type=int, default=int(DEFAULTS["max_num_seqs"]))
    p.add_argument("--print-only", action="store_true", help="仅打印命令不执行")
    args = p.parse_args()

    problems = check_model_dir(args.model)
    if problems:
        print(f"[vllm_launch] 权重目录校验失败，缺少: {problems}")
        print("  请先运行 finetune/merge_adapter.py 产出合并权重。")
        raise SystemExit(1)

    cmd = build_command(args.model, args.served_name, args.quantization)
    cmd[cmd.index("--max-num-seqs") + 1] = str(args.max_num_seqs)
    print("[vllm_launch] 启动命令：", " ".join(cmd))
    if args.print_only:
        print("[vllm_launch] --print-only 模式结束。")
        return
    if shutil.which("vllm") is None:
        print("[vllm_launch] 未检测到 vllm CLI（裸环境属正常），退出。")
        raise SystemExit(2)
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
