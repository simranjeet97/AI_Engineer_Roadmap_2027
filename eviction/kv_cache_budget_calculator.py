#!/usr/bin/env python3
"""
Apex-Infer KV Cache Memory Budget & Concurrency Ceiling Calculator
Computes exact per-token and aggregate KV cache memory requirements across MHA, GQA, and MLA architectures,
and derives maximum concurrent streams for target GPU clusters.
"""

import argparse
from dataclasses import dataclass


@dataclass
class ArchitectureSpec:
    name: str
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    bytes_per_param: int = 2  # 2 for BF16/FP16, 1 for FP8
    is_mla: bool = False
    mla_latent_dim: int = 512


MODELS = {
    "llama-3-8b": ArchitectureSpec("Llama-3-8B (GQA)", 32, 4096, 32, 8, 128),
    "llama-3-70b": ArchitectureSpec("Llama-3-70B (GQA)", 80, 8192, 64, 8, 128),
    "deepseek-v3": ArchitectureSpec("DeepSeek-V3 (MLA)", 61, 7168, 128, 128, 128, is_mla=True, mla_latent_dim=576),
    "llama-2-70b-mha": ArchitectureSpec("Llama-2-70B (Legacy MHA)", 80, 8192, 64, 64, 128),
}


def calculate_kv_cache_size(arch: ArchitectureSpec, seq_len: int, batch_size: int, dtype_bytes: int) -> float:
    """Returns KV cache memory in Gigabytes (GB)."""
    if arch.is_mla:
        # Multi-Head Latent Attention compresses KV heads into a single latent vector
        bytes_per_token = arch.num_layers * (arch.mla_latent_dim + 64) * dtype_bytes
    else:
        # Standard GQA / MHA: 2 * num_layers * num_kv_heads * head_dim * bytes
        bytes_per_token = 2 * arch.num_layers * arch.num_kv_heads * arch.head_dim * dtype_bytes

    total_bytes = bytes_per_token * seq_len * batch_size
    return total_bytes / (1024 ** 3)


def main():
    parser = argparse.ArgumentParser(description="KV Cache Memory Calculator")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="llama-3-70b")
    parser.add_argument("--gpu-mem", type=float, default=80.0, help="Per-GPU VRAM in GB")
    parser.add_argument("--gpus", type=int, default=8, help="Number of GPUs (Tensor Parallel)")
    parser.add_argument("--weights-mem", type=float, default=70.0, help="VRAM consumed by model weights (GB)")
    parser.add_argument("--fp8-kv", action="store_true", help="Use FP8 for KV cache (1 byte/elem instead of 2)")
    args = parser.parse_args()

    arch = MODELS[args.model]
    dtype_bytes = 1 if args.fp8_kv else 2
    total_cluster_vram = args.gpu_mem * args.gpus
    available_kv_vram = total_cluster_vram - args.weights_mem

    print("=" * 80)
    print(f"Apex-Infer KV Memory Budget: {arch.name}")
    print(f"Cluster: {args.gpus}x GPUs ({args.gpu_mem} GB each = {total_cluster_vram:.1f} GB Total)")
    print(f"Model Weights Footprint: {args.weights_mem:.1f} GB | Available KV VRAM: {available_kv_vram:.1f} GB")
    print(f"KV Cache Precision: {'FP8 (1 byte)' if args.fp8_kv else 'BF16/FP16 (2 bytes)'}")
    print("=" * 80)

    contexts = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
    print(f"{'Context Length':<16} | {'KV Size / Stream (GB)':<24} | {'Max Concurrent Streams'}")
    print("-" * 80)

    for ctx in contexts:
        kv_per_stream_gb = calculate_kv_cache_size(arch, ctx, 1, dtype_bytes)
        max_concurrency = int(available_kv_vram / kv_per_stream_gb) if kv_per_stream_gb > 0 else 0
        print(f"{ctx:<16,d} | {kv_per_stream_gb:<24.3f} | {max_concurrency:<10,d} streams")

    print("=" * 80)


if __name__ == "__main__":
    main()
