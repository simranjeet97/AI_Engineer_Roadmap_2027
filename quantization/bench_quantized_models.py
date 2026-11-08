#!/usr/bin/env python3
"""
Apex-Infer Quantization Performance & Bandwidth Benchmark
Simulates HBM bandwidth consumption, decode step latency, and memory footprint
for BF16, FP8, and INT4-AWQ/Marlin kernels across batch sizes.
"""

import argparse
from dataclasses import dataclass


@dataclass
class HardwareSpec:
    name: str
    peak_tflops: float
    hbm_bw_tb_s: float


@dataclass
class ModelSpec:
    name: str
    num_params_billion: float


def benchmark_quantization(hw: HardwareSpec, model: ModelSpec, batch_size: int, seq_len: int):
    # Weights footprint
    bytes_bf16 = model.num_params_billion * 1e9 * 2.0
    bytes_fp8  = model.num_params_billion * 1e9 * 1.0
    bytes_int4 = model.num_params_billion * 1e9 * 0.5

    # Memory bus bandwidth in Bytes/sec
    bw_bytes_sec = hw.hbm_bw_tb_s * 1e12

    # Decode step: every token generation loads all model weights from HBM
    t_decode_bf16_ms = (bytes_bf16 / bw_bytes_sec) * 1000.0
    t_decode_fp8_ms  = (bytes_fp8  / bw_bytes_sec) * 1000.0
    t_decode_int4_ms = (bytes_int4 / bw_bytes_sec) * 1000.0

    print("=" * 85)
    print(f"Quantization Shootout: {model.name} ({model.num_params_billion}B params) on {hw.name}")
    print(f"Batch Size: {batch_size} | Context Length: {seq_len} tokens")
    print("=" * 85)
    print(f"{'Precision':<12} | {'Weight VRAM':<14} | {'Ideal Decode Latency':<22} | {'Max Single-Stream Tok/s'}")
    print("-" * 85)
    print(f"{'BF16':<12} | {bytes_bf16 / 1e9:<10.1f} GB | {t_decode_bf16_ms:<18.2f} ms | {1000.0 / t_decode_bf16_ms:<10.1f} tok/s")
    print(f"{'FP8 (E4M3)':<12} | {bytes_fp8  / 1e9:<10.1f} GB | {t_decode_fp8_ms:<18.2f} ms | {1000.0 / t_decode_fp8_ms:<10.1f} tok/s (2.0x)")
    print(f"{'INT4-AWQ':<12} | {bytes_int4 / 1e9:<10.1f} GB | {t_decode_int4_ms:<18.2f} ms | {1000.0 / t_decode_int4_ms:<10.1f} tok/s (4.0x)")
    print("=" * 85)
    print("Note: In memory-bound single-stream decode, latency scales strictly linearly with weight footprint.")
    print("Marlin W4A16 kernels dequantize weights into FP16/BF16 in SRAM registers at line-rate.")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Quantized Model Memory Bandwidth")
    parser.add_argument("--params", type=float, default=70.0, help="Model parameters in billions (e.g. 8 or 70)")
    parser.add_argument("--batch-size", type=int, default=1, help="Concurrent batch size")
    parser.add_argument("--bw", type=float, default=3.35, help="GPU HBM bandwidth in TB/s (H100=3.35, A100=2.039)")
    args = parser.parse_args()

    hw = HardwareSpec(name="NVIDIA H100 SXM5", peak_tflops=989.0, hbm_bw_tb_s=args.bw)
    model = ModelSpec(name=f"Llama-3-{args.params:g}B", num_params_billion=args.params)
    benchmark_quantization(hw, model, args.batch_size, 4096)


if __name__ == "__main__":
    main()
