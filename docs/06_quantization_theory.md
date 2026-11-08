# 06. Quantization Theory: Formats, AWQ & Fused Marlin Kernels

## Low-Bit Numerical Formats
* **FP8 E4M3 (1-4-3)**: 1 sign, 4 exponent, 3 mantissa (Max: 448). Highest precision; preferred for **weights & activations**.
* **FP8 E5M2 (1-5-2)**: 1 sign, 5 exponent, 2 mantissa (Max: 57,344). Wide dynamic range; preferred for **KV caches & gradients**.
* **INT4**: 4-bit uniform integer quantization ($[-8, 7]$).

## Activation Outliers & AWQ
In models $> 6.7\text{B}$, $0.1\%$ of hidden channels exhibit activation magnitudes $> 100\times$ larger than normal channels.
* Naive INT4 quantization rounds these channels coarsely, causing perplexity collapse.
* **AWQ (Activation-Aware Weight Quantization)**: Multiplies salient weight channels by a per-channel scale factor $s > 1$ before quantization:
  $$W' = W \cdot \text{diag}(s), \quad X' = \text{diag}(s)^{-1} \cdot X$$
  Suppresses weight rounding error on critical channels by factor $s$, preserving accuracy with zero activation quantization overhead.

## The Marlin Kernel Breakthrough
In W4A16 serving, INT4 weights must be unpacked to FP16 before Tensor Core math.
* At $B \ge 16$, naive register unpacking instructions bottleneck the GPU, making naive INT4 **slower than FP16**.
* **Marlin**: An asynchronous, double-buffered fused GEMM kernel that overlaps global memory loads, dequantization, and Tensor Core math in custom PTX assembly, sustaining **$3.8\times$ speedups** across all batch sizes.
