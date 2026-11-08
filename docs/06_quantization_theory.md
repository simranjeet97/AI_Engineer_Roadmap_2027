# 06. Quantization Theory: From FP32 to INT4 — Every Bit Matters

> *"In inference serving, memory bandwidth is destiny. Reduce the bits transferred across the memory bus by half, and you double the speed of token generation."*

When engineers first encounter quantization, they often assume it is primarily about saving disk space or fitting models into smaller RAM. While reducing VRAM footprint is crucial, in high-throughput LLM serving, quantization serves an even more profound physical objective: **conquering the memory bandwidth bottleneck of the decode phase**.

Recall from Chapter 01 that during single-stream token generation, the arithmetic operational intensity is $\approx 1.0 \text{ FLOP/Byte}$. Modern Tensor Cores sit starved, waiting for weights to travel across the memory bus from HBM into registers.

If you represent each model weight with 8 bits instead of 16 bits, the memory bus transfers half as many bytes per forward pass. That directly translates into a **$2\times$ reduction in decode step latency**. If you represent weights with 4 bits, you unlock a **$4\times$ latency reduction**.

This technical memo derives quantization theory from absolute first principles: IEEE 754 bit-level representation, dynamic range versus precision, symmetric vs asymmetric mapping, Activation-Aware Weight Quantization (AWQ), and fused Marlin dequantization kernels.

---

## 1. How Computers Represent Numbers (First Principles)

At the silicon level, digital circuits only understand high and low voltage states: bits of `0` and `1`. To represent real-world decimal numbers (such as $3.14159$ or $-0.0042$), computer scientists developed floating-point formats governed by the IEEE 754 standard.

### The Anatomy of a Floating-Point Number
Every floating-point binary format divides its bits into three distinct fields:

```
Floating-Point Bit Layout:
┌──────┬────────────────────────┬──────────────────────────────────────────┐
│ Sign │ Exponent (E)           │ Mantissa / Fraction (M)                  │
│ (1b) │ Range / Magnitude      │ Precision / Significant Digits           │
└──────┴────────────────────────┴──────────────────────────────────────────┘
```

The decimal value represented by these bits is evaluated as:
$$\text{Value} = (-1)^{\text{Sign}} \times 2^{\text{Exponent} - \text{Bias}} \times \left(1 + \frac{\text{Mantissa}}{2^{N_{\text{mantissa}}}}\right)$$

Where:
* **Sign Bit ($1$ bit)**: `0` for positive, `1` for negative.
* **Exponent Bits ($N_{exp}$)**: Determines the **Dynamic Range** (how extraordinarily large or small the numbers can be before underflow or overflow).
* **Bias**: A constant offset ($2^{N_{exp}-1} - 1$) used so positive exponents don't require a second sign bit.
* **Mantissa Bits ($N_{mantissa}$)**: Determines the **Precision** (how many distinct decimal places between powers of 2 can be distinguished).

---

## 2. The Zoo of Numerical Precisions: FP32 to FP8

Let us dissect the primary formats used in modern deep learning:

```
Comparison of Floating-Point Formats:
FP32 (32-bit): [S:1][====== Exp: 8 ======][==================== Mantissa: 23 ====================]
FP16 (16-bit): [S:1][== Exp: 5 ==][========= Mantissa: 10 =========]
BF16 (16-bit): [S:1][====== Exp: 8 ======][=== Mantissa: 7 ===]
FP8 E4M3:      [S:1][= Exp: 4 =][ Mant: 3 ]
FP8 E5M2:      [S:1][== Exp: 5 ==][Mant: 2]
```

### Detailed Format Specification Matrix:

| Format | Total Bits | Sign | Exponent | Mantissa | Exponent Bias | Max Absolute Value | Smallest Positive Normal | Primary Role in AI |
|---|---|---|---|---|---|---|---|---|
| **FP32** | 32 bits | 1 | 8 | 23 | 127 | $\approx 3.4 \times 10^{38}$ | $\approx 1.18 \times 10^{-38}$ | Master training weights |
| **FP16** | 16 bits | 1 | 5 | 10 | 15 | $65,504$ | $6.10 \times 10^{-5}$ | Legacy inference (V100/T4) |
| **BF16** | 16 bits | 1 | 8 | 7 | 127 | $\approx 3.39 \times 10^{38}$ | $\approx 1.18 \times 10^{-38}$ | Modern LLM standard (A100/H100) |
| **FP8 (E4M3)** | 8 bits | 1 | 4 | 3 | 7 | **$448.0$** | $0.0156$ | **Inference Weights & Activations** |
| **FP8 (E5M2)** | 8 bits | 1 | 5 | 2 | 15 | **$57,344.0$** | $6.10 \times 10^{-5}$ | Training Gradients & KV Cache |

### Why BF16 Replaced FP16
Notice that FP16 only has 5 exponent bits, giving it a maximum value of $65,504$. In deep LLMs with hundreds of layers, activation values and gradients during training frequently exceed $65,504$, causing immediate **overflow to `+Infinity` / `NaN`**, crashing training runs.

Google Brain invented **Bfloat16 (BF16)**: they preserved the full 8-bit exponent of FP32 (preventing underflow and overflow) while sacrificing mantissa bits down to 7. Modern foundation models (Llama 3, DeepSeek, Mistral) are pre-trained natively in BF16.

---

## 3. FP8: The Production Sweet Spot

The release of NVIDIA's Hopper architecture (H100) and Ada Lovelace introduced dedicated **FP8 Tensor Cores**, establishing 8-bit floating point as the gold standard for high-performance inference.

Notice from our table that the Open Compute Project (OCP) standardized two completely different FP8 variants:

### 1. FP8 E4M3 (1 sign, 4 exponent, 3 mantissa)
* **Maximum value**: $448.0$
* **Precision**: 3 bits of mantissa $\to 2^3 = 8$ discrete levels between each power of 2.
* **Why it dominates inference**: Neural network weights and layer activations cluster closely around zero (typically in the range $[-3.0, 3.0]$). They do not require a massive exponent range up to $57,000$. They desperately need mantissa precision to preserve subtle weight differences. E4M3 offers twice the resolution of E5M2.

### 2. FP8 E5M2 (1 sign, 5 exponent, 2 mantissa)
* **Maximum value**: $57,344.0$
* **Precision**: Only 2 bits of mantissa $\to 2^2 = 4$ discrete levels between powers of 2.
* **Why it dominates gradients & long KV caches**: Gradients and certain attention values can exhibit massive dynamic variance across long sequences, making E5M2's wider exponent range essential to prevent numerical clipping.

### The First-Principles Speedup of FP8
Let us compute the hardware throughput for **Llama-3-70B** on an **8x NVIDIA H100 SXM5** cluster:
* Model Parameter Count: $70 \times 10^9$ parameters.
* Memory Bandwidth across 8 GPUs: $8 \times 3.35 \text{ TB/s} = \mathbf{26.8 \text{ TB/s}}$.

$$\text{Weight Footprint in BF16} = 70 \times 10^9 \times 2 \text{ bytes} = \mathbf{140 \text{ GB}}$$
$$\text{Weight Footprint in FP8} = 70 \times 10^9 \times 1 \text{ byte} = \mathbf{70 \text{ GB}}$$

$$\text{Decode Latency Floor (BF16)} = \frac{140 \times 10^9 \text{ Bytes}}{26.8 \times 10^{12} \text{ Bytes/sec}} \approx \mathbf{5.22 \text{ ms/token}} \implies \mathbf{191 \text{ tok/s}}$$
$$\text{Decode Latency Floor (FP8)} = \frac{70 \times 10^9 \text{ Bytes}}{26.8 \times 10^{12} \text{ Bytes/sec}} \approx \mathbf{2.61 \text{ ms/token}} \implies \mathbf{383 \text{ tok/s}}$$

**Switching to FP8 halves memory traffic and doubles single-stream generation throughput with virtually zero loss in perplexity!**

---

## 4. Integer Quantization: INT8 and INT4

While floating-point preserves non-linear spacing between numbers, integer quantization maps floating-point tensors onto a uniform, linear discrete integer grid.

### The Linear Quantization Formula
To map an arbitrary continuous float tensor $X \in [\alpha, \beta]$ into an $n$-bit signed integer grid $Q \in [-2^{n-1}, 2^{n-1}-1]$:

$$q = \text{clip}\left(\left\lfloor \frac{x}{S} \right\rceil + Z, \; -2^{n-1}, \; 2^{n-1}-1\right)$$

Where:
* $S$ is the **Scale Factor** (a floating-point number representing the step size between integer bins):
  $$S = \frac{\beta - \alpha}{2^n - 1}$$
* $Z$ is the **Zero-Point** (an integer representing the float value $0.0$ on the quantized grid).
* $\lfloor \cdot \rceil$ denotes rounding to the nearest integer.

To reconstruct the approximate float values during computation (**Dequantization**):
$$\hat{x} = (q - Z) \times S$$

```
Linear Quantization Grid (INT4: 16 discrete bins from -8 to +7):
Float Domain:    -1.2   -0.9   -0.6   -0.3    0.0    0.3    0.6    0.9    1.2
                  │      │      │      │      │      │      │      │      │
Quantized Grid:  [-8]   [-6]   [-4]   [-2]   [ 0]   [ 2]   [ 4]   [ 6]   [ 7]
```

### Symmetric vs Asymmetric Quantization:
* **Symmetric ($Z = 0$)**: The range is forced to be symmetric around zero ($[-\alpha, \alpha]$).
  $$S = \frac{\max(|X|)}{2^{n-1} - 1}$$
  Eliminates the zero-point addition, simplifying matrix multiplications into pure integer GEMMs.
* **Asymmetric ($Z \ne 0$)**: Accommodates skewed distributions (such as ReLU activations where values are strictly $\ge 0$).

---

## 5. The INT4 Accuracy Barrier & AWQ (Activation-Aware Weight Quantization)

Compressing models down to INT4 represents a **$4\times$ reduction in memory footprint** compared to BF16 (Llama-3-70B drops from $140\text{ GB}$ to just $35\text{ GB}$, fitting comfortably onto a single node of smaller enterprise GPUs).

However, naive INT4 quantization severely degrades model reasoning and accuracy. Why?

### The Outlier Weight Paradox
In 2022, researchers discovered that in large language models, weight importance is not uniform:
* Across billions of parameters, a tiny fraction ($\approx 0.1\% - 1.0\%$) of channels exhibit **massive activation magnitudes**.
* These are known as **Salient Channels**. If these specific weights are quantized naively to 4-bit integers with a coarse scale factor, the rounding error completely destroys the transformer's attention routing and mathematical capabilities.

```
Activation Outlier Phenomenon:
Channel 0:  [ 0.12,  0.08, -0.05,  0.11 ] -> Activation: 0.8
Channel 1:  [ 0.04, -0.02,  0.01, -0.03 ] -> Activation: 0.2
Channel 2:  [ 0.85, -0.92,  1.10, -0.78 ] -> Activation: 142.5  <-- OUTLIER!
                                                                     Carries 90% of signal!
```

### The AWQ Breakthrough
Published by Lin et al. (MIT, MLSys '24), **Activation-Aware Weight Quantization (AWQ)** introduced a brilliant insight:
1. **Weights don't exist in a vacuum**: A weight is only as important as the activation it multiplies!
2. Instead of quantizing weights based solely on weight magnitudes (like RTN - Round-To-Nearest), AWQ profiles the model on a small calibration dataset to observe **activation magnitudes ($s_X$)**.
3. It identifies the top $1\%$ salient weight channels.
4. Instead of keeping those weights in FP16 (which ruins kernel uniformity), AWQ applies a per-channel **mathematical protection scale ($s$)**:
   $$W' = W \cdot \text{diag}(s), \quad X' = \text{diag}(s)^{-1} \cdot X$$
   Notice that the mathematical output is completely unchanged:
   $$Y = X' W' = \left(X s^{-1}\right) (W s) = X W$$
5. Multiplying the salient weights by $s$ expands their dynamic range, dramatically reducing their relative rounding error when mapped onto the INT4 grid!

The result: **INT4-AWQ achieves near-lossless accuracy** compared to original FP16 weights across benchmarks:

| Model | Precision | WikiText Perplexity (Lower is better) | MMLU Score (Higher is better) | VRAM Footprint |
|---|---|---|---|---|
| **Llama-3-70B** | BF16 | $2.85$ | $79.2\%$ | $140\text{ GB}$ |
| **Llama-3-70B** | FP8 (E4M3) | $2.87$ | $79.1\%$ | $70\text{ GB}$ |
| **Llama-3-70B** | INT4 (Naive RTN) | **$18.42$ (Destroyed)** | **$34.1\%$** | $35\text{ GB}$ |
| **Llama-3-70B** | **INT4 (AWQ)** | **$2.92$ (Near Lossless)** | **$78.6\%$** | **$35\text{ GB}$** |

---

## 6. Marlin: Line-Rate Dequantization Kernels

Having an INT4 model on disk is useless if your GPU spends more time unpacking bits than it saves on memory bandwidth.

In naive INT4 serving (often called W4A16: 4-bit weights, 16-bit activations):
* Weights are fetched as 4-bit integers from HBM.
* The GPU must unpack and dequantize them into FP16 values before feeding them into Tensor Cores.
* If written poorly, the scalar ALU instructions spent shifting bits (`>> 4`, `& 0x0F`) cause severe compute bottlenecks, resulting in performance *slower* than native FP16!

### The Marlin Architecture
Developed by Frantar et al. (IST Austria), **Marlin** is an ultra-optimized CUDA/Triton kernel designed specifically for high-throughput LLM serving:
1. **Asynchronous HBM Loads**: Marlin uses NVIDIA `cp.async` instructions to stream 4-bit quantized weights directly from global HBM into GPU Shared Memory (SRAM), bypassing intermediate registers.
2. **Register-Level Bit Manipulation**: Unpacking from INT4 to FP16 occurs directly inside GPU register files using SIMD bit-permutation instructions (`prmt`).
3. **Line-Rate Dequantization**: Dequantization is completely fused and overlapped with the memory fetch pipeline. The Tensor Cores execute FP16 matrix multiplications at maximum clock frequency while the memory bus operates at full line-rate saturation.

```
Marlin W4A16 Execution Pipeline:
[ GPU HBM (35 GB) ]
       │
       │ (Transfers INT4 packed weights at 3.35 TB/s - 4x faster!)
       ▼
[ Shared Memory / Registers ]
       │
       │ (Fused Hardware Unpacking: INT4 -> FP16 at register clock rate)
       ▼
[ Tensor Cores (FP16 MMA) ] <─── [ Input Activations (FP16) ]
       │
       ▼
[ Output Hidden State (FP16) ]
```

---

## 7. Practical Engineering Decision Matrix

When should an ML systems architect choose BF16, FP8, or INT4-AWQ?

```
                    ┌──────────────────────────────────────┐
                    │ What is your primary bottleneck?     │
                    └──────────────────┬───────────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
       [ Hardware Constraints ]                       [ Quality Demands ]
                │                                             │
      ┌─────────┴─────────┐                         ┌─────────┴─────────┐
      ▼                   ▼                         ▼                   ▼
Limited VRAM /      Modern Cluster            Zero degradation     Reasoning / Math
Smaller GPUs        (H100 / L40S)             tolerance (Medical/  agents where FP8
(e.g. 1x A100-80G)  (Native FP8 Cores)        Financial R&D)       is supported
      │                   │                         │                   │
      ▼                   ▼                         ▼                   ▼
 [ INT4-AWQ/Marlin ]   [ FP8 (E4M3) ]           [ BF16 Base ]      [ FP8 (E4M3) ]
  4x VRAM saving       2x VRAM saving            No calibration     Near-zero loss
  4x Decode boost      2x Decode boost           Maximum cost       Industry default
```

---

## 8. Key Takeaways

1. **Quantization is a bandwidth optimization**: In memory-bound decoding, reducing bitwidth directly accelerates token generation speed by reducing HBM traffic.
2. **Precision vs Range**: Exponents dictate dynamic range (preventing overflow); mantissas dictate numerical precision (resolving small differences).
3. **FP8 is the modern default**: FP8 E4M3 delivers a $2\times$ decode speedup with negligible accuracy degradation on modern Hopper/Ada GPUs.
4. **Outliers ruin naive INT4**: $0.1\%$ of activation channels carry the majority of signal; quantizing them with naive clipping destroys model reasoning.
5. **AWQ protects salient weights**: By scaling weights relative to activation magnitudes, AWQ achieves near-lossless INT4 compression.
6. **Marlin unlocks INT4 hardware limits**: Fused register-level dequantization ensures that 4-bit weight savings translate directly into real-world $3.5\times - 4\times$ latency reductions.
