# 01. Hardware-Bound Physics & The Roofline Model for LLM Inference

> *"You can't optimize what you can't measure, and you can't measure what you don't understand."*

Before we can optimize any LLM, we must understand what the GPU actually does.
Every trick in the modern inference playbook — quantization, continuous batching, speculative
decoding, tensor parallelism — is ultimately a response to **hard physical limits** baked into
the silicon. This chapter builds the mental model you need to reason about *why* each
optimization exists, from absolute first principles.

---

## 1. Why Hardware Matters

A GPU has two fundamental resources:

| Resource | What It Is | Analogy |
|---|---|---|
| **Compute** (ALUs / Tensor Cores) | Arithmetic circuits that multiply and add numbers | Workers on an assembly line |
| **Memory Bandwidth** (HBM ↔ Compute) | The data pipe between storage (HBM) and compute units | The conveyor belt bringing parts to workers |

Think of a factory with 100 workers but a single narrow conveyor belt. The workers can
assemble 1,000 widgets per hour, but the belt only delivers enough parts for 200 widgets
per hour. **80% of your workers sit idle**, waiting for parts.

```
 ┌─────────────────────────────────────────────────────────┐
 │                        GPU Die                          │
 │                                                         │
 │   ┌──────────┐     Conveyor Belt (Bandwidth)            │
 │   │          │  ◄══════════════════════════════════►     │
 │   │   HBM    │     ║  ║  ║  ║  ║  ║  ║  ║  ║  ║       │
 │   │ (Weights,│     ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼  ▼       │
 │   │  KV Cache│   ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐      │
 │   │  Activ.) │   │SM│SM│SM│SM│SM│SM│SM│SM│SM│SM│      │
 │   │          │   │  │  │  │  │  │  │  │  │  │  │      │
 │   └──────────┘   └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘      │
 │                   Workers (Tensor Cores / ALUs)          │
 └─────────────────────────────────────────────────────────┘
```

This is the **central tension** of GPU computing:

> **If compute is fast but the conveyor belt is slow, workers starve for data.**
> **If the conveyor belt is fast but workers are slow, data piles up unused.**

Every LLM inference workload lives somewhere on this spectrum. Understanding *where*
is the key to knowing which optimization will actually help.

---

## 2. What Is a FLOP?

A **FLOP** (Floating Point OPeration) is one arithmetic operation on decimal numbers.

- Multiplying two numbers: `3.14 × 2.71` = **1 FLOP**
- Adding two numbers: `3.14 + 2.71` = **1 FLOP**
- A fused multiply-add (FMA): `a × b + c` = **2 FLOPs** (one multiply + one add)

### Matrix Multiplication FLOPs

For a matrix multiply $C = A \times B$ where $A$ is $(M \times K)$ and $B$ is $(K \times N)$:

$$\text{FLOPs} = 2 \times M \times K \times N$$

Why the factor of 2? Each element of $C$ requires $K$ multiplications and $K-1$ additions
(approximately $K$ adds). So each element costs $\approx 2K$ FLOPs, and there are $M \times N$
elements in $C$.

### Concrete Example: (2×3) × (3×2)

```
A = [1  2  3]      B = [7   8]
    [4  5  6]          [9  10]
                       [11 12]

C[0,0] = 1×7 + 2×9 + 3×11 = 7 + 18 + 33 = 58
         ───   ───   ────
         3 multiplies + 2 adds = 5 ops ≈ 2×K = 6 FLOPs
```

Total FLOPs = $2 \times M \times K \times N = 2 \times 2 \times 3 \times 2 = 24$ FLOPs.

### GPU Compute in TFLOPS

GPU specs report peak compute in **TFLOPS** (Tera-FLOPs per second = $10^{12}$ FLOPs/s).

| GPU | BF16 Tensor Core (Dense) | With Sparsity |
|-----|--------------------------|---------------|
| **H100 SXM** | 989 TFLOPS | 1,979 TFLOPS |
| **A100 SXM (80GB)** | 312 TFLOPS | 624 TFLOPS |
| **L40S** | 362 TFLOPS | 733 TFLOPS |

> **Note:** We use the **dense** (non-sparse) BF16 tensor core numbers throughout this
> document. Structured sparsity (2:4) requires special weight formats and is not universally
> applicable. Dense numbers represent the guaranteed compute ceiling.

An H100 can perform **989 trillion** multiply-add operations every single second. That's
roughly the number of grains of sand on Earth — *every second*.

---

## 3. What Is Memory Bandwidth?

### The Problem: Data Lives Far From Compute

The GPU can't compute on data sitting in HBM (High Bandwidth Memory). Data must be
loaded from HBM → L2 cache → registers (on the Streaming Multiprocessor) before any
math can happen.

```
 Speed:     Slow ◄──────────────────────────────► Fast
 Capacity:  Big  ◄──────────────────────────────► Small

 ┌────────────┐    ┌──────────┐    ┌───────────┐
 │    HBM     │───►│ L2 Cache │───►│ Registers │──► Tensor Core
 │  80 GB     │    │  50 MB   │    │ ~20 MB    │    (compute)
 │ ~3.35 TB/s │    │ ~12 TB/s │    │ ~100 TB/s │
 └────────────┘    └──────────┘    └───────────┘
       ▲
    Bottleneck!
```

**Memory Bandwidth** = how fast data moves from HBM to the compute units.

### Real Numbers

| GPU | HBM Type | Capacity | Bandwidth |
|-----|----------|----------|-----------|
| **H100 SXM** | HBM3 | 80 GB | **3,350 GB/s** (3.35 TB/s) |
| **A100 SXM** | HBM2e | 80 GB | **2,039 GB/s** (2.04 TB/s) |
| **L40S** | GDDR6 | 48 GB | **864 GB/s** (0.86 TB/s) |

What does 3,350 GB/s actually mean?

> An H100 can transfer **3,350 gigabytes** from HBM to compute units every second.
> That's like downloading the entire English Wikipedia (~22 GB) **150 times per second**.

But here's the catch: a 70B-parameter model in BF16 is **140 GB**. Even at 3.35 TB/s,
loading the entire model takes:

$$t = \frac{140 \text{ GB}}{3{,}350 \text{ GB/s}} = 0.0418 \text{ s} = 41.8 \text{ ms}$$

That's a hard floor. Physics says you *cannot* generate a single token faster than 41.8 ms
on one H100 for this model. We'll derive this formally in Section 6.

---

## 4. The Roofline Model

The Roofline Model is the single most important mental model for understanding GPU
performance. It tells you: **given my algorithm, am I limited by compute or by memory?**

### Step 1: Define Operational Intensity (OI)

$$OI = \frac{\text{FLOPs (work done)}}{\text{Bytes moved from/to HBM}}$$

The units are **FLOPs/Byte**. It measures how much *useful work* you extract per byte of
data transferred.

- **Low OI** (e.g., 1 FLOP/Byte): You load a lot of data but barely compute on it → **memory bound**
- **High OI** (e.g., 500 FLOPs/Byte): You load a little data and compute intensively → **compute bound**

### Step 2: The Two Ceilings

Your algorithm's actual performance is capped by whichever resource is the bottleneck:

$$\text{Attainable FLOPS} = \min(\text{Peak Compute}, \;\; OI \times \text{Bandwidth})$$

- **Compute ceiling** = Peak TFLOPS of the GPU (flat horizontal line)
- **Memory ceiling** = $OI \times \text{Bandwidth}$ (rising diagonal line)

### Step 3: The Roofline Diagram

```
Performance
(TFLOPS)
    ▲
    │
989 ┤─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  H100 Compute Ceiling
    │                         ╱▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
    │                        ╱
    │                       ╱
    │                      ╱  ← "Knee Point" (OI = 295)
    │                     ╱
    │                    ╱
312 ┤─ ─ ─ ─ ─ ─ ─ ─ ╱─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  A100 Compute Ceiling
    │                ╱
    │               ╱
    │              ╱    Memory-Bound        Compute-Bound
    │             ╱      Region    ──►  ◄──   Region
    │            ╱
    │           ╱   ← Slope = Bandwidth (3.35 TB/s for H100)
    │          ╱
    │         ╱
    │        ╱
    │       ╱
    │      ╱
    │     ╱
    │    ╱
    │   ╱
    │  ╱
    │ ╱
    │╱
    └──────────────────────────────────────────────────────────► OI (FLOPs/Byte)
    0    1    32    64   128   256   295   512   1024
                              ▲
                          Knee Point
                          (H100)
```

### Step 4: Calculate the Knee Point

The **knee point** is where the two ceilings meet — the transition from memory-bound
to compute-bound:

$$OI_{\text{knee}} = \frac{\text{Peak Compute (TFLOPS)}}{\text{Bandwidth (TB/s)}}$$

| GPU | Peak Compute (BF16 Dense) | Bandwidth | Knee Point (FLOPs/Byte) |
|-----|---------------------------|-----------|------------------------|
| **H100 SXM** | 989 TFLOPS | 3.35 TB/s | $\frac{989}{3.35} \approx$ **295** |
| **A100 SXM** | 312 TFLOPS | 2.04 TB/s | $\frac{312}{2.04} \approx$ **153** |
| **L40S** | 362 TFLOPS | 0.864 TB/s | $\frac{362}{0.864} \approx$ **419** |

**What does this mean in practice?**

- On an H100, if your algorithm does fewer than 295 FLOPs per byte loaded, **you are
  memory-bound**. Adding more compute won't help — you need to load less data or load it faster.
- On an L40S, the knee point is 419 — meaning it's **even harder** to become compute-bound.
  Despite having less compute than an H100, the L40S's much lower bandwidth makes it
  proportionally more memory-starved.

---

## 5. Prefill vs Decode: The Two Regimes of LLM Inference

This is the most important section of this document. Every LLM inference request has
two distinct phases, and they live in **completely different** regions of the roofline.

```
 User Prompt (S tokens)          Generated Tokens (one at a time)
 ┌───────────────────┐           ┌───┐ ┌───┐ ┌───┐ ┌───┐
 │ "Tell me about    │           │ T │ │ T │ │ T │ │ T │ ...
 │  the history of   │  ──────►  │ 1 │ │ 2 │ │ 3 │ │ 4 │
 │  quantum physics" │           └───┘ └───┘ └───┘ └───┘
 └───────────────────┘
   PREFILL (parallel)            DECODE (sequential, autoregressive)
   Matrix × Matrix               Matrix × Vector
   GEMM                          GEMV
   Compute-bound                 Memory-bound
```

### 5.1 Prefill: Matrix-Matrix Multiply (GEMM)

During prefill, all $S$ input tokens are processed **simultaneously**. Each linear layer
computes:

$$Y = X \cdot W$$

where $X$ is $(S \times d)$ and $W$ is $(d \times d)$ (for hidden dimension $d$).

### 5.2 Decode: Matrix-Vector Multiply (GEMV)

During decode, we process **one new token at a time**. Each linear layer computes:

$$y = x \cdot W$$

where $x$ is $(1 \times d)$ and $W$ is $(d \times d)$.

### 5.3 Full Numerical Example: Llama-3-8B

Let's use Llama-3-8B with `hidden_dim = 4096` and BF16 weights (2 bytes per parameter).

Consider a single linear layer with weight matrix $W$ of shape $(4096 \times 4096)$.

**Weight size:**

$$\text{Bytes}(W) = 4096 \times 4096 \times 2 = 33{,}554{,}432 \text{ bytes} \approx 33.6 \text{ MB}$$

---

#### Case 1: Prefill with S = 2048 tokens

**FLOPs:**

$$\text{FLOPs} = 2 \times S \times d \times d = 2 \times 2048 \times 4096 \times 4096 = 68{,}719{,}476{,}736 \approx 68.7 \text{ GFLOPs}$$

**Bytes loaded from HBM:**
- Weight matrix $W$: 33.6 MB
- Input activations $X$ $(2048 \times 4096 \times 2)$: 16.8 MB
- Total ≈ **50.3 MB** (we focus on weights as they dominate for multi-layer models)

For just the weight load (which dominates):

$$OI_{\text{prefill}} = \frac{68.7 \times 10^9 \text{ FLOPs}}{33.6 \times 10^6 \text{ Bytes}} = \frac{68.7 \text{ G}}{0.0336 \text{ G}} \approx \mathbf{2{,}048 \text{ FLOPs/Byte}}$$

This equals $S$ (the sequence length)! This is not a coincidence — it's a fundamental
relationship. With $S = 2048$:

$$OI_{\text{prefill}} = \frac{2 \times S \times d^2}{2 \times d^2} = S = 2048$$

**2,048 >> 295 (H100 knee point) → Prefill is deeply COMPUTE-BOUND** ✅

The GPU's compute units are fully saturated. Memory bandwidth is *not* the bottleneck.

---

#### Case 2: Decode with S = 1 token

**FLOPs:**

$$\text{FLOPs} = 2 \times 1 \times 4096 \times 4096 = 33{,}554{,}432 \approx 33.6 \text{ MFLOPs}$$

**Bytes loaded from HBM:**
- Weight matrix $W$: 33.6 MB (same — the ENTIRE weight matrix must be loaded!)
- Input activation $x$ $(1 \times 4096 \times 2)$: 8 KB (negligible)
- Total ≈ **33.6 MB**

$$OI_{\text{decode}} = \frac{33.6 \times 10^6 \text{ FLOPs}}{33.6 \times 10^6 \text{ Bytes}} = \mathbf{1 \text{ FLOP/Byte}}$$

**1 << 295 (H100 knee point) → Decode is deeply MEMORY-BOUND** ❌

The GPU loads 33.6 MB of data but only does 33.6 million FLOPs of work. The tensor cores
are almost completely idle. You're paying the full bandwidth cost to load weights but getting
almost no compute reuse.

---

#### Where Each Falls on the Roofline

```
Performance
(TFLOPS)
    ▲
989 ┤─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ★ PREFILL (OI=2048)
    │                         ╱▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
    │                        ╱   (hitting compute ceiling)
    │                       ╱
    │                      ╱
    │                     ╱
    │                    ╱
    │                   ╱
    │                  ╱
    │                 ╱
    │                ╱
    │               ╱
    │              ╱
    │             ╱
    │            ╱
    │           ╱
    │          ╱
    │         ╱
    │        ╱
    │       ╱
  3.35 ┤  ★ DECODE (OI=1)
    │     ╱  (far below compute ceiling, stuck on bandwidth slope)
    │    ╱
    │   ╱
    └──────────────────────────────────────────────────────────► OI
    0  1    32    128   295   512        2048
```

**The decode point achieves only ~3.35 TFLOPS out of 989 TFLOPS available — that's
0.34% utilization of the H100's compute!**

This is the fundamental insight: **single-token decode wastes >99% of the GPU's compute
capability** because it's entirely bottlenecked on memory bandwidth.

---

## 6. The Decode Latency Floor

Since decode is memory-bound, the time per token is dictated by how fast you can stream
model weights from HBM. Every decode step must reload **ALL** model weights through the
compute pipeline.

### The Formula

$$t_{\text{token}} \geq \frac{\text{Model Size (Bytes)}}{\text{HBM Bandwidth (Bytes/s)}}$$

This is a **hard physical floor**. No software optimization can beat it (at batch size 1).

### Worked Example: Llama-3-70B

Llama-3-70B has 70 billion parameters.

| Precision | Bytes/Param | Model Size | H100 (3.35 TB/s) | A100 (2.04 TB/s) |
|-----------|-------------|------------|-------------------|-------------------|
| **BF16** | 2 | 140 GB | $\frac{140}{3350}$ = **41.8 ms** | $\frac{140}{2039}$ = **68.7 ms** |
| **FP8** | 1 | 70 GB | $\frac{70}{3350}$ = **20.9 ms** | $\frac{70}{2039}$ = **34.3 ms** |
| **INT4** | 0.5 | 35 GB | $\frac{35}{3350}$ = **10.4 ms** | $\frac{35}{2039}$ = **17.2 ms** |

Let's convert to **tokens per second** (at batch size 1):

| Precision | H100 tok/s | A100 tok/s | Speedup vs BF16 |
|-----------|------------|------------|-----------------|
| **BF16** | $\frac{1}{0.0418}$ = **23.9** | $\frac{1}{0.0687}$ = **14.6** | 1× |
| **FP8** | $\frac{1}{0.0209}$ = **47.8** | $\frac{1}{0.0343}$ = **29.2** | **2×** |
| **INT4** | $\frac{1}{0.0104}$ = **95.7** | $\frac{1}{0.0172}$ = **58.3** | **4×** |

### This Is WHY Quantization Matters

Look at that table carefully:

```
 BF16 ████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░  23.9 tok/s
 FP8  ████████████████████████████████████████████████░░  47.8 tok/s  (2× faster!)
 INT4 ████████████████████████████████████████████████████  95.7 tok/s  (4× faster!)
```

Quantization from BF16 → INT4 gives a **4× speedup** not because the math is faster,
but because **you're moving 4× less data across the memory bus**. The bottleneck was
never compute — it was always bandwidth.

> **Key insight:** Quantization for inference is fundamentally a *bandwidth optimization*,
> not a compute optimization. Smaller numbers = less data to move = faster tokens.

---

## 7. Batch Size and Operational Intensity

If single-token decode has $OI = 1$ (deeply memory-bound), can we do better?
**Yes — by batching multiple requests together.**

### The Math of Batching

With batch size $B$, a linear layer computes:

$$Y = X \cdot W \quad \text{where } X \text{ is } (B \times d), \; W \text{ is } (d \times d)$$

- **FLOPs** = $2 \times B \times d \times d$ → scales linearly with $B$
- **Bytes loaded** = $2 \times d \times d$ (weights) + $2 \times B \times d$ (activations) ≈ $2 d^2$ for small $B$
  - Weights are loaded **once** and reused across all $B$ requests!

$$OI(B) = \frac{2 \times B \times d^2}{2 \times d^2 + 2 \times B \times d} = \frac{B \times d}{d + B}$$

For $d = 4096$:

| Batch Size $B$ | FLOPs | Bytes Loaded | OI (FLOPs/Byte) | Regime on H100 |
|---|---|---|---|---|
| 1 | 33.6 M | 33.6 MB | **1.0** | Deep memory-bound |
| 8 | 268 M | 33.6 MB | **8.0** | Memory-bound |
| 32 | 1.07 G | 33.8 MB | **31.8** | Memory-bound |
| 64 | 2.15 G | 34.1 MB | **63.0** | Memory-bound |
| 128 | 4.29 G | 34.6 MB | **124** | Memory-bound |
| 256 | 8.59 G | 35.6 MB | **241** | Approaching knee (295) |
| 295 | 9.91 G | 36.1 MB | **274** | Near knee point |
| 512 | 17.2 G | 38.0 MB | **452** | **Compute-bound!** |

```
Performance     H100 Roofline
(TFLOPS)
    ▲
989 ┤─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ★ B=512 ─ ─ ─ ─
    │                       ╱▔▔▔▔▔▔★ B=295
    │                      ╱
    │                     ╱
    │                   ★ B=256
    │                  ╱
    │                ╱
    │              ★ B=128
    │            ╱
    │          ╱
    │        ★ B=64
    │       ╱
    │     ★ B=32
    │    ╱
    │  ★ B=8
    │ ╱
    │★ B=1
    └──────────────────────────────────────────────────► OI
```

### What Batching Really Does

At $B = 1$: you load 33.6 MB of weights, do 33.6M FLOPs. **1 FLOP per byte.** Terrible.

At $B = 256$: you load 33.6 MB of weights (same!), but do 8.59G FLOPs. **241 FLOPs per byte.** 

The weights are loaded **once** and reused 256 times. You amortize the bandwidth cost
across all requests in the batch.

### This is WHY Continuous Batching Was Invented

Traditional static batching wastes GPU time: when one request finishes, its slot sits empty
until the entire batch completes. **Continuous batching** (pioneered by Orca/vLLM) dynamically
inserts new requests into freed slots, keeping $B$ as high as possible at all times.

Higher $B$ → higher $OI$ → closer to the compute ceiling → **more tokens per second per dollar**.

```
 Static Batching:                    Continuous Batching:
 ┌────────────────────┐              ┌────────────────────┐
 │ Req 1 ████████░░░░ │              │ Req 1 ████████     │
 │ Req 2 ██████████░░ │              │ Req 2 ██████████   │
 │ Req 3 ████░░░░░░░░ │              │ Req 3 ████ Req5 ███│
 │ Req 4 ████████████ │              │ Req 4 ████████████ │
 │       ░ = wasted   │              │ Req 5      ████████│
 └────────────────────┘              └────────────────────┘
   Batch waits for                     New requests fill
   longest request                     freed slots immediately
```

---

## 8. Practical Example: Sizing a Cluster

**Goal:** Serve Llama-3-70B in FP8 at a sustained throughput of **1,000 tokens/second**.

### Step 1: Single H100 Maximum Throughput

From Section 6, the theoretical minimum time per token at $B = 1$ is:

$$t_{\text{token}} = \frac{70 \text{ GB}}{3{,}350 \text{ GB/s}} = 20.9 \text{ ms}$$

But with batching, we amortize weight loads. At batch size $B$, the time per batch-step is
approximately the same (still bandwidth-limited for moderate $B$), producing $B$ tokens:

$$\text{Throughput} \approx \frac{B}{t_{\text{token}}} = B \times \frac{\text{Bandwidth}}{\text{Model Size}}$$

For the transition to compute-bound (at the knee point), maximum throughput becomes:

$$\text{Throughput}_{\text{max}} = \frac{\text{Peak Compute (FLOPs/s)}}{2 \times \text{params}}$$

For H100 with Llama-3-70B FP8 (using FP8 tensor cores at ~1,979 TFLOPS):

$$\text{Throughput}_{\text{max}} = \frac{1{,}979 \times 10^{12}}{2 \times 70 \times 10^{9}} = \frac{1.979 \times 10^{15}}{1.4 \times 10^{11}} \approx 14{,}136 \text{ tok/s}$$

But this is the **theoretical maximum** at 100% compute utilization, which never happens.
In practice, accounting for:
- Memory capacity limits on batch size and KV cache (~60-70% of theoretical)
- Attention computation overhead (~10-15%)
- Kernel launch overhead and pipeline bubbles (~5-10%)

**Realistic single-H100 throughput: ~2,000-3,500 tok/s** (depending on sequence length
and KV cache pressure).

### Step 2: But Wait — Can 70B Even Fit on One H100?

Llama-3-70B in FP8 = 70 GB of weights. H100 has 80 GB HBM3. That leaves only
10 GB for KV cache, activations, and CUDA overhead. At reasonable sequence lengths
(2K-4K tokens) with batch sizes needed for throughput, this is **too tight**.

**Solution: Tensor Parallelism (TP) across multiple GPUs.**

Common deployment: **TP=2** (2 × H100) or **TP=4** (4 × H100).

### Step 3: Tensor Parallelism Overhead

With TP = $N$ GPUs, each GPU holds $\frac{1}{N}$ of the weights. But every layer requires
an **AllReduce** synchronization across GPUs via NVLink:

| TP Size | Weight per GPU | NVLink AllReduce per Layer | Communication Overhead |
|---------|---------------|---------------------------|----------------------|
| TP=1 | 70 GB | 0 | 0% |
| TP=2 | 35 GB | ~8 KB × $d$ | ~5-10% |
| TP=4 | 17.5 GB | ~16 KB × $d$ | ~10-20% |
| TP=8 | 8.75 GB | ~32 KB × $d$ | ~20-35% |

NVLink bandwidth on H100: **900 GB/s** bidirectional per GPU.

With TP=2:
- Each GPU loads 35 GB per decode step → $t = \frac{35}{3350} = 10.4$ ms per token per step
- AllReduce overhead ≈ 0.5-1.0 ms per layer × 80 layers ≈ additional ~2-3 ms
- Effective: ~13 ms per token at $B = 1$
- At higher batch sizes (B=64-128): ~2,500-4,000 tok/s aggregate across 2 GPUs

### Step 4: Final Sizing

| Configuration | GPUs | Throughput (tok/s) | Target Coverage |
|---|---|---|---|
| 1× TP=2 (2 H100s) | 2 | ~3,000 | 3× of target |
| 1× TP=4 (4 H100s) | 4 | ~4,500 | 4.5× of target |

For **1,000 tok/s with reliability margin** (targeting ~33% utilization for burst handling):

> **Recommendation: 1 replica of TP=2 (2 × H100 SXM) for Llama-3-70B FP8 at 1,000 tok/s.**

With a safety factor for production (2-3× headroom for bursts, tail latency):

> **Production recommendation: 2 replicas of TP=2 = 4 × H100 SXM total.**

```
 Client Requests
       │
       ▼
 ┌─────────────┐
 │ Load Balancer│
 └──────┬──────┘
   ┌────┴────┐
   ▼         ▼
┌──────┐  ┌──────┐
│Rep 1 │  │Rep 2 │
│GPU 0 │  │GPU 2 │
│GPU 1 │  │GPU 3 │
│(TP=2)│  │(TP=2)│
└──────┘  └──────┘
 ~3K t/s   ~3K t/s
    Combined: ~6K tok/s capacity
    (for 1K sustained with headroom)
```

---

## 9. Putting It All Together: The Optimization Landscape

Now you can understand *why* every major inference optimization exists:

| Problem (Physics) | Optimization | Section |
|---|---|---|
| Decode is memory-bound ($OI = 1$) | **Quantization** (reduce bytes moved) | §6 |
| Decode is memory-bound ($OI = 1$) | **Batching** (increase OI) | §7 |
| Weight loads dominate | **Continuous batching** (maximize utilization) | §7 |
| KV cache grows with sequence length | **PagedAttention** (efficient memory management) | future doc |
| KV cache bandwidth cost | **GQA / MQA** (fewer KV heads = fewer bytes) | future doc |
| Model doesn't fit on one GPU | **Tensor Parallelism** (shard weights) | §8 |
| Decode generates one token at a time | **Speculative Decoding** (guess multiple tokens) | future doc |
| Prefill is compute-bound | **FlashAttention** (reduce memory traffic for attention) | future doc |

Every single optimization maps back to the roofline model: either **reduce bytes moved**
(shift right on the x-axis) or **increase compute utilization** (shift up on the y-axis).

---

## 10. Key Takeaways

1. **Two resources, one bottleneck.** A GPU has compute (tensor cores) and memory bandwidth (HBM). Your workload is bottlenecked by whichever runs out first. The Roofline Model tells you which one.

2. **Operational Intensity ($OI$) is the key metric.** $OI = \frac{\text{FLOPs}}{\text{Bytes}}$. If $OI$ is below the GPU's knee point, you are memory-bound. If above, you are compute-bound.

3. **Prefill ≠ Decode.** Prefill (processing the prompt) is compute-bound ($OI \approx S$, the sequence length). Decode (generating tokens) is memory-bound ($OI \approx 1$ at batch size 1). They require completely different optimization strategies.

4. **The decode latency floor is $\frac{\text{Model Size}}{\text{Bandwidth}}$.** For Llama-3-70B BF16 on H100, this is 41.8 ms/token — no software can beat this. This is the fundamental reason quantization speeds up inference.

5. **Quantization is a bandwidth optimization, not a compute optimization.** Going BF16→INT4 moves 4× less data per decode step → 4× faster token generation. The math isn't faster; there's just less data on the conveyor belt.

6. **Batching transforms memory-bound into compute-bound.** At $B = 1$, $OI = 1$. At $B = 256$, $OI \approx 241$. Weights are loaded once and reused $B$ times. This is why serving systems obsess over continuous batching and high utilization.

7. **Always check the roofline before optimizing.** If you're memory-bound, making the math faster (better kernels, fancier algorithms) won't help. If you're compute-bound, reducing precision further won't help. Know your bottleneck, then attack it.

---

## Appendix: GPU Quick-Reference Table

| Spec | H100 SXM | A100 SXM (80GB) | L40S | H200 SXM |
|------|----------|------------------|------|----------|
| Architecture | Hopper | Ampere | Ada Lovelace | Hopper |
| HBM Type | HBM3 | HBM2e | GDDR6 | HBM3e |
| Memory | 80 GB | 80 GB | 48 GB | 141 GB |
| Bandwidth | 3,350 GB/s | 2,039 GB/s | 864 GB/s | 4,800 GB/s |
| BF16 Dense TFLOPS | 989 | 312 | 362 | 989 |
| FP8 Dense TFLOPS | 1,979 | — | 733 | 1,979 |
| Knee (BF16) | 295 | 153 | 419 | 206 |
| TDP | 700W | 400W | 350W | 700W |
| NVLink BW | 900 GB/s | 600 GB/s | — | 900 GB/s |

---

*Next up: [02. Quantization Deep Dive](02_quantization_deep_dive.md) — Now that you understand WHY quantization speeds up inference (it's bandwidth!), we'll explore HOW different quantization schemes work.*
