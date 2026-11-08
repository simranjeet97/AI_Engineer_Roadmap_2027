# 07. Speculative Decoding & Long-Context KV Cache Eviction

> *"Why wait for a 70-billion parameter giant to deliberate on every common word when an agile 1-billion apprentice can draft five words in advance and have them verified in a single step?"*

In modern LLM inference, two distinct architectural frontiers push against the physical limits of hardware:
1. **The Single-Token Autoregressive Bottleneck**: The sequential nature of token generation forces the memory bus to reload billions of parameters to output just one single token at a time.
2. **The Long-Context Memory Wall**: As context lengths scale from $4\text{K}$ to $128\text{K}$ and beyond, the Key-Value (KV) cache grows linearly ($O(B \cdot S)$), cannibalizing GPU VRAM and causing concurrency to collapse.

This technical memo derives the solutions to both frontiers from first principles: the statistical mechanics of **Speculative Decoding** (draft-verify pipelines, lossless rejection sampling, acceptance bounds) and **Long-Context KV Cache Compression** (the Attention Sink phenomenon, StreamingLLM, and Heavy Hitter eviction).

---

# Part A: Speculative Decoding

## 1. The Core Dilemma: Autoregressive Memory Starvation

Recall from Chapter 01 the fundamental arithmetic intensity equation for the decode phase:
$$I_{\text{decode}} \approx \frac{2 \times B \times P}{P \times \text{bytes\_per\_weight}} = \frac{2 B}{\text{bytes\_per\_weight}}$$

For a single user stream ($B = 1$) on a $70\text{B}$ parameter model in FP16 ($140\text{ GB}$ of weights):
* To emit **one single token**, the GPU must stream **$140\text{ GB}$ of data across the memory bus**.
* On an NVIDIA H100 SXM5 ($3,350\text{ GB/s}$ memory bandwidth):
  $$t_{\text{token}} \ge \frac{140 \times 10^9}{3.35 \times 10^{12}} \approx \mathbf{41.8\text{ ms}} \implies \mathbf{23.9\text{ tokens/sec}}$$
* During this $41.8\text{ ms}$, the GPU executes only $\approx 1.4 \times 10^{11} \text{ FLOPs}$.
* But the H100 Tensor Cores can execute $989 \times 10^{12} \text{ FLOPs/sec}$, meaning they could have executed $4.13 \times 10^{13} \text{ FLOPs}$ in that same window!

**The GPU Tensor Cores are operating at less than $0.35\%$ of their computational capacity.**
Over $99.6\%$ of the silicon is sitting idle, waiting for memory traffic.

### The Key Question
Can we compute **multiple tokens per forward pass** without retraining the model or altering the output distribution?
The answer is **Speculative Decoding** (Leviathan et al., Chen et al., 2023).

---

## 2. The Draft-and-Verify Pipeline

Speculative decoding relies on a critical asymmetry in modern GPU hardware:
* **Generating $K$ tokens sequentially (one-by-one)** requires $K$ separate memory-bandwidth-bound GEMV passes $\to$ **Takes $K \times 41.8\text{ ms}$**.
* **Evaluating $K$ tokens in parallel (all at once)** is a compute-bound GEMM pass $\to$ **Takes nearly the same time as 1 step ($\approx 43\text{ ms}$)!**

### The Assistant Analogy
Think of an executive writing a complex strategy memo:
* **Without Speculative Decoding**: The CEO writes every single word manually. Common filler phrases ("In accordance with the aforementioned data...") take just as long to write as brilliant strategic breakthroughs.
* **With Speculative Decoding**: An agile junior assistant writes a rough draft of the next 5 words. The CEO reads all 5 words simultaneously in a single glance. If the assistant wrote: `"In accordance with the facts"`, the CEO approves the first 4 words, corrects `"facts"` to `"data"`, and moves forward. Five words were finalized in the time it took the CEO to write one!

```
Speculative Decoding Timeline:
Step 1: Draft Model (Small, fast, e.g. Llama-3-1B)
        Emits K=4 candidate tokens sequentially:
        [ "The" -> "capital" -> "of" -> "France" ]      (Takes ~8ms total)

Step 2: Target Model (Large, e.g. Llama-3-70B)
        Evaluates ALL 4 tokens in ONE parallel forward pass:
        Inputs:  [ prompt, "The", "capital", "of" ]
        Outputs: Probabilities for each position simultaneously! (Takes ~43ms)

Step 3: Statistical Verification (Lossless Rejection Sampling)
        - Token 1 ("The"):     Accepted!
        - Token 2 ("capital"): Accepted!
        - Token 3 ("of"):      Accepted!
        - Token 4 ("France"):  Accepted!
        Result: 4 tokens emitted in 51ms (instead of 4 x 41.8ms = 167ms)!
        Speedup: 3.27x!
```

---

## 3. Lossless Rejection Sampling: Mathematical Proof

A common misconception is that speculative decoding produces an approximation of the large model's output.
**This is mathematically false.** Speculative decoding is proven to be **strictly lossless**: the output probability distribution is guaranteed to be bit-for-bit identical to running the large target model alone.

How is this achieved? Through **Rejection Sampling**.

### The Verification Algorithm
Let:
* $p(x)$: Probability distribution emitted by the **Draft Model** ($M_D$) for candidate token $x$.
* $q(x)$: Probability distribution emitted by the **Target Model** ($M_T$) for token $x$.

For each speculative candidate token $\tilde{x}_i \sim p(x)$ at position $i = 1, \dots, K$:
1. Compute the **Acceptance Ratio**:
   $$\beta = \min\left(1, \; \frac{q(\tilde{x}_i)}{p(\tilde{x}_i)}\right)$$
2. Draw a uniform random number $r \sim \text{Uniform}(0, 1)$.
3. **If $r \le \beta$**: The token is **ACCEPTED**.
4. **If $r > \beta$**: The token is **REJECTED**.
   * Discard candidate $\tilde{x}_i$ and all subsequent speculative tokens $\tilde{x}_{i+1 \dots K}$.
   * Sample a replacement token $x_{corrected}$ from the normalized positive difference distribution:
     $$q'(x) = \frac{\max(0, \; q(x) - p(x))}{\sum_{x'} \max(0, \; q(x') - p(x'))}$$
   * Halt the speculative verification step immediately.

### Mathematical Proof of Equivalence:
We must prove that the marginal probability of emitting token $x$ under this algorithm equals $q(x)$:

$$P(x) = P(\text{accepted}) \cdot P(x \mid \text{accepted}) + P(\text{rejected}) \cdot P(x \mid \text{rejected})$$

$$P(x) = p(x) \cdot \min\left(1, \frac{q(x)}{p(x)}\right) + \left(1 - \sum_{x'} \min(p(x'), q(x'))\right) \cdot \frac{\max(0, q(x) - p(x))}{\sum_{x'} \max(0, q(x') - p(x'))}$$

Notice that $1 - \sum_{x'} \min(p(x'), q(x')) = \sum_{x'} \max(0, q(x') - p(x'))$. The denominator cancels out!
$$P(x) = \min(p(x), q(x)) + \max(0, q(x) - p(x))$$

Since for any two real numbers $a, b$:
$$\min(a, b) + \max(0, a - b) = a \quad \text{(when } a \le b \implies a + 0 = a; \text{ when } a > b \implies b + (a - b) = a\text{)}$$

Substituting $a = q(x)$ and $b = p(x)$:
$$P(x) \equiv q(x) \quad \mathbf{\text{Q.E.D.}}$$

The output tokens follow the exact target distribution $q(x)$. Not a single epsilon of quality is sacrificed!

---

## 4. Expected Acceptance Rate & Speedup Bounds

What determines the speedup of speculative decoding?

Let $\alpha$ denote the average **Token Acceptance Rate** ($\alpha \in [0.0, 1.0]$):
$$\alpha = \mathbb{E}_{x \sim p}\left[\min\left(1, \frac{q(x)}{p(x)}\right)\right]$$

For a speculation lookahead window of $K$ tokens:
* The probability of accepting $0$ tokens is $1 - \alpha$.
* The probability of accepting exactly $1$ token is $\alpha(1 - \alpha)$.
* The probability of accepting all $K$ tokens is $\alpha^K$.

The **Expected Number of Generated Tokens per Step** ($\mathbb{E}[N]$) follows a truncated geometric sum:
$$\mathbb{E}[N] = 1 + \sum_{i=1}^{K} \alpha^i = \frac{1 - \alpha^{K+1}}{1 - \alpha}$$

```
Expected Tokens Generated per Target Forward Pass (vs Lookahead K):
┌───────┬────────────┬────────────┬────────────┬────────────┐
│ α     │ K = 1      │ K = 3      │ K = 5      │ K = 8      │
├───────┼────────────┼────────────┼────────────┼────────────┤
│ 0.50  │ 1.50 tok   │ 1.88 tok   │ 1.97 tok   │ 1.99 tok   │
│ 0.70  │ 1.70 tok   │ 2.47 tok   │ 2.94 tok   │ 3.20 tok   │
│ 0.85  │ 1.85 tok   │ 3.02 tok   │ 3.86 tok   │ 4.67 tok   │
│ 0.95  │ 1.95 tok   │ 3.66 tok   │ 5.07 tok   │ 6.74 tok   │
└───────┴────────────┴────────────┴────────────┴────────────┘
```

### The Realized Wall-Clock Speedup Formula:
$$\text{Speedup} = \frac{\mathbb{E}[N] \times t_{\text{target}}}{K \cdot t_{\text{draft}} + t_{\text{target}}}$$

Where:
* $t_{\text{target}}$: Time for target model forward pass ($\approx 41.8\text{ ms}$).
* $t_{\text{draft}}$: Time for draft model forward pass ($\approx 2.5\text{ ms}$).

### Numerical Example:
For Llama-3-70B with a Llama-3-1B draft model ($K = 5, \alpha = 0.75$):
$$\mathbb{E}[N] = \frac{1 - 0.75^6}{1 - 0.75} = \frac{1 - 0.178}{0.25} \approx \mathbf{3.29 \text{ tokens/step}}$$
$$\text{Cost per Step} = 5 \times 2.5\text{ ms} + 41.8\text{ ms} = 12.5 + 41.8 = \mathbf{54.3\text{ ms}}$$
$$\text{Time per Token} = \frac{54.3\text{ ms}}{3.29} \approx \mathbf{16.5\text{ ms/token}}$$
$$\text{Effective Throughput} = \frac{1000}{16.5} \approx \mathbf{60.6 \text{ tok/s}} \quad (\mathbf{2.53\times \text{ faster than baseline 23.9 tok/s}!})$$

---

# Part B: Long-Context KV Cache Eviction

## 5. The Long-Context Memory Wall

As foundation models support context windows up to $128\text{K}$, $1\text{M}$, and $10\text{M}$ tokens, serving infrastructure encounters an insurmountable physical barrier: **KV Cache Memory Saturation**.

Recall the memory formula for a single request's KV cache:
$$\text{KV Bytes per Token} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{bytes\_per\_elem}$$

For **Llama-3-70B (GQA: 80 layers, 8 KV heads, head dim 128, FP16)**:
$$\text{KV Bytes per Token} = 2 \times 80 \times 8 \times 128 \times 2 = \mathbf{327,680 \text{ Bytes}} \approx \mathbf{320 \text{ KB / token}}$$

Let us compute the KV cache memory consumed by a single request across varying context lengths:
* Context $2,048$ tokens: $320 \text{ KB} \times 2,048 \approx \mathbf{0.65 \text{ GB}}$
* Context $8,192$ tokens: $320 \text{ KB} \times 8,192 \approx \mathbf{2.62 \text{ GB}}$
* Context $32,768$ tokens: $320 \text{ KB} \times 32,768 \approx \mathbf{10.48 \text{ GB}}$
* Context $131,072$ tokens ($128\text{K}$): $320 \text{ KB} \times 131,072 \approx \mathbf{41.94 \text{ GB / request!}}$

### The Catastrophic Concurrency Collapse:
On an 8x NVIDIA H100 cluster ($640\text{ GB}$ total VRAM):
* Model weights consume $140\text{ GB}$ (in FP16).
* Available KV VRAM = $640 - 140 = \mathbf{500 \text{ GB}}$.

```
Cluster Concurrency vs Context Length:
Context:  4,096 tokens   -> Max Concurrency: 382 concurrent streams
Context: 32,768 tokens   -> Max Concurrency:  47 concurrent streams
Context: 131,072 tokens  -> Max Concurrency:  11 concurrent streams! (97.1% drop!)
```

At $128\text{K}$ context, an entire $USD 300,000$ 8-GPU cluster can only serve **11 users simultaneously** before crashing with CUDA Out-Of-Memory!

---

## 6. The Attention Sink Phenomenon (StreamingLLM)

In late 2023, Xiao et al. (MIT, Meta) published **StreamingLLM**, uncovering a shocking empirical property of autoregressive transformers: **Attention Sinks**.

### The Naive Eviction Experiment:
To prevent memory exhaustion during long text streaming, engineers attempted a simple FIFO sliding-window cache:
* Retain only the most recent $W = 1,024$ tokens in the KV cache.
* When token $t = 1,025$ arrives, drop token $0$.

**The Result was Catastrophic:**
The moment token $0$ was dropped, the model's perplexity exploded from $3.2$ to over **$10,000$**. The model began outputting completely incoherent gibberish!

```
Perplexity Curve Under Naive Window Eviction:
Perplexity
   │
10⁴│                               ┌────────────────────── (Complete Model Collapse!)
   │                               │
10²│                               │
   │                               │
 10│═══════════════════════════════┘
  0└───────────────────────────────┬──────────────────────── Sequence Length
   0                              1024 (Token 0 dropped)
```

### Why Did the Model Collapse?
The researchers analyzed the attention weight distributions across all layers and heads:
1. In causal self-attention, the softmax operator enforces that all attention weights sum to $1.0$:
   $$\sum_{j=0}^{t} \text{Softmax}(Q_t K_j^T) = 1.0$$
2. Because the first token (position $0$, typically `<s>` or `<|begin_of_text|>`) is visible to *every single subsequent token* throughout the entire pre-training process, the transformer learns to dump **massive excess attention probability mass into the first token**.
3. Even when the initial token carries zero semantic relevance to the current sentence, it acts as a numerical **Attention Sink** (a denominator stabilizer for the softmax equation).
4. Dropping the first token removes the mathematical sink, causing softmax scores across the remaining tokens to fluctuate wildly and destroying generation quality.

```
Attention Distribution Across Tokens:
Position:      0 (Sink)     1     2     ...   1020   1021   1022   1023 (Current)
Attention %:  [  52.4%  ] [ 1% ][0.5%]       [ 8% ] [ 12% ][ 11% ][ 15% ]
                  ▲
           Absorbs majority of attention mass!
```

---

## 7. StreamingLLM: Sinks + Rolling Window

The solution proposed by StreamingLLM is elegantly simple:
> **Retain the first 4 tokens (Attention Sinks) indefinitely, and maintain a rolling sliding window of the most recent $W$ tokens.**

```
StreamingLLM KV Cache Layout:
┌───────────────────────────────┬────────────────────────────────────────────────────────┐
│ Initial Tokens (Sinks: 0 - 3) │ Rolling Local Window (Recent W tokens: t-W+1 ... t)    │
│ Pinned permanently in memory! │ Dynamic FIFO Eviction                                  │
└───────────────────────────────┴────────────────────────────────────────────────────────┘
```

### The Invariant:
$$\text{Cached Tokens} = [0, 1, 2, 3] \cup [t - W + 1, \dots, t]$$

Total memory consumption is strictly constant:
$$\text{Memory} = (4 + W) \times \text{bytes\_per\_token} = \mathbf{O(1) \text{ Constant Space!}}$$

### Results:
* With just **4 sink tokens** pinned in VRAM, Llama models can stream text for **over 4 million tokens** continuously without a single point of perplexity divergence!
* Memory footprint remains pegged at a fixed budget (e.g. $1\text{ GB}$), eliminating the risk of Out-Of-Memory crashes.

---

## 8. Heavy Hitter Oracle (H2O) & Dynamic Eviction

While StreamingLLM excels for infinite text generation, what about tasks that require reasoning over historical facts embedded in the middle of a $100\text{K}$ document?

A pure sliding window forgets facts mentioned in the middle of the text. To solve this, researchers developed **H2O (Heavy Hitter Oracle)** (Zhang et al., NeurIPS '23).

### Heavy Hitters (H2)
Attention matrices in LLMs exhibit extreme sparsity:
* A small subset of historical tokens (dubbed **Heavy Hitters**) receive recurring high attention scores across many generation steps (e.g., entity names, numerical constants, function signatures).
* $95\%$ of tokens receive near-zero attention after their immediate local window passes.

### The H2O Eviction Policy:
1. Every token in the KV cache tracks its **cumulative attention score**:
   $$A_j = \sum_{\tau} \text{Attn}(\text{Query}_\tau \to \text{Key}_j)$$
2. When the KV cache reaches its capacity limit $M$:
   * Protect the **Attention Sinks** ($k = 4$ initial tokens).
   * Protect the **Recent Window** ($W$ local tokens).
   * For all historical tokens in the middle, retain only the top $H$ tokens with the highest cumulative attention scores ($A_j$).
   * Evict the remaining low-attention tokens from physical memory.

```
H2O Dynamic Memory Topology:
[ Sink Tokens (4) ] ── [ Sparse Heavy Hitters (Top H) ] ── [ Dense Local Window (W) ]
  (Always pinned)       (Entities, facts, numbers)          (Syntactic coherence)
```

H2O reduces KV cache memory consumption by **up to $80\%$** while preserving $>95\%$ of the model's retrieval and needle-in-a-haystack accuracy!

---

## 9. Key Takeaways

1. **Autoregressive decode wastes $99\%$ of GPU compute**: Memory bandwidth limits single-stream generation to $\approx 24\text{ tok/s}$ on 70B models.
2. **Speculative decoding breaks the serial barrier**: A tiny draft model proposes $K$ tokens, and the giant target model verifies all $K$ tokens in a single parallel GEMM step.
3. **Lossless rejection sampling**: Speculative decoding is mathematically guaranteed to preserve the exact probability distribution of the target model.
4. **Speedups scale with acceptance rate $\alpha$**: At $\alpha = 0.75$ and lookahead $K = 5$, speculative decoding delivers a $2.5\times$ wall-clock speedup.
5. **Long context breaks VRAM**: A single $128\text{K}$ request on Llama-3-70B consumes $42\text{ GB}$ of KV cache, collapsing cluster concurrency by $97\%$.
6. **Attention sinks prevent model collapse**: Retaining the first 4 tokens preserves the softmax denominator, enabling stable generation with $O(1)$ constant memory via StreamingLLM.
7. **H2O balances memory and retrieval**: Retaining attention sinks, heavy hitters, and a local window cuts KV memory by $80\%$ while maintaining reasoning fidelity.
