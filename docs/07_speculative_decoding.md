# 07. Speculative Decoding & Long-Context KV Cache Eviction

## Long-Context KV Cache Bottleneck
At 128k context, KV cache memory scales linearly ($O(B \cdot L)$), consuming $\approx 10.5\text{ GB}$ per request on Llama-3-70B. Concurrency collapses by $98.5\%$.

## StreamingLLM Attention Sinks (ICLR '24)
* Softmax attention denominators dump massive surplus probability mass onto tokens $0 \dots 3$ (**Attention Sinks**).
* Retaining **4 sink tokens $+$ a rolling local window (e.g. 1020 tokens)** keeps KV cache memory strictly bounded ($O(1)$) with zero perplexity explosion.

## Speculative Decoding Mechanics (Leviathan et al.)
1. Fast Draft Model proposes $K$ tokens.
2. Slow Target Model verifies all $K$ tokens in a single parallel GEMM pass.
3. Modified Rejection Sampling accepts tokens:
   $$P_{\text{accept}}(x) = \min\left(1, \; \frac{P_{\text{target}}(x)}{P_{\text{draft}}(x)}ight)$$
4. If rejected, samples replacement token from residual distribution. Lossless equivalence guaranteed.

## The Breaking Point of Speculative Decoding:
$$\text{Speedup} = \frac{\mathbb{E}[N]}{1 + \gamma \cdot K} \quad \text{where } \mathbb{E}[N] = \frac{1 - \alpha^{K+1}}{1 - \alpha}$$
* At $B > 32$, target model transitions from memory-bound to compute-bound; running the draft model steals compute cycles.
* At $\alpha < 0.60$, draft rejections waste compute. Speedup drops below $1.0\times$ (slowdown).
