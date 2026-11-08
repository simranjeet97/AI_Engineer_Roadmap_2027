# 04. Advanced Batching: Continuous Batching & Chunked Prefill

## Iteration-Level Continuous Batching (Orca)
Static batching ties up execution slots until the longest request completes, creating $30\% - 50\%$ idle GPU compute bubbles.

### Continuous Batching Invariant:
Execution operates at the granularity of single token iterations. At forward step $k$:
* Requests finishing at step $k-1$ release their batch slots immediately.
* Waiting requests in the scheduler queue are inserted into the vacated slots at step $k$.
* GPU slot utilization is maintained at $\approx 100\%$.

## Chunked Prefill (Sarathi-Serve)
A monolithic 8k prompt prefill takes $\approx 220\text{ms}$ on an H100, freezing all active decode streams and blowing past ITL SLOs (**Decode Starvation**).

### Chunked Prefill Mechanics:
Slice long prompts into chunks of size $C = 512$:
$$\text{Iteration Time} = t_{\text{decode}}(B) + t_{\text{prefill}}(C)$$
* Step 1: Process Chunk 1 ($512$ tokens) $+$ 32 decode streams (takes $\approx 35\text{ms}$).
* Step 2: Process Chunk 2 ($512$ tokens) $+$ 32 decode streams (takes $\approx 35\text{ms}$).
* ...
* Step 16: Final Chunk completes prefill.
* **Result**: Decode streams advance continuously at every step with zero freezes!
