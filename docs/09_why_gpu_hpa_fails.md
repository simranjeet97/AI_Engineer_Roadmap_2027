# 09. Kubernetes Orchestration: Why GPU% HPA Fails

## The Failure of Standard GPU% HPA for LLMs
1. **VRAM Pre-allocation**: vLLM pre-allocates $\approx 90\%$ of GPU VRAM at boot for PagedAttention. Traditional HPA reads $90\%$ memory usage and keeps maximum pods running 24/7.
2. **Busy Polling**: Continuous batching engines spin in tight CUDA polling loops, reporting high SM utilization even when zero requests are active.

## The Staff Solution: KEDA on Custom Prometheus Metrics
Autoscale strictly on:
* `vllm:num_requests_waiting` (Scale up when queue depth $> 8$ for $> 30\text{s}$).
* `vllm:gpu_cache_usage_factor` (Scale up when KV cache utilization $> 0.85$).
* Enforce a **300-second stabilization cooldown window** on scale-down to prevent pod flapping and preserve warm prefix caches.

## The `/dev/shm` Shared Memory Trap
Default Kubernetes containers provide only $64\text{ MB}$ of `/dev/shm`. PyTorch IPC crashes with `Bus error` during tensor-parallel communication.
* **Fix**: Mount an `emptyDir` volume with `medium: Memory` and size $\ge 16\text{GiB}$ to `/dev/shm`.
