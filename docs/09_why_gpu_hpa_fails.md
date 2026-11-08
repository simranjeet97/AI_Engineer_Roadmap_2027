# 09. Kubernetes Orchestration: Why Standard GPU% Autoscaling Fails

> *"In cloud infrastructure, scaling on the wrong metric is worse than not scaling at all. Scaling on GPU utilization in an LLM cluster will either bankrupt your company or starve your users."*

In traditional cloud-native software engineering, autoscaling stateless microservices (such as REST APIs or web gateways) is straightforward:
* You configure a **Horizontal Pod Autoscaler (HPA)** targeting `CPUUtilizationPercentage: 70%` or `MemoryUtilizationPercentage: 80%`.
* When HTTP traffic surges, CPU usage rises linearly. The HPA detects this and spins up additional replica pods.
* When traffic subsides, CPU usage falls, and the HPA scales the deployment back down to minimum replicas.

When infrastructure teams attempt to manage Large Language Model inference engines (such as vLLM, SGLang, or TensorRT-LLM) using this same classical mental model by attaching an HPA to NVIDIA Data Center GPU Manager (DCGM) metrics like `DCGM_FI_DEV_GPU_UTIL` (GPU SM %) or `DCGM_FI_DEV_FB_USED` (GPU Memory Used), **the system fails completely**.

The autoscaler either:
1. Scales up to maximum replicas immediately at boot time and **refuses to ever scale down**, burning tens of thousands of cloud dollars per day on an empty cluster.
2. OR, sits completely inert during massive traffic surges while user requests pile up in endless waiting queues, causing catastrophic latency meltdowns.

This technical memo derives why standard GPU-level metrics fail for LLM autoscaling from absolute first principles, and provides the production-proven architecture: **Queue-Depth and KV Cache Saturation autoscaling via KEDA and Prometheus**.

---

## 1. The Two Fatal Failure Modes of GPU-Level HPA

Let us dissect the physical mechanics of why standard GPU metrics report deceptive signals during LLM serving:

```
Failure Mode 1: The VRAM Pre-Allocation Trap
┌────────────────────────────────────────────────────────────────────────┐
│ Total GPU VRAM: 80 GB (NVIDIA A100 / H100)                             │
├───────────────────────┬────────────────────────────────────────────────┤
│ Model Weights (14 GB) │ Pre-allocated PagedAttention KV Pool (62 GB)   │
└───────────────────────┴────────────────────────────────────────────────┘
 ▲                                                                        ▲
 └──────────────────────────── 76 GB (95%) ───────────────────────────────┘
  Standard HPA sees: "Memory Usage = 95%! PANIC! SCALE UP REPLICAS!"
  Reality: ZERO user requests are currently being processed!
```

### Failure Mode 1: The VRAM Pre-Allocation Illusion
Recall from Chapter 02 the core architecture of PagedAttention:
* At container startup, the inference engine allocates the model weights (e.g. $14\text{ GB}$ for Llama-3-8B).
* Then, to avoid costly CUDA runtime allocations (`cudaMalloc`) during execution, the engine invokes `cudaMalloc` once to claim **almost all remaining GPU VRAM** (typically `--gpu-memory-utilization 0.90` or `0.95`).
* It formats this VRAM into a static pool of physical memory blocks.

**What the Infrastructure Sees:**
* The NVIDIA DCGM driver queries the GPU framebuffer: `DCGM_FI_DEV_FB_USED = 76,000 MB / 80,000 MB` ($\mathbf{95\% \text{ VRAM utilized}}$).
* The classical Kubernetes HPA evaluates the metric:
  $$\text{Memory \%} = 95\% \gg 80\% \text{ Target Threshold}$$
* The HPA immediately triggers scale-up events, spinning up maximum GPU pods ($2 \to 4 \to 8 \to 16$ nodes).
* Even if **zero requests arrive all night**, the GPU memory usage never drops below $95\%$.
* **The cluster stays permanently pinned at maximum pod capacity**, generating enormous cloud infrastructure bills for completely idle GPUs!

---

### Failure Mode 2: The GPU Compute Busy-Polling Illusion
What happens if you scale on GPU Streaming Multiprocessor (SM) Compute Utilization (`DCGM_FI_DEV_GPU_UTIL`)?

```
Failure Mode 2: The Busy-Polling Illusion
Engine Iteration:
┌───────────────────────────────┐
│ Forward Pass (5ms)            │ -> SMs Active (100%)
├───────────────────────────────┤
│ Scheduler Poll Loop (10ms)    │ -> Tight CUDA event polling / CPU-GPU sync (Reports High SM!)
└───────────────────────────────┘
DCGM Sample Window (1s): Reports 85% SM Utilization even under 10% actual capacity!
```

Modern high-performance inference engines use asynchronous execution pipelines, CUDA streams, and continuous batching schedulers.
1. When handling a tiny trickle of traffic (e.g., 2 concurrent requests on an 8-GPU cluster capable of 250 requests), the GPU rapidly finishes each token decode step in $\approx 15\text{ ms}$.
2. In the remaining idle time, the engine's background threads spin in tight polling loops, awaiting new tokens or checking CUDA completion events.
3. Because the DCGM metrics engine samples SM activity at coarse intervals ($1\text{ second}$ windows), any transient CUDA kernel activity causes the counter to report high SM activity ($60\% - 90\%$).
4. The metric fails to distinguish between:
   * A GPU processing **$2$ tokens per step** (nearly empty).
   * A GPU processing **$256$ tokens per step** (fully saturated).

Both report high SM activity, rendering GPU SM% completely useless as an elasticity metric!

---

## 2. The True Operational Signals of LLM Saturation

To scale an LLM cluster accurately, we must monitor the **application-level state** of the inference engine, not the underlying silicon counters.

There are exactly two metrics that govern the saturation state of an inference engine:

```
The Two Pillars of LLM Autoscaling:
┌────────────────────────────────────────┐      ┌────────────────────────────────────────┐
│       1. Waiting Queue Depth           │      │    2. GPU KV Cache Usage Factor        │
│    vllm:num_requests_waiting           │      │     vllm:gpu_cache_usage_factor        │
├────────────────────────────────────────┤      ├────────────────────────────────────────┤
│ Measures: Traffic overload & Backpressure│      │ Measures: Memory pool saturation       │
│ Meaning: Requests arriving faster than │      │ Meaning: Fraction of physical blocks   │
│          active batch capacity.        │      │          actively holding user state.  │
│ Threshold: > 5 - 10 requests waiting   │      │ Threshold: > 80% - 85% cache utilized  │
└────────────────────────────────────────┘      └────────────────────────────────────────┘
```

### Signal 1: Waiting Queue Depth (`vllm:num_requests_waiting`)
In continuous batching, when the engine's active running batch (`max_num_seqs`) is full, or when VRAM cannot allocate a new prompt's initial blocks, incoming requests are placed into the **Waiting Queue**.
* Under healthy conditions: `vllm:num_requests_waiting == 0`. Every incoming request is admitted into the next forward-pass iteration within milliseconds.
* Under saturation: `vllm:num_requests_waiting > 0`. Requests are accumulating in queue memory. By Little's Law ($L = \lambda W$), wait time (TTFT) grows linearly with queue depth!

### Signal 2: GPU KV Cache Usage Factor (`vllm:gpu_cache_usage_factor`)
This metric exposes the exact fraction of PagedAttention physical blocks currently allocated to running or cached requests:
$$\text{Cache Usage Factor} = \frac{\text{Allocated Physical Blocks}}{\text{Total Physical Blocks Pool}}$$
* Under empty load: `vllm:gpu_cache_usage_factor == 0.0` (0%).
* Under healthy load: `0.20 \le \text{Usage Factor} \le 0.75`.
* Under dangerous saturation: `\text{Usage Factor} > 0.85` (85%).
* When usage hits $100\%$, the engine can no longer allocate blocks for ongoing decodes. It is forced to **preempt** active requests, swapping them to CPU RAM or discarding their KV cache and forcing expensive recomputations!

---

## 3. Production Architecture: Kubernetes + KEDA + Prometheus

To implement queue-aware autoscaling, Kubernetes requires **KEDA (Kubernetes Event-driven Autoscaling)**.

```
KEDA Autoscaling Control Plane:
[ vLLM Engine Pods ] ── (Prometheus Exporter: /metrics) 
                                │
                                │ (Scraped every 1s)
                                ▼
                    [ Prometheus Time Series DB ]
                                │
                                │ (Polled every 5s)
                                ▼
                       [ KEDA Operator ]
                                │
                                │ (Evaluates Thresholds)
                                ▼
               [ Kubernetes Horizontal Pod Autoscaler ]
                                │
                                │ (Adjusts replica count)
                                ▼
                    [ vLLM Pod Deployment ]
                    (Scales between 2 and 16 pods)
```

### How the Pipeline Operates:
1. Every inference pod exposes its internal application gauges on `/metrics` (scraped by Prometheus at high-frequency $1\text{s}$ intervals).
2. The KEDA operator runs a periodic query against Prometheus:
   * Query 1: `sum(vllm:num_requests_waiting)`
   * Query 2: `avg(vllm:gpu_cache_usage_factor)`
3. If the waiting queue exceeds $10$ requests OR the average cache utilization exceeds $85\%$, KEDA calculates the required replica count and immediately scales up the Kubernetes Deployment.
4. When queue depth returns to $0$ and cache utilization drops below the threshold, KEDA initiates a controlled, stabilized scale-down.

---

## 4. Defending Against the Scale-Down Stampede (Cooldown Mechanics)

Autoscaling GPU workloads requires specialized care during **Scale-Down**.

### The Cold-Start Penalty
Spinning up an 8x H100 inference pod is extraordinarily slow compared to an Nginx container:
1. Kubernetes schedules the pod onto an available GPU worker node.
2. Container runtime pulls the $30\text{ GB}$ Docker image ($\approx 45\text{ seconds}$).
3. Container initializes CUDA runtime and PyTorch ($\approx 15\text{ seconds}$).
4. Model weights are loaded from high-speed NVMe / network storage into GPU VRAM:
   * Llama-3-70B FP8 ($70\text{ GB}$) takes $\approx 25 - 60\text{ seconds}$ to load and verify weights.
5. CUDA kernel compilation and warmup forward passes take $\approx 20\text{ seconds}$.

**Total Cold-Start Time: $2$ to $3$ minutes minimum.**

### The Trashing / Flapping Danger
If your autoscaler terminates a GPU pod the moment the waiting queue hits zero:
* At 12:00 PM: Queue drops to 0 $\to$ Pod is immediately terminated.
* At 12:01 PM: A new burst of requests arrives $\to$ Autoscaler triggers scale-up $\to$ Cluster experiences a 3-minute cold start while users wait in agony!
* This violent cycle of scaling up and down is called **Flapping (or Trashing)**.

### The Solution: Aggressive Scale-Up, Conservative Scale-Down
We configure asymmetric scaling policies inside the KEDA `ScaledObject`:

```yaml
scaleDown:
  stabilizationWindowSeconds: 300   # Must remain quiet for 5 FULL MINUTES before terminating a pod!
  policies:
  - type: Percent
    value: 20                       # Terminate at most 20% of pods at a time
    periodSeconds: 60
scaleUp:
  stabilizationWindowSeconds: 0     # ZERO delay on scale-up!
  policies:
  - type: Percent
    value: 100                      # Double the cluster capacity instantly when saturated!
    periodSeconds: 15
```

---

## 5. Complete Production KEDA ScaledObject Manifest

Here is the exact, staff-grade manifest deployed in production clusters:

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: apex-infer-keda-autoscaler
  namespace: inference
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: apex-infer-vllm
  minReplicaCount: 2                  # Always maintain warm baseline capacity
  maxReplicaCount: 16                 # Bounded to protect cloud budgets
  cooldownPeriod: 300                 # 5 minutes stabilization before scale-down
  pollingInterval: 5                  # Check metrics every 5 seconds
  advanced:
    horizontalPodAutoscalerConfig:
      behavior:
        scaleDown:
          stabilizationWindowSeconds: 300
          policies:
          - type: Percent
            value: 20
            periodSeconds: 60
        scaleUp:
          stabilizationWindowSeconds: 0
          policies:
          - type: Percent
            value: 100
            periodSeconds: 15
  triggers:
  # Trigger 1: Scale on Waiting Queue Depth
  - type: prometheus
    metadata:
      serverAddress: http://apex-prometheus.inference.svc.cluster.local:9090
      metricName: vllm_num_requests_waiting
      query: sum(vllm:num_requests_waiting)
      threshold: '10'                 # Scale up if >10 total requests waiting across cluster
  # Trigger 2: Scale on KV Cache Saturation
  - type: prometheus
    metadata:
      serverAddress: http://apex-prometheus.inference.svc.cluster.local:9090
      metricName: vllm_gpu_cache_usage_factor
      query: avg(vllm:gpu_cache_usage_factor)
      threshold: '0.85'               # Scale up if average KV cache exceeds 85%
```

---

## 6. Key Takeaways

1. **Standard GPU metrics deceive**: GPU VRAM is pre-allocated at boot ($95\%$ full on idle); GPU SM utilization spikes during tight CUDA event polling.
2. **Never scale on DCGM VRAM or SM%**: Doing so leads to permanent maximum replica lockup or zero-scaling during actual request backlogs.
3. **Queue depth is the true measure of backpressure**: When `vllm:num_requests_waiting > 0`, incoming requests are accumulating latency and demand immediate horizontal scaling.
4. **KV cache usage reveals internal saturation**: When `vllm:gpu_cache_usage_factor > 0.85`, running sequences risk preemption and OOM eviction.
5. **Cold starts mandate asymmetric scaling**: Scaling up must be instantaneous (0s stabilization); scaling down must be heavily damped ($300\text{s}$ stabilization) to prevent flapping and cold-start latency spikes.
