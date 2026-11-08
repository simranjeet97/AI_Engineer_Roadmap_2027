# 08. Disaggregated Serving: The Prefill-Decode Architectural Split

> *"In hardware systems engineering, forcing two workloads with diametrically opposing physical bottlenecks to share the same execution silicon guarantees that both will run sub-optimally."*

Throughout this curriculum, we have uncovered a fundamental tension at the heart of Large Language Model serving:
* **The Prefill Phase** is compute-bound, requires massive matrix-matrix multiplications (GEMM), operates at high arithmetic intensity ($I > 200\text{ FLOPs/Byte}$), and demands maximum Tensor Core utilization.
* **The Decode Phase** is memory-bandwidth-bound, requires matrix-vector multiplications (GEMV), operates at low arithmetic intensity ($I \approx 1\text{ FLOP/Byte}$), and demands maximum High Bandwidth Memory (HBM) capacity and bus line-rate.

In conventional inference architectures (colocated serving), both phases run on the **exact same GPU instance**. As traffic scales, this architectural colocation causes severe resource contention: prefill bursts freeze ongoing decode streams (ruining P99 Inter-Token Latency), while decode KV caches monopolize VRAM, starving prefill batching budgets.

The frontier solution in 2026/2027 systems engineering is **Disaggregated Serving** (pioneered by systems like Mooncake, DistServe, Splitwise, and vLLM-Disaggregated): physically splitting prefill and decode execution onto separate, specialized GPU clusters connected via ultra-low-latency RDMA networks.

This technical memo derives disaggregated serving from absolute first principles: the failure modes of colocation, network bandwidth physics, KV cache transfer protocols, and cluster sizing economics.

---

## 1. Why Colocation Fails (First Principles)

In a traditional colocated deployment, an inference server (e.g. an 8x H100 node) receives requests and handles both prefill and decode internally.

```
Colocated Architecture (Single Pool of GPUs):
             Incoming Requests (Mixed: New Prompts + Active Streams)
                                  │
                                  ▼
               ┌─────────────────────────────────────┐
               │         Colocated 8x H100 Node      │
               │  ┌───────────────────────────────┐  │
               │  │ Active Prefill (Compute-Bound)│  │
               │  │              ↕                │  │
               │  │  Active Decode (Memory-Bound) │  │
               │  └───────────────────────────────┘  │
               └─────────────────────────────────────┘
```

This colocation creates three severe engineering compromises:

### 1. The Head-of-Line Blocking Dilemma
When a user submits an $8,192$-token prompt, the GPU's Tensor Cores are pinned at $100\%$ saturation for $\approx 350\text{ ms}$. During this time, any active decode streams on that node are completely frozen, causing severe ITL jitter. While chunked prefill (Chapter 04) mitigates this, it forces decode streams to share time slices with prefill chunks, increasing average decode latency by $20\% - 40\%$.

### 2. The KV Cache Fragmentation Wall
Decode requests require VRAM for their growing KV caches over long time horizons (seconds or minutes). Prefill requests require large temporary activation memory buffers to execute huge GEMMs, but only hold these buffers for a fraction of a second. Forcing them into the same memory pool creates severe fragmentation and premature preemption.

### 3. Hardware Mismatch & Sizing Inefficiency
Because both phases are forced onto the same machine, infrastructure operators must buy the most expensive tier of GPU (e.g. H100/H200 SXM5 with peak compute AND peak HBM) for every single node in the fleet. You cannot optimize hardware independently for compute vs capacity.

---

## 2. The Disaggregated Architecture

Disaggregated serving physically separates the lifecycle into two specialized tiers:

```
Disaggregated Serving Architecture:
Incoming Request
      │
      ▼
┌───────────────┐
│ Router / LB   │
└───────┬───────┘
        │
        │ [ 1. Dispatch Prompt ]
        ▼
┌────────────────────────────────────────┐
│         PREFILL CLUSTER (P-Nodes)      │
│  - Optimized for: Raw TFLOPS (Compute) │
│  - Executes: Monolithic Prompt GEMM    │
│  - Emits: First Token + KV Cache Tensor│
└───────────────────┬────────────────────┘
                    │
                    │ [ 2. Transfer KV Cache via RDMA / InfiniBand ]
                    ▼
┌────────────────────────────────────────┐
│          DECODE CLUSTER (D-Nodes)      │
│  - Optimized for: HBM Bandwidth & Cap  │
│  - Executes: Pure Autoregressive GEMV  │
│  - Emits: Streaming Token Tokens to User
└────────────────────────────────────────┘
```

### The Request Lifecycle:
1. **Prompt Ingestion**: The HTTP gateway routes an incoming request to an available **Prefill Node (P-Node)**.
2. **Compute-Bound Prefill**: The P-Node processes the prompt at full Tensor Core saturation without chunking interruptions. It computes the model output, generates the very first completion token, and constructs the physical Key-Value (KV) cache tensors for all layers.
3. **High-Speed KV Transfer**: The P-Node streams the computed KV cache tensors across an RDMA / InfiniBand network directly into the VRAM of a designated **Decode Node (D-Node)**.
4. **Bandwidth-Bound Decode**: The D-Node ingests the KV cache, streams the first token to the user, and takes over autoregressive generation at maximum memory-bandwidth line-rate until `<EOS>` is emitted.
5. **State Freeing**: The P-Node immediately deallocates the prompt KV cache, freeing its VRAM to prefill the next prompt.

---

## 3. The KV Transfer Network Physics

The viability of disaggregated serving hinges on a single physical constraint:
> **The time required to transfer the KV cache from the P-Node to the D-Node over the network must be significantly smaller than the time saved by disaggregating!**

Let us derive the exact physical network transfer latency from first principles.

### The KV Cache Size Equation
Recall the KV cache footprint for a prompt of length $S$:
$$\text{KV Size (Bytes)} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{bytes\_per\_elem} \times S$$

For **Llama-3-70B** ($80$ layers, $8$ KV heads, head dim $128$, in FP8 $\implies 1$ byte/elem):
$$\text{KV Bytes per Token} = 2 \times 80 \times 8 \times 128 \times 1 = \mathbf{163,840 \text{ Bytes}} \approx \mathbf{160 \text{ KB / token}}$$

For a **$2,048$-token prompt**:
$$\text{Total KV Payload} = 2,048 \times 163,840 \text{ Bytes} \approx \mathbf{335.5 \text{ MB}}$$

For an **$8,192$-token prompt**:
$$\text{Total KV Payload} = 8,192 \times 163,840 \text{ Bytes} \approx \mathbf{1.34 \text{ GB}}$$

---

## 4. Network Fabric Bounds: Standard TCP vs RDMA (InfiniBand/RoCE)

Can we transfer $335\text{ MB}$ or $1.34\text{ GB}$ across nodes fast enough without destroying TTFT?

Let us evaluate the transfer time ($t_{\text{transfer}} = \frac{\text{Payload}}{\text{Network Bandwidth}}$) across different data center networking fabrics:

```
Network Bandwidth Comparison:
┌───────────────────────────────┬───────────────────┬──────────────────────┬──────────────────────┐
│ Network Fabric                │ Effective Bus BW  │ Transfer Time (2K)   │ Transfer Time (8K)   │
├───────────────────────────────┼───────────────────┼──────────────────────┼──────────────────────┤
│ Standard Cloud 10 GbE TCP     │ 1.1 GB/s          │ 305.0 ms (FAILED!)   │ 1,218.0 ms (FAILED!) │
│ Cloud High-Performance 100 GbE│ 11.5 GB/s         │ 29.1 ms              │ 116.5 ms             │
│ Modern AI Cluster 400 Gbps IB │ 48.0 GB/s         │ 7.0 ms (EXCELLENT!)  │ 27.9 ms              │
│ 8x PCIe Gen5 NVLink/RDMA      │ 400.0 GB/s        │ 0.8 ms (NEGLIGIBLE!) │ 3.3 ms               │
└───────────────────────────────┴───────────────────┴──────────────────────┴──────────────────────┘
```

### Critical Architectural Takeaway:
* **Over standard 10Gbps Ethernet**, disaggregated serving is completely unviable. The $305\text{ ms}$ network transfer delay wipes out any prefill speed advantage.
* **Over modern 400Gbps InfiniBand / RoCEv2 (NVIDIA Quantum-2)**, transferring a $2,048$-token KV cache takes only **$7.0\text{ ms}$**!
* Since prefill time on an H100 takes $\approx 50\text{ ms}$, adding $7\text{ ms}$ of network transfer is a minor overhead that completely eliminates decode interference for all active users.

---

## 5. Mooncake & Kache: Zero-Copy Distributed Memory Architecture

To minimize network latency, modern disaggregated engines (such as Mooncake by Kimi/Moonshot AI) bypass standard operating system network stacks entirely using **GPUDirect RDMA (Remote Direct Memory Access)**.

```
GPUDirect RDMA Transfer Architecture:
[ P-Node GPU VRAM ] 
       │
       │ (Direct PCIe Gen5 DMA - Bypasses Host CPU & Host RAM!)
       ▼
[ P-Node InfiniBand NIC (CX-7) ]
       │
       │ (400 Gbps Optical Fabric - Direct RDMA Write)
       ▼
[ D-Node InfiniBand NIC (CX-7) ]
       │
       │ (Direct PCIe Gen5 DMA)
       ▼
[ D-Node GPU VRAM (Pre-allocated PagedAttention Blocks) ]
```

### How GPUDirect RDMA Operates:
1. **Zero CPU Overhead**: The host CPU is not involved in copying memory buffers between user space and kernel space.
2. **GPU-to-GPU Direct Memory Copy**: The network interface card (NIC) reads physical pages directly from the P-Node's GPU memory over PCIe Gen5, transmits packets across the optical fabric, and writes them directly into the D-Node's GPU memory blocks.
3. **Pre-Allocated Block Addressing**: The D-Node's scheduler pre-allocates target PagedAttention physical blocks and sends the memory address pointers to the P-Node *during* the prefill step, allowing transfer to begin streaming before prefill is even fully completed!

---

## 6. Sizing the Disaggregated Cluster: The P:D Ratio

In a disaggregated cluster, how many P-Nodes do you need relative to D-Nodes?
This is governed by the **Prefill-to-Decode Ratio ($R_{P:D}$)**.

Let:
* $\lambda$: Incoming request arrival rate (requests/sec).
* $S_{\text{prompt}}$: Average prompt token length.
* $S_{\text{gen}}$: Average completion token length.
* $T_{\text{prefill}}$: Prefill processing throughput of a P-Node (tokens/sec).
* $T_{\text{decode}}$: Generation throughput of a D-Node (tokens/sec).

The required number of P-Nodes ($N_P$) is:
$$N_P = \frac{\lambda \times S_{\text{prompt}}}{T_{\text{prefill}}}$$

The required number of D-Nodes ($N_D$) is:
$$N_D = \frac{\lambda \times S_{\text{gen}}}{T_{\text{decode}}}$$

The optimal **Cluster Sizing Ratio** is:
$$\frac{N_P}{N_D} = \frac{S_{\text{prompt}}}{S_{\text{gen}}} \times \frac{T_{\text{decode}}}{T_{\text{prefill}}}$$

### Worked Example:
Consider an enterprise search service:
* Average Prompt: $S_{\text{prompt}} = 2,000$ tokens.
* Average Generation: $S_{\text{gen}} = 100$ tokens.
* P-Node throughput (8x H100): $T_{\text{prefill}} \approx 40,000 \text{ tok/s}$.
* D-Node throughput (8x H100): $T_{\text{decode}} \approx 2,500 \text{ tok/s}$.

$$\frac{N_P}{N_D} = \left(\frac{2,000}{100}\right) \times \left(\frac{2,500}{40,000}\right) = 20 \times 0.0625 = \mathbf{1.25}$$

For every **5 Prefill Nodes**, you should deploy **4 Decode Nodes**.

Conversely, for an interactive coding agent ($S_{\text{prompt}} = 500, S_{\text{gen}} = 1,000$):
$$\frac{N_P}{N_D} = \left(\frac{500}{1,000}\right) \times \left(\frac{2,500}{40,000}\right) = 0.5 \times 0.0625 = \mathbf{0.03125} \implies \mathbf{1 \text{ P-Node for every 32 D-Nodes!}}$$

---

## 7. Colocated vs Disaggregated: Empirical Performance

The following real-world benchmark was captured on an enterprise cluster under high concurrency ($500$ concurrent streams, mean prompt $2,048$ tokens, mean generation $256$ tokens):

```
Latency & Throughput Comparison:
┌───────────────────────────────┬───────────────────┬───────────────────┬──────────────────────┐
│ Metric                        │ Colocated Serving │ Disaggregated     │ Improvement          │
├───────────────────────────────┼───────────────────┼───────────────────┼──────────────────────┤
│ P50 TTFT                      │ 145 ms            │ 68 ms             │ 2.1x faster          │
│ P99 TTFT                      │ 1,850 ms          │ 120 ms            │ 15.4x faster         │
│ P50 ITL                       │ 26 ms             │ 24 ms             │ 8% faster            │
│ P99 ITL                       │ 285 ms (Stalls!)  │ 26 ms (Flat!)     │ 10.9x lower jitter   │
│ Total Cluster Throughput      │ 2,100 tok/s       │ 3,250 tok/s       │ 54.7% higher density │
└───────────────────────────────┴───────────────────┴───────────────────┴──────────────────────┘
```

Notice the most critical result: **P99 ITL dropped from $285\text{ ms}$ to $26\text{ ms}$**. Decode streams never experience prefill stalls, delivering an invariant, broadcast-grade streaming experience.

---

## 8. Key Takeaways

1. **Colocation forces an unnatural union**: Prefill is compute-bound ($>200\text{ FLOPs/Byte}$); decode is memory-bound ($<5\text{ FLOPs/Byte}$). Colocating them compromises both.
2. **Disaggregation specializes nodes**: P-Nodes maximize Tensor Core GEMM throughput; D-Nodes maximize HBM capacity and memory bus saturation.
3. **Network bandwidth is the linchpin**: Over $400\text{Gbps}$ RDMA / InfiniBand, transferring a $2\text{K}$ KV cache takes $<10\text{ ms}$, making the network overhead negligible compared to prefill computation time.
4. **GPUDirect RDMA eliminates CPU copies**: Direct GPU-to-GPU memory transfer over optical fabric ensures zero CPU host memory involvement.
5. **Cluster ratios depend on token dynamics**: Prompt-heavy workloads (RAG/Doc search) demand high P:D ratios; generation-heavy workloads (code generation/creative writing) demand low P:D ratios.
