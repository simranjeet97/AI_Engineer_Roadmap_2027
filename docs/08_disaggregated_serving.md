# 08. Disaggregated Prefill & Decode Serving Architectures

## Why Colocation Fails
Prefill requires raw compute (TFLOPS), decode requires memory bandwidth (TB/s). Colocating them causes mutual destruction: prefill bursts freeze decode streams, while decode KV caches starve prefill batch memory.

## Disaggregated Network Wire Transfer Latency
$$\text{Transfer Time} = \frac{\text{KV Cache Size (Bytes)}}{\text{Network Bandwidth (Bytes/s)}}$$
For Llama-3-70B at 4,096 tokens ($1.28\text{ GB}$ KV cache):
* Over 10 GbE: $1,024\text{ ms}$ (unacceptable).
* Over 400 Gbps RoCEv2 ($50\text{ GB/s}$): **$25.6\text{ ms}$** (matches 1 decode step!).
* Over 800 Gbps InfiniBand ($100\text{ GB/s}$): **$12.8\text{ ms}$**.

## Mooncake Architecture (Kimi AI / Moonshot)
Decouples prefill and decode using **Concorde**: a distributed memory pool combining:
* Tier 1: Local GPU HBM ($1.5\mu\text{s}$)
* Tier 2: Host CPU RAM pooled across cluster over RoCEv2 RDMA ($25\mu\text{s}$)
* Tier 3: Local NVMe Storage spillover ($150\mu\text{s}$)

## Sizing Ratio
Because decode takes $100\times - 250\times$ longer than prefill ($5\text{s}$ vs $20\text{ms}$), **1 high-TFLOPS Prefill GPU (L40S) easily feeds 8 to 16 high-HBM Decode GPUs (H100/H200)**.
