# 10. Foundational Inference Systems Research Taxonomy

| Paper | Conference | Breakthrough Contribution |
| :--- | :--- | :--- |
| **Orca** | *OSDI '22* | Pioneered **iteration-level continuous batching**, eliminating GPU bubbles. |
| **vLLM** | *SOSP '23* | Introduced **PagedAttention**, partitioning KV cache into non-contiguous physical blocks. |
| **FlashAttention 1-3** | *NeurIPS '22 / ICLR '24* | IO-aware tiling fusing attention computation in fast SRAM; eliminated $O(N^2)$ memory traffic. |
| **Sarathi-Serve** | *OSDI '24* | Introduced **chunked prefill**, interleaving prompt chunks with decode iterations to preserve ITL. |
| **Mooncake** | *ArXiv '24* | Disaggregated KV-centric serving engine using distributed memory pools over RDMA (Kimi architecture). |
| **Splitwise** | *ISCA '24* | Formalized hardware specialization for prefill (compute) vs decode (bandwidth). |
| **DistServe** | *OSDI '24* | Disaggregated prefill and decode to decouple TTFT and ITL SLOs. |
| **AWQ** | *MLSys '24* | Activation-aware weight quantization protecting salient channels for lossless INT4 inference. |
| **Marlin** | *IST Austria '24* | Fused asynchronous double-buffered INT4 GEMM kernels achieving near-ideal compute speedups. |
| **StreamingLLM** | *ICLR '24* | Discovered **Attention Sinks**, enabling infinite context streaming with bounded $O(1)$ memory. |
| **FastServe** | *OSDI '23* | Preemptive distributed scheduling using Multi-Level Feedback Queues to minimize job completion time. |
| **SGLang** | *ArXiv '24* | Dynamic **RadixAttention**, treating GPU physical memory as an LRU prefix cache tree. |
