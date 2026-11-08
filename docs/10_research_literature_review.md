# 10. Foundational Research Literature: The Systems Evolution of LLM Inference

> *"To engineer the frontier of tomorrow, you must master the seminal papers that solved the impossible bottlenecks of yesterday."*

Modern LLM inference engines (vLLM, SGLang, TensorRT-LLM, Mooncake) did not appear overnight. They are the synthesis of groundbreaking systems and machine learning research published between 2020 and 2026. Every major feature in production today — from continuous batching to FlashAttention, PagedAttention, and disaggregated serving — was forged in peer-reviewed literature to solve specific physical bottlenecks in silicon, memory, and networking.

This technical memo serves as the annotated canonical bibliography and architectural taxonomy for the entire 10-week curriculum. It outlines each seminal paper from first principles: the historical problem it confronted, the core algorithmic breakthrough, and its lasting impact on production infrastructure.

---

## 1. Architectural Taxonomy of Inference Innovations

```
The Genealogy of Modern Inference Systems:
                                 [ Transformer (Vaswani et al., 2017) ]
                                                   │
                       ┌───────────────────────────┴───────────────────────────┐
                       ▼                                                       ▼
            [ Algorithmic Attention ]                               [ Scheduling & Memory ]
                       │                                                       │
         ┌─────────────┴─────────────┐                           ┌─────────────┴─────────────┐
         ▼                           ▼                           ▼                           ▼
[ FlashAttention-1/2/3 ]    [ Quantization ]            [ Iteration Batching ]      [ Virtual Memory ]
(Dao et al., 2022-2024)     - AWQ (Lin '23)             - Orca (Yu '22)             - vLLM / PagedAttn
SRAM Tiling & IO Aware      - Marlin (Frantar '24)      Continuous batching         (Kwon et al., '23)
                             Fused INT4 Kernels                                     Zero Fragmentation
                                     │                                       │
                                     └───────────────────┬───────────────────┘
                                                         │
                                                         ▼
                                          [ Advanced Systems Integration ]
                                          - Chunked Prefill (Sarathi, Agrawal '24)
                                          - Radix Tree Caching (SGLang, Zheng '24)
                                          - Speculative Decoding (Leviathan '23)
                                          - Disaggregated Serving (Mooncake, Qin '24)
```

---

## 2. Annotated Bibliography & Deep Technical Memos

### 1. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
* **Authors**: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré (Stanford University)
* **Venue**: NeurIPS 2022 / FlashAttention-2 (ICLR 2024) / FlashAttention-3 (2024)
* **The Problem**: Standard self-attention materializes the full $N \times N$ attention matrix $S = QK^T$ and $P = \text{Softmax}(S)$ in slow GPU HBM. For sequence length $N = 8,192$, the intermediate matrix requires massive memory traffic ($O(N^2)$ reads and writes), making attention strictly memory-bandwidth bound.
* **The Breakthrough**: 
  1. **Tiling**: FlashAttention splits the input $Q, K, V$ matrices into small blocks that fit entirely within high-speed GPU on-chip Shared Memory (SRAM: $19\text{ TB/s}$ bandwidth).
  2. **Online Softmax**: By utilizing the online softmax trick, it computes the scaling and softmax normalization dynamically without ever writing the full $N \times N$ matrix to HBM.
* **Production Impact**: Achieved a $2\times - 4\times$ speedup in attention forward passes and enabled scaling sequence lengths from $2\text{K}$ to $128\text{K}+$. FlashAttention kernels are integrated into every modern inference engine and framework.

---

### 2. Orca: A Distributed Serving System for Transformer-Based Generative Models
* **Authors**: Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, Byung-Gon Chun (Seoul National University)
* **Venue**: OSDI 2022 (USENIX Symposium on Operating Systems Design and Implementation)
* **The Problem**: Traditional serving systems batched requests at the request level (Static Batching). When requests in a batch finished at different times, short requests were held hostage by the longest sequence, wasting $30\% - 60\%$ of GPU compute on empty padding.
* **The Breakthrough**: **Iteration-Level Continuous Batching**. Orca schedules execution at the granularity of a single token forward pass ($\approx 25\text{ ms}$). After every iteration, finished requests immediately exit and new waiting requests are inserted into the running batch.
* **Production Impact**: Tripled GPU serving throughput across enterprise deployments worldwide. Continuous batching is the universal execution foundation of vLLM, SGLang, and TensorRT-LLM.

---

### 3. Efficient Memory Management for Large Language Model Serving with PagedAttention
* **Authors**: Woosuk Kwon, Zhuohan Li, Siyuan Shen, Lingfan Yu, Yanping Huang, Yida Wang, Christian Sarofeen, Parminder Singh, Minjia Zhang, Matthijs Douze, Ion Stoica, Joseph E. Gonzalez (UC Berkeley, Anyscale)
* **Venue**: SOSP 2023 (ACM Symposium on Operating Systems Principles)
* **The Problem**: Before PagedAttention, engines pre-allocated contiguous memory buffers for the entire `max_model_len` (e.g. 4096 tokens) for every request. This caused severe internal fragmentation ($90\%$ space wasted on short generations) and external fragmentation, wasting $60\% - 80\%$ of total GPU VRAM.
* **The Breakthrough**: Inspired by operating system virtual memory paging, **PagedAttention** partitions the KV cache into fixed-size physical blocks ($16$ or $32$ tokens) stored non-contiguously in a global memory pool. A per-request Block Table maps logical sequence tokens to physical block addresses dynamically on demand. It also introduced **Copy-on-Write (CoW)** for parallel sampling.
* **Production Impact**: Created the **vLLM** project. Slashed memory waste from $80\%$ down to $<4\%$, quadrupled concurrent serving capacity on identical hardware, and democratized production LLM serving.

---

### 4. SGLang: Efficient Execution of Structured Language Model Programs
* **Authors**: Lianmin Zheng, Liangsheng Yin, Zhiqiang Shen, Xupeng Miao, Zhanghao Wu, Dachuan Shi, Hanjun Dai, Joseph E. Gonzalez, Ion Stoica (UC Berkeley, LMSYS Org)
* **Venue**: ICML 2024
* **The Problem**: Complex agentic workflows, multi-turn conversations, few-shot prompts, and structured JSON decoding generate massive prefix overlap. Standard PagedAttention block tables could not efficiently match or reuse hierarchical prefixes across branched requests.
* **The Breakthrough**: **RadixAttention**. SGLang maintains the global KV cache as an LRU-managed **Radix Tree (Trie)**. Multiple requests sharing prefixes traverse the tree from the root, matching cached physical blocks with zero redundant computation and sub-token granularity.
* **Production Impact**: Created the **SGLang** engine. Delivered a $5\times - 20\times$ TTFT latency reduction in multi-turn chat, RAG, and multi-agent loops. Became the default inference backend for leading research labs and LMSYS Chatbot Arena.

---

### 5. Sarathi-Serve / Taming Throughput-Latency Tradeoffs with Chunked Prefills
* **Authors**: Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, Ramachandran Ramjee (Microsoft Research, Georgia Tech)
* **Venue**: OSDI 2024
* **The Problem**: Continuous batching suffered from severe **Prefill-Decode Interference**. When a user submitted an $8,192$-token prompt, the GPU executed a monolithic prefill GEMM for $300\text{ ms}$, freezing all active decode streams and causing catastrophic Inter-Token Latency (ITL) spikes.
* **The Breakthrough**: **Chunked-Prefills**. Sarathi splits long prompt prefills into smaller chunks ($C = 512$ tokens) and interleaves them with active decode tokens in every forward-pass iteration. Decode streams never stall, and compute-bound prefill chunks fill the idle Tensor Core capacity of memory-bound decode steps.
* **Production Impact**: Preserved flat P99 ITL distributions ($<40\text{ ms}$) even under heavy prompt loads, while driving overall GPU throughput up by $15\% - 25\%$. Integrated natively into vLLM v0.4+ and SGLang.

---

### 6. AWQ: Activation-Aware Weight Quantization for On-Device LLM Compression and Acceleration
* **Authors**: Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Chen, Wei-Chen Wang, Song Han (MIT, NVIDIA)
* **Venue**: MLSys 2024
* **The Problem**: Naive INT4 quantization (Round-To-Nearest) causes severe accuracy loss because models contain rare, large-magnitude outlier activations that carry critical reasoning capabilities.
* **The Breakthrough**: AWQ discovered that protecting the top $1\%$ of salient weight channels (identified by observing activation distributions) is sufficient to preserve model perplexity. Instead of leaving them in FP16, AWQ applies an per-channel scaling transformation that protects them mathematically on the INT4 grid.
* **Production Impact**: Made INT4 quantization practical and near-lossless for 70B+ foundation models. Powered efficient enterprise edge serving and provided the algorithmic basis for AutoAWQ.

---

### 7. Marlin: A Fast 4-bit GEMM Kernel for LLM Inference
* **Authors**: Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh (IST Austria, ETH Zurich)
* **Venue**: arXiv 2024
* **The Problem**: Existing 4-bit quantization runtimes spent too much time unpacking bits in registers, making INT4 models run *slower* than native FP16 models on batch sizes $>1$.
* **The Breakthrough**: Marlin introduced an ultra-optimized CUDA/Triton GEMM kernel that loads INT4 weights asynchronously from HBM directly into Shared Memory (`cp.async`), and executes fused dequantization directly inside register files line-rate with Tensor Core matrix multiplications.
* **Production Impact**: Unlocked near-theoretical $3.8\times - 4.0\times$ speedups over FP16, making INT4-AWQ the highest-throughput inference format for memory-constrained deployments.

---

### 8. Fast Inference from Transformers via Speculative Decoding
* **Authors**: Yaniv Leviathan, Matan Kalman, Yossi Matias (Google Research) / Angel et al. (DeepMind)
* **Venue**: ICML 2023
* **The Problem**: Autoregressive decode executes strictly one token per forward pass, running at the mercy of GPU memory bandwidth and leaving Tensor Cores $99\%$ idle.
* **The Breakthrough**: **Draft-and-Verify Rejection Sampling**. A small, fast draft model generates $K$ candidate tokens quickly. The large target model evaluates all $K$ tokens in a single parallel GEMM step. A rigorous statistical rejection sampling algorithm verifies the candidates, proving mathematically that the output distribution is bit-for-bit identical to the target model alone.
* **Production Impact**: Achieved a $2\times - 3\times$ reduction in single-stream generation latency without model retraining or quality loss. Foundation for Medusa, EAGLE, and production speculative decoding engines.

---

### 9. Efficient Streaming Language Models with Attention Sinks (StreamingLLM)
* **Authors**: Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis (MIT, Meta AI)
* **Venue**: ICLR 2024
* **The Problem**: In infinite streaming applications (or long contexts $>32\text{K}$), KV cache memory grows linearly, eventually crashing the GPU with OOM. However, using a naive sliding window cache caused model perplexity to explode immediately into complete incoherence.
* **The Breakthrough**: The discovery of **Attention Sinks**. Because the initial token (position 0) is visible to every subsequent token during pre-training, transformers dump massive unneeded attention probability mass into it to stabilize the softmax denominator. StreamingLLM proved that retaining just **the first 4 tokens (sinks)** alongside a rolling local window preserves perfect perplexity across millions of tokens in $O(1)$ constant memory.
* **Production Impact**: Enabled infinite-streaming conversational agents, memory-bounded continuous monitoring, and inspired subsequent sparse eviction architectures (H2O, Scissorhands).

---

### 10. Mooncake: A KVI-Centric Disaggregated Architecture for LLM Serving
* **Authors**: Qin et al. (Moonshot AI / Kimi, Tsinghua University)
* **Venue**: arXiv 2024
* **The Problem**: Colocating compute-heavy prefill and memory-heavy decode on the same cluster compromises both: long prompts induce decode jitter, and long decode sessions exhaust VRAM needed for prefill batching.
* **The Breakthrough**: **KVI-Centric Disaggregation**. Mooncake completely separates prefill nodes from decode nodes, utilizing a high-speed distributed memory store connected via $400\text{Gbps}$ GPUDirect RDMA. KV caches computed on prefill nodes are written directly into decode node VRAM over optical fabrics, bypassing host CPUs.
* **Production Impact**: Powers the production infrastructure of Moonshot AI (serving millions of 200K+ context queries daily). Demonstrated a $50\%+$ improvement in cluster throughput and flat P99 ITL. Established disaggregated serving as the standard blueprint for hyperscale frontier AI clusters.

---

## 3. Summary Matrix of Foundational Papers

| Innovation | Seminal Paper | Primary Hardware Bottleneck Solved | Core Metric Optimized |
|---|---|---|---|
| **FlashAttention** | Dao et al. (2022) | GPU HBM Read/Write Traffic ($O(N^2)$) | IO Memory Bandwidth |
| **Continuous Batching** | Yu et al. - Orca (2022) | Static Batch GPU Idle Bubbles ($30\% - 60\%$) | System Token Throughput |
| **PagedAttention** | Kwon et al. - vLLM (2023) | KV Cache Internal & External Fragmentation | Memory Capacity & Concurrency |
| **RadixAttention** | Zheng et al. - SGLang (2024) | Redundant Prefix Computation (Chat/RAG) | Time to First Token (TTFT) |
| **Chunked Prefill** | Agrawal et al. - Sarathi (2024) | Prefill-induced Decode Stalls & Jitter | P99 Inter-Token Latency (ITL) |
| **AWQ & Marlin** | Lin (2023) / Frantar (2024) | Decode Memory Bus Bandwidth ($140\text{ GB}$ weights) | Token Generation Latency |
| **Speculative Decoding** | Leviathan et al. (2023) | Single-Token Autoregressive Serialization | Wall-Clock Generation Speed |
| **StreamingLLM** | Xiao et al. (2024) | Linear KV Cache Memory Growth ($O(S)$) | $O(1)$ Constant VRAM Footprint |
| **Disaggregated Serving**| Mooncake / DistServe (2024) | Prefill/Decode Silicon Colocation Conflict | Tail Latency & Cluster TCO |

---

## 4. Key Takeaways

1. **Every optimization solves a physical constraint**: Modern inference techniques do not invent mathematical tricks in a vacuum; they circumvent hardware bottlenecks in memory bandwidth, SRAM capacity, or Tensor Core starvation.
2. **The shift from batch-level to iteration-level**: Orca and vLLM redefined serving from coarse request scheduling to fine-grained iteration and block-level virtualization.
3. **Prefixes are the new cache hierarchy**: RadixAttention and Mooncake treat KV caches as distributed, hierarchically-indexed memory tiers rather than transient session buffers.
4. **The future is disaggregated**: As context lengths scale into the millions, physical disaggregation of prefill (compute) and decode (bandwidth/capacity) is the definitive architecture for 2026/2027 enterprise systems.
