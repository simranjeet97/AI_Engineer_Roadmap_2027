# 05. Automatic Prefix Caching (APC), Radix Trees & Workload Profiling

> *"The most efficient computation in distributed systems is the one that was computed yesterday and retrieved in constant time today."*

In modern enterprise LLM applications, a staggering fraction of input tokens are completely redundant. Whether it is an 80-line system prompt setting a persona, a 20-page legal document in a Retrieval-Augmented Generation (RAG) pipeline, multi-turn conversational history in customer support, or multi-step agent scratchpads, the exact same sequence of tokens is repeatedly dispatched to the inference engine.

Without caching, the inference engine computes Key ($K$) and Value ($V$) tensors for those identical tokens over and over again, consuming hundreds of thousands of Tensor Core cycles and burning unnecessary megawatts of power.

This technical memo derives **Automatic Prefix Caching (APC)** and **Radix Attention Trees** from absolute first principles: the mathematical determinism of attention, hash-based vs trie-based memory indexing, eviction mechanics, and workload profiling.

---

## 1. The Redundant Computation Problem

### The Teacher Analogy
Imagine a university professor grading 200 final exam essays:
* Every essay prompt begins with an identical 5-page case study describing the financial collapse of an airline.
* The students answer 3 short questions at the end of the case study.
* Does the professor read the 5-page case study from scratch for student #1, then read it from scratch for student #2, and read it from scratch 200 times?
* Of course not! The professor reads the case study once, keeps the mental model in working memory, and for every subsequent student, only reads the student's unique answers.

In naive LLM serving, the GPU behaves like an amnesiac professor: every incoming HTTP request causes the model to re-process the exact same 5-page prompt token by token, projecting it through 80 transformer layers from scratch.

### Quantifying the Waste
Consider an enterprise RAG agent serving customer queries over a company knowledge base:
* System prompt + retrieved documents: **$4,096$ tokens**
* User question: **$64$ tokens**
* Generation output: **$128$ tokens**
* Daily query volume: **$100,000$ requests/day**

Let us compute the daily compute cost of the prefill stage on an 8x H100 cluster running Llama-3-70B:
$$\text{Redundant Tokens per Request} = 4,096 \text{ tokens}$$
$$\text{Daily Redundant Tokens} = 100,000 \times 4,096 = \mathbf{409,600,000 \text{ tokens/day}}$$

Using our first-principles prefill throughput for 70B FP8 on 8x H100 ($\approx 35,000 \text{ prefill tokens/sec}$):
$$\text{Daily GPU Time Wasted} = \frac{409,600,000}{35,000} \approx \mathbf{11,702 \text{ seconds}} \approx \mathbf{3.25 \text{ hours of 100\% GPU compute}}$$

At an H100 cluster rental rate of $\$26.00/\text{hr}$, that is over **$\$30,000$ per year spent computing the exact same matrix multiplications repeatedly on identical text!**

Even worse, every user suffers an unnecessary prefill latency delay of $\approx 120\text{ ms}$ on every single turn of interaction.

---

## 2. What Exactly Gets Cached?

### The Mathematical Determinism of Transformer Attention
Recall the self-attention formulation for token position $t$:
$$Q_t = X_t W_Q, \quad K_t = X_t W_K, \quad V_t = X_t W_V$$

In a causal transformer with standard positional embeddings (RoPE - Rotary Position Embedding):
1. The token representations at layer $l$ depend **strictly** on:
   * The input token IDs from position $0$ up to position $t$.
   * The static model weights ($W_Q, W_K, W_V, W_O, W_{MLP}$).
2. Because inference uses greedy or fixed sampling during forward evaluation, the Key and Value vectors for position $t$ are **100% deterministic**.

If Request A begins with:
`[128000, 9906, 1917, 330, 279, ...]`
and Request B begins with the exact same token IDs:
`[128000, 9906, 1917, 330, 279, ...]`

The $K$ and $V$ tensors computed for those tokens at layer $1, 2, \dots, L$ are **bit-for-bit identical**.

### The Cache Reuse Mechanism
Instead of discarding the $K$ and $V$ tensors after Request A finishes, the inference engine **retains those physical memory blocks in GPU VRAM**.

When Request B arrives:
1. The engine scans the input tokens and identifies that the first $4,096$ tokens already exist in GPU memory.
2. The engine **skips the prefill computation** for those $4,096$ tokens entirely!
3. The engine points the block table of Request B directly to the existing physical KV blocks.
4. The GPU only performs prefill on the remaining $64$ novel user tokens.

$$\text{TTFT}_{\text{cold}} = t_{\text{queue}} + t_{\text{prefill}}(4160) \approx 5\text{ ms} + 120\text{ ms} = \mathbf{125\text{ ms}}$$
$$\text{TTFT}_{\text{warm}} = t_{\text{queue}} + t_{\text{prefill}}(64) \approx 5\text{ ms} + 2\text{ ms} = \mathbf{7\text{ ms}}$$

**TTFT drops from $125\text{ ms}$ to $7\text{ ms}$ — an astounding $17.8\times$ latency reduction!**

---

## 3. Hash-Based Block Caching (vLLM Architecture)

How does an engine determine whether a sequence of tokens has already been computed?

In vLLM, prefix caching is implemented on top of the PagedAttention memory manager using **Cryptographic Block Hashing**.

### The Mechanism
1. Recall that PagedAttention divides the sequence into discrete logical blocks of size $B_s$ (typically $B_s = 16$ tokens).
2. The hash of a block is computed as a chained function of the previous block's hash and the current block's token IDs:
   $$h_0 = \text{hash}(\text{tokens}[0 \dots 15])$$
   $$h_1 = \text{hash}(h_0, \text{tokens}[16 \dots 31])$$
   $$h_k = \text{hash}(h_{k-1}, \text{tokens}[k \cdot B_s \dots (k+1) \cdot B_s - 1])$$
3. The engine maintains a global lookup table in CPU host memory:
   $$\text{HashTable}: \text{Block Hash } h_k \longrightarrow \text{Physical Block ID in GPU HBM}$$

```
Hash-Based Block Table Lookup:
Request Tokens: [T0 ... T15]  [T16 ... T31]  [T32 ... T47]  [T48 ... T55]
                └─────┬─────┘  └─────┬──────┘  └─────┬──────┘  └─────┬─────┘
                      │              │               │               │
                   Hash h0        Hash h1         Hash h2         Partial (<16)
                      │              │               │               │
                      ▼              ▼               ▼               ▼
                 [HIT: Blk 42]  [HIT: Blk 88]   [MISS: Compute] [MISS: Compute]
                      │              │               │               │
                      └───────┬──────┘               └───────┬───────┘
                              │                              │
                     Reused from GPU HBM             Prefilled by GPU
```

### The Block-Alignment Limitation
Because hashing operates on fixed-size blocks ($B_s = 16$):
* Any token match that does not align to a multiple of 16 cannot be shared via a cached block.
* For example, if a system prompt is $505$ tokens long:
  $$505 = 31 \times 16 + 9$$
* The first $31$ blocks ($496$ tokens) are cached and reused.
* The remaining $9$ tokens in the incomplete block must be recomputed on every request.

---

## 4. Radix Tree Prefix Caching (SGLang Architecture)

To overcome the rigid constraints of block-aligned hashing and maximize memory reuse across complex agentic workflows, **SGLang** introduced **RadixAttention**.

### What is a Radix Tree (Trie)?
A Radix Tree (or compact trie) is a tree data structure where each edge represents a sequence of tokens rather than a single character.

Let us look at a simple vocabulary example with strings:
* Inserting `"apple"`, `"apply"`, `"application"`, `"banana"`

```
Radix Tree String Representation:
               (Root)
              /      \
       "appl"          "banana"
       /    \
     "e"    "y"
             |
          "cation"
```
Notice how `"appl"` is stored only once and shared across `"apple"`, `"apply"`, and `"application"`.

### Applying Radix Trees to GPU KV Caches
In SGLang's `RadixCache`:
* Every node in the tree corresponds to a sequence of tokens.
* Every node holds a list of **physical GPU KV cache block pointers**.
* A reference counter (`ref_count`) tracks how many active requests are currently reading or appending to that path in the tree.

```
SGLang Radix Tree for Enterprise Customer Chat:
                                    (Root)
                                       │
                      ["System: You are an enterprise assistant..."]
                      [Tokens 0 - 511 | Cached in Physical Blocks 1-32]
                                       │
                       ┌───────────────┴───────────────┐
                       │                               │
            ["User: Check order #1234"]     ["User: Reset my password"]
            [Tokens 512-530 | Blocks 33-34] [Tokens 512-525 | Blocks 40-41]
                       │                               │
            ["Agent: Shipped via FedEx..."]  ["Agent: Email sent to..."]
            [Tokens 531-560 | Blocks 35-37] [Tokens 526-550 | Blocks 42-43]
                       │
            ["User: What is the tracking #?"]
            [Tokens 561-575 | New Prefill!]
```

### The Lookup and Match Algorithm
When a new request arrives with token sequence $T$:
1. `match_prefix(T)` traverses the Radix Tree from the root, greedily following matching edges.
2. It returns:
   * The list of physical block IDs for the matched prefix.
   * The offset of the first unmatched token.
3. The model only executes prefill on the unmatched suffix!
4. The newly generated tokens are then added as a new child branch to the tree, immediately available for future requests or subsequent chat turns.

---

## 5. Multi-Turn Conversation Dynamics

Multi-turn conversational AI is the ultimate beneficiary of prefix caching.

Consider a 3-turn customer support conversation:

### Turn 1:
* Input: `[System Prompt (500 tok)]` + `[User Question 1 (50 tok)]`
* Prefill: Computes all $550$ tokens.
* Generation: Emits $100$ tokens (`[Agent Answer 1]`).
* Tree state: Path of length $650$ tokens stored in KV cache.

### Turn 2:
* Input: `[System Prompt (500 tok)]` + `[User Question 1 (50 tok)]` + `[Agent Answer 1 (100 tok)]` + `[User Question 2 (40 tok)]`
* Total Context: $690$ tokens.
* Without Prefix Caching: GPU must prefill all $690$ tokens!
* With Radix Cache:
  * Tree matches the first $650$ tokens with $100\%$ precision!
  * GPU only prefills the novel $40$ tokens of Question 2.
  * **$94.2\%$ of the prefill computation is eliminated!**

### Turn 3:
* Total Context: $850$ tokens.
* Radix Cache matches $790$ tokens from Turns 1 and 2.
* GPU only prefills the $60$ tokens of Question 3.
* **$92.9\%$ of the prefill computation is eliminated!**

```
Turn-by-Turn Latency Comparison (Llama-3-70B on 8x H100):
┌─────────┬──────────────────────┬──────────────────────┬───────────┐
│ Turn    │ TTFT Without Caching │ TTFT With RadixCache │ Speedup   │
├─────────┼──────────────────────┼──────────────────────┼───────────┤
│ Turn 1  │ 22.5 ms              │ 22.5 ms (Cold)       │ 1.0x      │
│ Turn 2  │ 26.2 ms              │ 2.1 ms (Warm)        │ 12.5x     │
│ Turn 3  │ 31.8 ms              │ 2.4 ms (Warm)        │ 13.2x     │
│ Turn 5  │ 48.0 ms              │ 2.8 ms (Warm)        │ 17.1x     │
│ Turn 10 │ 89.5 ms              │ 3.2 ms (Warm)        │ 27.9x     │
└─────────┴──────────────────────┴──────────────────────┴───────────┘
```

As conversations lengthen, non-cached TTFT degrades linearly, while prefix-cached TTFT remains flat at $\approx 2 - 3\text{ ms}$!

---

## 6. Eviction Policies: LRU Under Memory Pressure

GPU VRAM is finite. An 8x H100 cluster has $640\text{ GB}$ total VRAM. After subtracting model weights ($70\text{ GB}$ in FP8) and runtime buffers, approximately $550\text{ GB}$ remains for the KV cache pool.

When thousands of users submit diverse prompts, the Radix Tree expands until the free block pool is exhausted. What happens next?

### Reference Counting & Tree Eviction
Every node in the Radix Tree maintains:
* `ref_count`: The number of active requests currently reading or appending to this node.
* `last_access_time`: Timestamp of when this node was last matched by a request.

Rules of the Eviction Loop:
1. **Nodes with `ref_count > 0` are PINNED**: They cannot be evicted because a running request is actively computing attention over their KV blocks.
2. **Nodes with `ref_count == 0` are EVICTABLE**: These are cached historical prefixes from completed requests.
3. When free memory drops below a safety watermark (e.g., $<5\%$ free blocks):
   * The scheduler invokes `_evict_lru()`.
   * It scans all evictable leaf nodes (`ref_count == 0`).
   * It evicts the leaf node with the oldest `last_access_time`.
   * Evicting a node returns its physical blocks to the engine's free block pool.
   * If a parent node now has zero children and `ref_count == 0`, it becomes a leaf node and is eligible for eviction in the next cycle.

```
Radix Tree LRU Eviction:
              (Root)
             /      \
        Node A      Node B [ref_count=2, PINNED!]
       /      \
   Node C     Node D [ref_count=0, last_accessed=10 mins ago] -> EVICTED FIRST!
  [ref=1]     (Physical blocks returned to free pool)
```

---

## 7. Workload Profiling & Cache Hit Rate

Prefix caching is not a silver bullet for every workload. Its efficacy depends entirely on the **locality of the token distribution**.

### Workload Taxonomy:

| Workload Type | Prefix Sharing Characteristics | Expected Cache Hit Rate | Typical TTFT Speedup |
|---|---|---|---|
| **Multi-Turn Chat** | Shared system prompt + shared history per user | $70\% - 95\%$ | **$8\times - 25\times$** |
| **Agent Workflows (ReAct)** | Repetitive system instructions + tool definitions | $80\% - 98\%$ | **$10\times - 30\times$** |
| **RAG over Fixed Docs** | Shared corporate documents / PDF attachments | $50\% - 85\%$ | **$4\times - 12\times$** |
| **Code Completion** | Shared repository context, imports, type definitions | $60\% - 85\%$ | **$5\times - 15\times$** |
| **Independent Ad-Hoc Q&A** | Diverse single-turn prompts with no shared prefix | $< 5\%$ | **$1.0\times$ (Negligible)** |

### Monitoring in Prometheus
Track the following Prometheus metrics to measure cache effectiveness:
* `vllm:gpu_prefix_cache_hit_rate`: Ratio of prompt tokens found in cache vs total prompt tokens.
* `sglang:tree_cache_hit_rate`: Fraction of requests matching radix tree nodes.

**Rule of Thumb**:
* If hit rate $> 60\%$: Prefix caching is your primary latency and cost saver.
* If hit rate $< 10\%$: Your workload consists of disjoint, single-shot queries; memory spent retaining old KV blocks might be better repurposed for larger batch sizes.

---

## 8. Key Takeaways

1. **Repetitive prefill is pure waste**: Recomputing identical $K, V$ matrices for recurring system prompts and RAG contexts burns GPU power and inflates TTFT.
2. **Attention is deterministic**: For a fixed model, identical prefix token IDs always generate identical $K, V$ vectors across all layers.
3. **vLLM uses block hashing**: Fast, fixed $O(1)$ lookups, but constrained to 16-token block-aligned boundaries.
4. **SGLang uses Radix Trees**: Compact tries that support arbitrary prefix lengths, partial matching, and hierarchical branching for agentic workflows.
5. **Multi-turn latency is transformed**: Multi-turn TTFT drops from $50\text{ ms} - 100\text{ ms}$ down to $2\text{ ms} - 3\text{ ms}$, maintaining near-instantaneous interactivity across long conversations.
6. **LRU tree eviction**: GPU memory stays bounded by evicting unreferenced leaf nodes in least-recently-used order when the free pool drops below the safety watermark.
