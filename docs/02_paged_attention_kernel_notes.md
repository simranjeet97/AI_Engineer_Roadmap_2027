# PagedAttention: Virtual Memory for GPU KV Caches

## 1. The Memory Problem That Almost Killed LLM Serving

Imagine you're running a restaurant. Each customer (request) needs a table (GPU memory). But here's the catch — you don't know in advance how long each customer will stay or how much food they'll order. They might order a seven-course feast, or they might just grab a coffee and leave.

In early LLM serving (before 2023), the "solution" was brutal: **reserve the biggest possible table for every single customer**. Ordered a coffee? Doesn't matter — you get the 20-seat banquet table, and nobody else can sit there.

Translated to GPU memory: every incoming request pre-allocated KV cache memory for the **full maximum sequence length** — typically 4096 or 8192 tokens — even if the actual generation ended up being only 50 tokens.

```
Early LLM Serving Memory Layout (one A100 80GB GPU):
┌──────────────────────────────────────────────────────────┐
│                  Model Weights (~14 GB)                   │
├──────────────────────────────────────────────────────────┤
│  Request 1: Reserved 512 MB  │ Actually used: 6 MB       │  ← 98.8% WASTED
├──────────────────────────────────────────────────────────┤
│  Request 2: Reserved 512 MB  │ Actually used: 25 MB      │  ← 95.1% WASTED
├──────────────────────────────────────────────────────────┤
│  Request 3: Reserved 512 MB  │ Actually used: 100 MB     │  ← 80.5% WASTED
├──────────────────────────────────────────────────────────┤
│  ... maybe 5-6 more requests fit ...                      │
├──────────────────────────────────────────────────────────┤
│  WASTED SPACE (can't serve more requests)                 │
└──────────────────────────────────────────────────────────┘
```

The result? A $30,000 GPU serving maybe **5-10 concurrent requests** when it could theoretically handle **50-100+**. Throughput was artificially throttled by memory management, not by compute. This was the crisis that PagedAttention solved.

---

## 2. What is the KV Cache?

Before we understand *why* the KV cache eats so much memory, we need to understand *what* it is and *why* it exists. Let's build this up from the core mechanism of transformers: self-attention.

### 2.1 Self-Attention: The Quick Review

In a transformer layer, self-attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

For a sequence of $n$ tokens, we project every token into three vectors:

- **Q** (Query): "What am I looking for?" — shape $(n, d_k)$
- **K** (Key): "What do I contain?" — shape $(n, d_k)$
- **V** (Value): "What information do I provide?" — shape $(n, d_v)$

The attention matrix $Q K^T$ has shape $(n, n)$ — every token attends to every other token.

### 2.2 The Problem During Generation

During text **generation** (inference), we produce tokens **one at a time**. When we generate token $t_{n+1}$, we need to compute:

$$\text{attention}_{n+1} = \text{softmax}\left(\frac{q_{n+1} \cdot [k_1, k_2, \ldots, k_n]^T}{\sqrt{d_k}}\right) \cdot [v_1, v_2, \ldots, v_n]$$

The new token's query $q_{n+1}$ must attend to the keys and values of **ALL previous tokens**.

**Without a cache**, every new token forces us to:
1. Re-run the entire sequence through the model
2. Recompute all $K$ and $V$ projections for all $n$ previous tokens
3. Then compute attention

This is $O(n^2)$ work **per generated token** — for a 4096-token sequence, generating the last token alone requires processing all 4096 tokens through every layer. Generate 100 tokens and you've done $\sum_{i=1}^{100}(4096+i) \approx 419{,}600$ token-layer passes. Absurdly wasteful.

**With a cache**, we:
1. Store $K$ and $V$ vectors for all previously processed tokens
2. For each new token, only compute $q_{new}$, $k_{new}$, $v_{new}$
3. Append $k_{new}$ and $v_{new}$ to the cache
4. Compute attention of $q_{new}$ against the full cached $K$ and $V$

This is $O(n)$ work per token — a massive speedup.

```
WITHOUT KV Cache (Naive):                WITH KV Cache:
                                          
Token 1: process [t1]                     Token 1: process [t1], store K1,V1
Token 2: process [t1, t2]                 Token 2: process [t2], lookup K1,V1 from cache
Token 3: process [t1, t2, t3]             Token 3: process [t3], lookup K1..2,V1..2 from cache
Token 4: process [t1, t2, t3, t4]         Token 4: process [t4], lookup K1..3,V1..3 from cache
  ...                                       ...
Token n: process [t1, ..., tn]            Token n: process [tn], lookup K1..n-1,V1..n-1

Total work: 1+2+3+...+n = n(n+1)/2       Total work: 1+1+1+...+1 = n (for projections)
         = O(n²)                                    + n(n+1)/2 for attention dots
                                                    (but attention is cheap vs. full forward pass)
```

The KV cache is **essential** for efficient inference. Without it, LLM serving would be 10-100x slower. But it comes at a steep memory cost.

### 2.3 Calculating KV Cache Memory: Byte by Byte

For each token position in the sequence, we must store:
- One **K** vector and one **V** vector
- For **every** attention layer in the model
- For **every** KV head in that layer (note: models using Grouped Query Attention have fewer KV heads than query heads)

The formula:

$$\text{bytes per token} = 2 \times n_{layers} \times n_{kv\_heads} \times d_{head} \times b_{dtype}$$

Where:
| Symbol | Meaning |
|--------|---------|
| $2$ | One K vector + one V vector |
| $n_{layers}$ | Number of transformer layers |
| $n_{kv\_heads}$ | Number of key-value heads (may differ from query heads in GQA) |
| $d_{head}$ | Dimension of each attention head |
| $b_{dtype}$ | Bytes per element (2 for BF16/FP16, 1 for INT8) |

### 2.4 Worked Example: Llama-3-8B

Llama-3-8B uses Grouped Query Attention (GQA):
- $n_{layers} = 32$
- $n_{kv\_heads} = 8$ (not 32! GQA shares KV heads across query head groups)
- $d_{head} = 128$
- $b_{dtype} = 2$ (BF16)

$$\text{bytes per token} = 2 \times 32 \times 8 \times 128 \times 2 = 131{,}072 \text{ bytes} = 128 \text{ KB}$$

**128 KB per token.** That might sound small, but watch what happens at scale:

| Sequence Length | KV Cache per Request | 100 Concurrent Requests |
|----------------|---------------------|------------------------|
| 50 tokens      | 6.25 MB             | 625 MB                 |
| 512 tokens     | 64 MB               | 6.25 GB                |
| 2048 tokens    | 256 MB              | 25 GB                  |
| 4096 tokens    | 512 MB              | 50 GB                  |
| 8192 tokens    | 1 GB                | 100 GB (!)             |

For a single request at max context (4096 tokens): **512 MB** just for the KV cache. The model weights themselves are only ~16 GB in BF16. The KV cache can easily **dwarf** the model.

### 2.5 Larger Models Are Even Hungrier

For Llama-3-70B ($n_{layers}=80$, $n_{kv\_heads}=8$, $d_{head}=128$, BF16):

$$\text{bytes per token} = 2 \times 80 \times 8 \times 128 \times 2 = 327{,}680 \text{ bytes} = 320 \text{ KB}$$

At 4096 tokens: $320 \text{ KB} \times 4096 = 1.28 \text{ GB}$ per request. Ten concurrent requests consume **12.8 GB** of KV cache alone.

---

## 3. Internal vs External Fragmentation

Now that we know the KV cache is enormous, let's understand the two types of memory waste that plagued pre-PagedAttention systems.

### 3.1 Internal Fragmentation: The Bus-in-a-Motorcycle-Spot Problem

Think of a parking lot. You have parking spots designed for buses (4096 tokens). A motorcycle (50-token generation) pulls up and parks. The spot is "occupied" — no other vehicle can use it — but 98.8% of the space is empty.

```
Pre-allocated KV Cache for max_seq_len = 4096:

Request A (actual: 50 tokens):
┌──┬──────────────────────────────────────────────────────┐
│██│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
└──┴──────────────────────────────────────────────────────┘
 50 tokens used              4046 tokens WASTED (98.8%)

Request B (actual: 200 tokens):
┌────────┬────────────────────────────────────────────────┐
│████████│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
└────────┴────────────────────────────────────────────────┘
 200 tokens used             3896 tokens WASTED (95.1%)

Request C (actual: 1000 tokens):
┌────────────────────┬────────────────────────────────────┐
│████████████████████│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│
└────────────────────┴────────────────────────────────────┘
 1000 tokens used            3096 tokens WASTED (75.4%)
```

In practice, the average generation length is far shorter than `max_model_len`. Studies show that typical chatbot completions average 200-500 tokens. That means **80-95% internal fragmentation** is common.

**Concrete calculation:**
- Average actual sequence length: 200 tokens
- Pre-allocated for: 4096 tokens
- Internal fragmentation: $(4096 - 200) / 4096 = 95.1\%$

You're paying for 20x more memory than you actually use. On an 80 GB GPU, this is like having 76 GB reserved but only 4 GB doing useful work.

### 3.2 External Fragmentation: The Swiss Cheese Problem

Even after requests complete and "free" their memory, you end up with scattered holes that can't be used efficiently.

```
GPU Memory Timeline (time flows downward):
─────────────────────────────────────────────

T=0:  [  Req A: 512MB  ][  Req B: 512MB  ][  Req C: 512MB  ][ Free: 512MB ]

T=1:  [  Req A: 512MB  ][    FREE 512MB   ][  Req C: 512MB  ][ Free: 512MB ]
      (Req B completed)

T=2:  [    FREE 512MB   ][    FREE 512MB   ][  Req C: 512MB  ][ Free: 512MB ]
      (Req A completed)

      Total free: 1536 MB  — but it's in THREE non-contiguous chunks!
      
      New Req D needs 600 MB contiguous? DENIED! 
      (No single free block is big enough, even though total free > 600 MB)
```

This is **external fragmentation**: you have enough total free memory, but it's scattered across non-contiguous regions, and no single region is large enough for a new allocation.

### 3.3 The Combined Impact

Let's do the math on an A100-80GB serving Llama-3-8B:

```
Total GPU Memory:                           80 GB
Model weights (BF16):                      -16 GB
Activations, CUDA overhead, etc.:           -4 GB
                                           ──────
Memory available for KV cache:              60 GB

Without PagedAttention (worst case):
  Per-request reservation:                  512 MB (4096 tokens × 128 KB)
  Max concurrent requests:                  60 GB / 512 MB = 117 requests
  But avg actual usage per request:         25 MB  (200 tokens × 128 KB)
  Actual memory doing useful work:          117 × 25 MB = 2.86 GB
  Effective utilization:                    2.86 / 60 = 4.8% (!!)

  Plus external fragmentation eats another 20-30% of capacity.
  Real-world concurrent requests:           ~60-80 (not even 117)
```

**4.8% memory utilization.** You bought an $30,000 GPU and 95% of the KV cache memory is sitting idle. This is the crisis PagedAttention was designed to solve.

---

## 4. PagedAttention: The Virtual Memory Insight

In 2023, Woosuk Kwon et al. at UC Berkeley published *"Efficient Memory Management for Large Language Model Serving with PagedAttention"*. The core insight was beautifully simple: **operating systems solved this exact same problem 50 years ago**.

### 4.1 How Your OS Handles Memory

When you run a program on your computer, it thinks it has a nice, contiguous block of memory — addresses 0x0000 through 0xFFFF (or wherever). But that's a **virtual** address space. In reality, the OS maps these virtual addresses to **physical** pages scattered throughout RAM.

```
Virtual Memory (what the program sees):     Physical RAM (actual hardware):
┌──────────┐                                ┌──────────┐
│ Page 0   │ ─────────────────────────────► │ Frame 47 │
├──────────┤                                ├──────────┤
│ Page 1   │ ──────────────┐                │ Frame 3  │ ◄── Page 2
├──────────┤               │                ├──────────┤
│ Page 2   │ ──────┐       │                │ Frame 91 │ ◄── Page 0  (was Frame 47)
├──────────┤       │       │                ├──────────┤
│ Page 3   │ ──┐   │       │                │ Frame 12 │ ◄── Page 3
├──────────┤   │   │       │                ├──────────┤
│ (unused) │   │   │       └──────────────► │ Frame 55 │ ◄── Page 1
└──────────┘   │   └──────────────────────► │  ...     │
               └──────────────────────────► │  ...     │
                                            └──────────┘

              Page Table
           ┌────────┬──────────┐
           │ VPage  │ PFrame   │
           ├────────┼──────────┤
           │   0    │   91     │
           │   1    │   55     │
           │   2    │    3     │
           │   3    │   12     │
           └────────┴──────────┘
```

The program doesn't care where its data physically lives. The **page table** handles the translation transparently. This eliminates fragmentation because physical frames don't need to be contiguous.

### 4.2 PagedAttention: The Same Trick for KV Caches

PagedAttention applies this exact principle to GPU memory:

| OS Concept | PagedAttention Equivalent |
|-----------|--------------------------|
| Virtual page | Logical KV block |
| Physical frame | Physical GPU memory block |
| Page table | Block table |
| Page size (4 KB) | Block size (16 tokens' worth of KV data) |
| Process | Request / Sequence |
| Page fault (allocate on demand) | Allocate new block when current block fills up |

Each request has a **block table** — a simple array that maps logical block indices to physical block indices in GPU memory.

```
Sequence "Tell me about quantum computing..."
(currently 37 tokens generated)

Logical View (what the sequence "sees"):
┌─────────────┬─────────────┬─────────────┐
│ Block 0     │ Block 1     │ Block 2     │
│ tokens 0-15 │ tokens 16-31│ tokens 32-47│
│ (full)      │ (full)      │ (5/16 used) │
└─────────────┴─────────────┴─────────────┘

Block Table:
┌──────────┬────────────────┐
│ Logical  │ Physical Block │
├──────────┼────────────────┤
│    0     │      42        │
│    1     │     147        │
│    2     │       8        │
└──────────┴────────────────┘

Physical GPU Memory (scattered blocks):
Block 8:   [tokens 32-36 KV data][empty slots 37-47]  ← Block 2 lives here
Block 42:  [tokens 0-15 KV data]                       ← Block 0 lives here  
Block 147: [tokens 16-31 KV data]                      ← Block 1 lives here

Note: Physical blocks are NOT contiguous! And that's perfectly fine.
```

### 4.3 How Big is a Physical Block?

Each physical block stores KV data for `block_size` tokens (typically 16). Using our Llama-3-8B numbers:

$$\text{block size in bytes} = \text{block\_size} \times \text{bytes\_per\_token} = 16 \times 128 \text{ KB} = 2 \text{ MB}$$

For a 60 GB KV cache budget:

$$\text{total physical blocks} = \frac{60 \text{ GB}}{2 \text{ MB}} = 30{,}720 \text{ blocks}$$

That's a pool of 30,720 blocks to be shared across ALL concurrent requests, allocated on demand.

---

## 5. Address Translation Step by Step

Let's trace exactly how the PagedAttention kernel finds the KV data for a specific token. This is the critical inner loop that runs on the GPU.

### 5.1 The Lookup Process

Say we need to read the K and V vectors for **token position 37** in a sequence:

```
Step 1: Compute logical block index
        logical_block = floor(37 / 16) = 2

Step 2: Compute offset within the block
        block_offset = 37 % 16 = 5

Step 3: Look up physical block from block table
        block_table[2] = physical_block_8

Step 4: Compute physical memory address
        physical_addr = base_ptr 
                      + physical_block_8 × block_stride
                      + 5 × token_stride
                      
        where:
          block_stride = block_size × num_kv_heads × head_dim × sizeof(dtype)
                       = 16 × 8 × 128 × 2 = 32,768 bytes
          token_stride = num_kv_heads × head_dim × sizeof(dtype)
                       = 8 × 128 × 2 = 2,048 bytes
```

### 5.2 The CUDA Kernel (Simplified)

Here's a simplified version of what the PagedAttention CUDA kernel does. The actual vLLM implementation is heavily optimized, but this captures the core logic:

```cuda
// Simplified PagedAttention kernel
// Each thread block handles one query head for one sequence
__global__ void paged_attention_kernel(
    const float* __restrict__ q,           // [num_seqs, num_heads, head_dim]
    const float* __restrict__ k_cache,     // [num_blocks, num_kv_heads, block_size, head_dim]
    const float* __restrict__ v_cache,     // [num_blocks, num_kv_heads, block_size, head_dim]
    const int*   __restrict__ block_tables,// [num_seqs, max_num_blocks]
    const int*   __restrict__ context_lens,// [num_seqs]
    float*       __restrict__ output,      // [num_seqs, num_heads, head_dim]
    int block_size,
    int num_kv_heads,
    int head_dim,
    float scale                            // 1/sqrt(head_dim)
) {
    int seq_idx  = blockIdx.x;   // Which sequence
    int head_idx = blockIdx.y;   // Which attention head
    int context_len = context_lens[seq_idx];
    
    // Load query vector for this head into shared memory
    float q_vec[HEAD_DIM];
    for (int d = 0; d < head_dim; d++) {
        q_vec[d] = q[seq_idx * num_heads * head_dim + head_idx * head_dim + d];
    }
    
    // Map query head → KV head (for GQA)
    int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    
    float max_score = -INFINITY;
    float exp_sum = 0.0f;
    float acc[HEAD_DIM] = {0};
    
    // Iterate over ALL tokens in context via block table
    int num_blocks = (context_len + block_size - 1) / block_size;
    
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        // ===== THIS IS THE KEY INDIRECTION =====
        // Look up physical block number from block table
        int physical_block = block_tables[seq_idx * max_num_blocks + block_idx];
        
        // Determine how many tokens are valid in this block
        int tokens_in_block = min(block_size, 
                                  context_len - block_idx * block_size);
        
        for (int tok = 0; tok < tokens_in_block; tok++) {
            // Compute physical address of this token's K vector
            // k_cache layout: [num_blocks, num_kv_heads, block_size, head_dim]
            int k_offset = physical_block * (num_kv_heads * block_size * head_dim)
                         + kv_head_idx * (block_size * head_dim)
                         + tok * head_dim;
            
            // Compute dot product: q · k
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q_vec[d] * k_cache[k_offset + d];
            }
            score *= scale;  // Apply 1/sqrt(d_k)
            
            // Online softmax (numerically stable)
            float old_max = max_score;
            max_score = fmaxf(max_score, score);
            float exp_score = expf(score - max_score);
            float correction = expf(old_max - max_score);
            exp_sum = exp_sum * correction + exp_score;
            
            // Accumulate weighted V values
            int v_offset = physical_block * (num_kv_heads * block_size * head_dim)
                         + kv_head_idx * (block_size * head_dim)
                         + tok * head_dim;
            
            for (int d = 0; d < head_dim; d++) {
                acc[d] = acc[d] * correction + exp_score * v_cache[v_offset + d];
            }
        }
    }
    
    // Normalize by softmax denominator and write output
    for (int d = 0; d < head_dim; d++) {
        output[seq_idx * num_heads * head_dim + head_idx * head_dim + d] 
            = acc[d] / exp_sum;
    }
}
```

The magic line is:
```cuda
int physical_block = block_tables[seq_idx * max_num_blocks + block_idx];
```

This single table lookup is what enables non-contiguous memory. Instead of assuming KV data lives at `base + seq_idx * max_seq_len * kv_size`, we look up the **actual** physical location through the block table. It's one extra memory read — a trivial cost compared to the memory savings.

---

## 6. The Block Allocator

The block allocator is the component that manages the free pool of physical blocks. It's surprisingly simple — much simpler than a general-purpose memory allocator.

### 6.1 Initialization

At server startup, the KV cache memory is pre-divided into fixed-size physical blocks:

```python
# Simplified block allocator (Python-side, runs on CPU)
class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        # All blocks start as free
        self.free_blocks: List[int] = list(range(num_blocks))
        # Reference counts for copy-on-write
        self.ref_counts: Dict[int, int] = {i: 0 for i in range(num_blocks)}
    
    def allocate(self) -> int:
        """Allocate a single physical block."""
        if not self.free_blocks:
            raise RuntimeError("Out of GPU memory blocks!")
        block_id = self.free_blocks.pop()
        self.ref_counts[block_id] = 1
        return block_id
    
    def free(self, block_id: int) -> None:
        """Return a block to the free pool."""
        self.ref_counts[block_id] -= 1
        if self.ref_counts[block_id] == 0:
            self.free_blocks.append(block_id)
    
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)
```

### 6.2 On-Demand Allocation During Generation

Here's the critical insight: blocks are allocated **one at a time** as the sequence grows. Not all at once.

```
Token generation timeline for a request:

Token  0: Allocate Block 0 (physical #42) → block_table = [42]
Token  1-15: Fill Block 0 → no new allocation needed
Token 16: Block 0 is full! Allocate Block 1 (physical #147) → block_table = [42, 147]
Token 17-31: Fill Block 1
Token 32: Block 1 is full! Allocate Block 2 (physical #8) → block_table = [42, 147, 8]
Token 33-36: Fill Block 2 (partial — 5/16 used)
Token 37: Request completes! Free blocks 42, 147, 8 → all return to free pool

Memory used: 3 blocks × 2 MB = 6 MB
Without PagedAttention: would have reserved 512 MB (4096 tokens)
Savings: 98.8%
```

### 6.3 Why This Nearly Eliminates Fragmentation

| Fragmentation Type | Without PagedAttention | With PagedAttention |
|-------------------|----------------------|-------------------|
| **Internal** | Reserve 4096 tokens, use 37. Waste = 99.1% | Last block has 11 empty slots out of 16. Waste = $\frac{11}{37+11} \approx 22\%$ of last block only, or $\frac{11}{48} \times 2\text{MB} = 1.375\text{MB}$ total waste |
| **External** | Non-contiguous free regions can't be used | Blocks are all same size and independent — any free block fits any request |

**Internal fragmentation** is bounded: at most `block_size - 1` tokens wasted per sequence (the partially filled last block). With `block_size = 16`, that's at most 15 wasted token slots — roughly $15 \times 128 \text{ KB} = 1.875 \text{ MB}$. Compare that to the old regime's potential $4096 \times 128 \text{ KB} = 512 \text{ MB}$ waste.

**External fragmentation** is **completely eliminated**. Since all blocks are the same size, any free block can satisfy any allocation request. There's no "too small to use" problem.

---

## 7. Copy-on-Write for Parallel Sampling

One of the most elegant features of PagedAttention is **Copy-on-Write (CoW)** — borrowed directly from how Unix `fork()` works.

### 7.1 The Problem: Parallel Sampling

Many LLM applications generate **multiple completions** from the same prompt:
- Beam search (keep top-k candidates)
- Parallel sampling (generate N diverse responses, pick the best)
- Best-of-N sampling for RLHF

Without sharing, if you sample 4 completions from a 1024-token prompt:

$$\text{prompt KV memory} = 4 \times 1024 \times 128 \text{ KB} = 512 \text{ MB}$$

But all 4 sequences share the **exact same prompt tokens**! Their K and V vectors for the prompt portion are identical. We're storing 4 copies of the same data.

### 7.2 The Solution: Shared Blocks with Reference Counting

With PagedAttention, all parallel sequences can **point to the same physical blocks** for their shared prompt prefix:

```
4 parallel samples from a 1024-token prompt:

Prompt tokens (shared): 1024 tokens = 64 blocks

Sequence 1 block_table: [42, 17, 93, ...(64 blocks)..., 201, 55]
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^
                         shared prompt blocks (ref_count=4) unique to seq 1

Sequence 2 block_table: [42, 17, 93, ...(64 blocks)..., 88, 12]
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^
                         same physical blocks!              unique to seq 2

Sequence 3 block_table: [42, 17, 93, ...(64 blocks)..., 177, 203]
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^
                         same physical blocks!              unique to seq 3

Sequence 4 block_table: [42, 17, 93, ...(64 blocks)..., 64, 119]
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^
                         same physical blocks!              unique to seq 4

Physical block 42:  ref_count = 4  (shared by all 4 sequences)
Physical block 17:  ref_count = 4
...
Physical block 201: ref_count = 1  (only sequence 1)
Physical block 88:  ref_count = 1  (only sequence 2)
```

### 7.3 Copy-on-Write: When Sequences Diverge

The last shared block is where things get interesting. At the boundary between prompt and generation, a block might be partially filled with prompt tokens. When a sequence needs to write a new generated token into a shared block:

```
Before CoW:
Physical Block 99 (ref_count = 4):
┌──────────────────────────────────────────┬─────────────────┐
│ Prompt tokens at positions 1008-1023     │   empty slots   │
│ (shared by all 4 sequences)             │   (4 slots)     │
└──────────────────────────────────────────┴─────────────────┘

Sequence 1 wants to write its first generated token at slot 0 of the empty section.
But it can't modify block 99 — other sequences are using it!

After CoW:
Physical Block 99 (ref_count = 3):        ← decremented
┌──────────────────────────────────────────┬─────────────────┐
│ Prompt tokens at positions 1008-1023     │   empty slots   │
│ (shared by sequences 2, 3, 4)           │                 │
└──────────────────────────────────────────┴─────────────────┘

Physical Block 201 (ref_count = 1):        ← NEW copy for seq 1
┌──────────────────────────────────────────┬────┬────────────┐
│ Prompt tokens at positions 1008-1023     │gen1│   empty    │
│ (copied from block 99)                  │    │            │
└──────────────────────────────────────────┴────┴────────────┘
          Sequence 1 can now write freely here
```

Only the **modified block** is copied — not the entire prompt KV cache.

### 7.4 Worked Example: Memory Savings

4 parallel samples from a 1024-token prompt, each generating 100 tokens:

```
WITHOUT Copy-on-Write:
  Prompt KV per sequence:    1024 × 128 KB = 128 MB
  Generation KV per sequence: 100 × 128 KB ≈ 12.5 MB
  Total per sequence:        140.5 MB
  Total for 4 sequences:     4 × 140.5 = 562 MB

WITH Copy-on-Write:
  Shared prompt KV (1 copy):   1024 × 128 KB = 128 MB
  Generation KV (4 separate):  4 × 100 × 128 KB = 50 MB
  Boundary block copies:       ≈ 3 × 2 MB = 6 MB  (3 CoW copies of boundary block)
  Total:                       128 + 50 + 6 = 184 MB

  Savings: 562 - 184 = 378 MB saved (67% reduction!)
```

With longer prompts and more samples, the savings are even more dramatic:

| Scenario | Without CoW | With CoW | Savings |
|----------|------------|----------|---------|
| 4 samples, 1024-token prompt  | 562 MB  | 184 MB  | 67%  |
| 8 samples, 2048-token prompt  | 2.19 GB | 356 MB  | 84%  |
| 16 samples, 4096-token prompt | 8.75 GB | 712 MB  | 92%  |

The savings grow as $O(N)$ — with $N$ parallel samples sharing a long prompt, you save roughly $(N-1) \times \text{prompt\_kv\_size}$.

---

## 8. Memory Savings: Before vs After (The Full Picture)

Let's put it all together with a realistic production scenario.

### 8.1 Setup

- **GPU**: NVIDIA A100 80GB
- **Model**: Llama-3-8B (BF16), weights = ~16 GB
- **Overhead**: CUDA context, activations, etc. = ~4 GB
- **Available for KV cache**: 60 GB
- **KV per token**: 128 KB
- **Block size**: 16 tokens (2 MB per block)
- **Total blocks**: 60 GB / 2 MB = 30,720 blocks

### 8.2 Scenario: 100 Concurrent Chat Requests

Assume typical chatbot workload:
- Average prompt length: 150 tokens
- Average generation length: 200 tokens
- Average total sequence: 350 tokens
- Max model context: 4096 tokens

**Without PagedAttention (pre-allocate max):**

```
Per request: 4096 tokens × 128 KB = 512 MB reserved
100 requests: 100 × 512 MB = 50 GB reserved

But actual memory doing useful work:
100 × 350 tokens × 128 KB = 4.27 GB

Utilization: 4.27 / 50 = 8.5%
Wasted: 45.73 GB (!)

Max concurrent requests possible:
60 GB / 512 MB = 117 requests (hard cap, regardless of actual usage)
```

**With PagedAttention (allocate on demand):**

```
Per request: 350 tokens → ceil(350/16) = 22 blocks × 2 MB = 44 MB
100 requests: 100 × 44 MB = 4.30 GB actually allocated

Utilization: ~100% (only last-block waste: ~100 × 15 × 128 KB ≈ 188 MB)

Remaining free: 60 - 4.30 = 55.7 GB 
Additional requests possible: 55.7 GB / 44 MB ≈ 1,265 MORE requests

Max concurrent requests at avg 350 tokens:
60 GB / 44 MB ≈ 1,395 requests (!)
```

### 8.3 The Throughput Multiplier

```
┌─────────────────────────────────────────────────────────────┐
│                   Throughput Comparison                       │
│                                                              │
│  Without PagedAttention:    ~117 max concurrent requests     │
│  With PagedAttention:       ~1,395 max concurrent requests   │
│                                                              │
│  Throughput improvement:    ~12x for this workload!          │
│                                                              │
│  (Real-world improvement varies: typically 2-5x for mixed   │
│   workloads because some requests DO use longer contexts)    │
└─────────────────────────────────────────────────────────────┘
```

The vLLM paper reports **2-4x throughput improvements** over HuggingFace Transformers and **2.2x over NVIDIA FasterTransformer** on real-world workloads. The improvement is largest when:

1. **Average sequence length << max sequence length** (more internal fragmentation eliminated)
2. **Many concurrent requests** (more sharing opportunities)
3. **Parallel sampling** (CoW kicks in)
4. **Variable-length outputs** (fixed allocation wastes more on short outputs)

### 8.4 The Bigger Picture: Why This Matters

```
Before PagedAttention (2022):              After PagedAttention (2023+):
                                           
- Serve 5-10 users per GPU                - Serve 50-200+ users per GPU
- $0.06 per 1K tokens                      - $0.01 per 1K tokens (or less)
- Long context = luxury                    - Long context = practical
- Parallel sampling = expensive            - Parallel sampling = cheap
- Memory is the bottleneck                 - Compute becomes the bottleneck
                                             (as it should be!)
```

PagedAttention didn't just improve efficiency — it **changed the economics** of LLM serving. The cost per token dropped dramatically, making applications like chatbots, code assistants, and search feasible at scale. It's one of the key innovations that enabled the explosion of LLM-powered products in 2023-2024.

---

## 9. Key Takeaways

- **The KV cache is essential but expensive.** Every generated token requires storing K and V vectors across all layers and heads. For Llama-3-8B, that's 128 KB per token — a 4096-token sequence consumes 512 MB of GPU memory per request.

- **Pre-allocation wastes 80-95% of memory.** Before PagedAttention, systems reserved KV cache for the maximum possible sequence length, even though average generations are far shorter. A 200-token generation reserved 4096 tokens — 95% waste.

- **PagedAttention borrows virtual memory from OS design.** KV cache is stored in fixed-size blocks (typically 16 tokens). A block table maps logical blocks to physical GPU memory locations, just like a page table maps virtual pages to physical frames.

- **Blocks are allocated on demand, one at a time.** New blocks are allocated only when the current block fills up. When a request completes, all its blocks return to the free pool immediately.

- **External fragmentation is eliminated.** All blocks are the same size, so any free block can serve any request. No more "Swiss cheese" memory with unusable gaps.

- **Internal fragmentation is bounded to one block.** At most `block_size - 1` token slots are wasted per sequence — roughly 1.9 MB instead of 512 MB.

- **Copy-on-Write enables cheap parallel sampling.** Multiple sequences sharing the same prompt point to the same physical blocks. Only the diverging blocks are copied, saving 67-92% memory for parallel sampling workloads.

- **Real-world impact: 2-5x throughput improvement.** By eliminating memory waste, PagedAttention allows 2-5x more concurrent requests on the same GPU, directly reducing the cost per token by the same factor.

- **The address translation is cheap.** One extra memory read per block in the attention kernel is negligible compared to the massive memory savings. The block table lookup adds minimal latency.

- **vLLM made it practical.** The open-source vLLM serving engine (built around PagedAttention) became the de facto standard for LLM serving, used in production by companies serving billions of tokens daily.
