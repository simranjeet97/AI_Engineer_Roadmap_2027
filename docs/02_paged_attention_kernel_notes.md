# 02. PagedAttention Kernel Mechanics & Virtual Memory

## The Fragmentation Dilemma
Before PagedAttention, KV caches were allocated contiguously for `max_model_len` (e.g. 4096 tokens).
* **Internal Fragmentation**: Requests generating 150 tokens wasted $>90\%$ of allocated space.
* **External Fragmentation**: Memory gaps between terminating requests could not be merged.
Total memory waste in naive serving: **60% to 80%**.

## PagedAttention Address Resolution
PagedAttention adapts OS virtual memory paging to GPU HBM:
* Logical sequence space is divided into logical blocks of size $B_s$ (typically 16 or 32 tokens).
* Physical Key and Value tensors are stored in non-contiguous physical blocks in a pre-allocated pool.

### Address Translation:
$$\text{Logical Block Number} = \left\lfloor \frac{\text{token\_idx}}{B_s} \right\rfloor$$
$$\text{Block Offset} = \text{token\_idx} \pmod{B_s}$$
$$\text{Physical Block ID} = \text{block\_table}[\text{Logical Block Number}]$$

In the CUDA kernel (`vllm/csrc/attention/attention_kernels.cu`):
```cuda
const int64_t physical_block_number = block_table[logical_block_idx];
const int64_t physical_offset = physical_block_number * block_stride + block_offset * token_stride;
const scalar_t* key_ptr = k_cache + physical_offset;
```

## Copy-on-Write (CoW)
When parallel sampling generates $N$ completions from a single prompt:
1. All $N$ child sequences share the physical block pointers of the prompt tokens.
2. The `ref_count` of each physical prompt block is incremented.
3. When sequence $i$ generates a new token in a partially filled block, the engine forks a private physical block only for the modified block:
```python
if pool.ref_counts[physical_block] > 1:
    new_block = pool.allocate()
    pool.k_pool[new_block] = pool.k_pool[physical_block].clone()
    pool.free(physical_block)
    block_table[logical_idx] = new_block
```
