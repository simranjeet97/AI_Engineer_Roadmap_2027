# 02. Comparative Scheduling Analysis: vLLM vs SGLang

## Architectural Comparison

| Dimension | vLLM (BlockManager) | SGLang (RadixCache) |
| :--- | :--- | :--- |
| **KV Cache Data Structure** | Flat Block Table array per request | Dynamic LRU Radix Tree (Trie) |
| **Prefix Matching** | Exact block-aligned hash matching | Arbitrary token sub-sequence matching |
| **Multi-Turn Handling** | Per-sequence tracking; re-allocates on miss | Hierarchical parent-child node traversal |
| **Eviction Policy** | Watermark-based preemptive swapping/recompute | LRU eviction on unreferenced tree leaf nodes |
| **Workload Sweet Spot** | High-concurrency independent requests | Agentic loops, RAG, multi-turn branched chat |

## vLLM Scheduler Step Invariant
In `vllm/core/scheduler.py`:
1. Check `waiting` queue: If `block_manager.can_allocate(req)`, move to `running`.
2. For each `running` request: Check if appending 1 token requires a new block.
3. If memory exhausted: Invoke `_preempt(victim)`:
   * **Recompute**: Clear block table, push back to front of `waiting`.
   * **Swap**: Transfer blocks over PCIe to host CPU RAM.

## SGLang RadixCache Step Invariant
In `sglang/srt/managers/router/radix_cache.py`:
1. `match_prefix(tokens)`: Traverses Radix Tree from root, returning matched physical blocks and remaining unmatched tokens.
2. Only unmatched tokens are scheduled for prefill.
3. Newly computed tokens are appended as child tree nodes with `ref_count = 1`.
4. When memory exceeds threshold, `_evict_lru()` reclaims physical blocks from unreferenced leaves (`ref_count == 0`).
