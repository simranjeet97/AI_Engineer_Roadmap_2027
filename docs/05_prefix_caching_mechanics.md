# 05. Automatic Prefix Caching (APC) & Workload Profiling

## The Physics of Prefix Caching
Prefix caching reuses physical KV cache blocks for identical prompt token prefixes:
$$\text{TTFT}_{\text{cold}} = t_{\text{queue}} + t_{\text{prefill}}(S) \approx 180\text{ ms}$$
$$\text{TTFT}_{\text{warm}} = t_{\text{queue}} + t_{\text{prefill}}(\Delta S) \approx 15\text{ ms} \quad (\mathbf{12\times \text{ faster}})$$

## Enterprise High-Leverage Topologies:
1. **Multi-Agent Systems**: 4k-token system prompt + tool schemas shared across multi-turn ReAct loops.
2. **Document RAG**: 16k context document loaded once; shared across 10+ user questions ($89.8\%$ compute reduction).
3. **Multi-Turn Chat**: Dialogue history accumulation.

## The Top 5 Cache-Killer Antipatterns:
1. **Timestamp at Position 0**: Prepending `Current Time: 10:20:01` invalidates token 0, destroying downstream cache hits ($0\%$ hit rate). Fix: Move timestamps to the very end of the prompt.
2. **User ID / Session ID in Header**: Place dynamic session tokens at the end.
3. **Randomized Few-Shot Shuffling**: Keep few-shot ordering deterministic.
4. **Template Whitespace Jitter**: Ensure Jinja templates produce deterministic spaces/newlines.
5. **Cache Thrashing**: Unique shared prefixes exceeding GPU VRAM trigger constant LRU eviction loops.
