# 03. Production Observability & Inference Golden Signals

## The Four Golden Signals of LLM Inference

1. **Time To First Token (TTFT)**:
   $$\text{TTFT} = t_{\text{first\_token}} - t_{\text{arrival}}$$
   * Evaluates: Prompt prefill compute speed + scheduler queue wait time.
   * Target SLO: $< 200\text{ ms}$ (interactive chat).

2. **Inter-Token Latency (ITL / TPOT)**:
   $$\text{ITL}_i = t_i - t_{i-1} \quad \text{for } i \in [2, N]$$
   $$\text{TPOT} = \frac{t_{\text{finish}} - t_{\text{first\_token}}}{N - 1}$$
   * Evaluates: Autoregressive decode efficiency & memory bandwidth saturation.
   * Target SLO: $< 25\text{--}35\text{ ms}$ ($> 30\text{--}40\text{ tok/s}$).

3. **Aggregate Token Throughput**:
   $$\text{Throughput} = \frac{\sum \text{Tokens Generated}}{\Delta t} \quad [\text{tokens/sec}]$$

4. **Queue Depth & KV Cache Saturation**:
   $$\text{Queue Depth} = \text{len}(\text{waiting\_requests})$$
   $$\text{KV Saturation} = \frac{\text{allocated\_blocks}}{\text{total\_blocks}}$$

## Prometheus Histogram Bucket Calibration
Default buckets (`[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]`) cause $> 35\%$ linear interpolation error in PromQL `histogram_quantile()`.

### Staff-Grade Inference Buckets:
```yaml
# ITL Buckets (Seconds): Centered on 10ms - 100ms
itl_buckets: [0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050, 0.075, 0.100, 0.200, 0.500]

# TTFT Buckets (Seconds): Spanning 50ms - 5000ms
ttft_buckets: [0.050, 0.100, 0.150, 0.200, 0.300, 0.500, 0.750, 1.000, 1.500, 2.000, 5.000]
```
