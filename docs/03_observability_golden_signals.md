# Production Observability & The Golden Signals of LLM Inference

## Why Observability Matters

You've deployed your LLM. Users are hitting the endpoint. But how do you know if it's working well?

Let me tell you a story that plays out at every company running inference at scale. It's a Tuesday afternoon. Your chat product is serving 200 requests per second on a cluster of 8 H100s. Your dashboard shows **average latency: 180ms**. Everything looks green. You go to lunch.

What you didn't see: at 11:47 AM, a customer uploaded a 12,000-token document for summarization. Then another. Then a batch job from an internal team kicked in, sending 50 long-context requests simultaneously. Here's what happened under the hood:

```
Timeline of a Silent Outage
════════════════════════════════════════════════════════════════
11:45 AM  Queue depth: 2       P99 TTFT: 200ms     ← Normal
11:47 AM  Queue depth: 12      P99 TTFT: 800ms     ← Starting to slip
11:52 AM  Queue depth: 45      P99 TTFT: 3,200ms   ← Users noticing delays
11:58 AM  Queue depth: 120     P99 TTFT: 8,100ms   ← Users leaving
12:05 PM  Queue depth: 340     P99 TTFT: 22,000ms  ← Complete meltdown

Average latency at 12:05 PM?  Still 420ms.
Why? 95% of short requests still complete fast, masking the tail.
```

The **average** looked fine the entire time. But the P99 — the latency experienced by the worst 1% of your users — went from 200ms to 22 seconds. Those users waited, got frustrated, and left. Your support queue filled up before your monitoring ever fired an alert.

This is why observability matters. Not "logging" — *observability*. The ability to understand the internal state of your system by examining its external outputs. For LLM inference, this means tracking the right signals, at the right percentiles, with the right alerting thresholds.

---

## The Google SRE Golden Signals

In 2016, Google published the *Site Reliability Engineering* book, which introduced the concept of **four golden signals** — the minimum set of metrics you need to monitor any distributed system:

| Signal | What It Measures | The Question It Answers |
|---|---|---|
| **Latency** | Time to serve a request | "How fast is my system?" |
| **Traffic** | Demand on the system | "How much work is it doing?" |
| **Errors** | Rate of failed requests | "How often does it break?" |
| **Saturation** | How "full" the system is | "How close to capacity am I?" |

These four signals work for web servers, databases, and microservices. But LLM inference has unique characteristics that require us to **specialize** these signals. Here's the mapping:

```
Google SRE Signal        LLM Inference Equivalent
═══════════════════      ══════════════════════════════════════════
Latency           ──►    TTFT (Time to First Token)
                         ITL / TPOT (Inter-Token Latency)
Traffic           ──►    Throughput (tokens/second, requests/second)
Errors            ──►    HTTP errors, generation failures, timeouts
Saturation        ──►    Queue depth, KV cache usage %, GPU util
```

Notice that **latency splits into two signals** for LLMs. That's because streaming changes everything — users don't just care about total response time, they care about how quickly the *first* token appears (perceived responsiveness) and how smoothly subsequent tokens stream (perceived fluency). Let's dive deep into each.

---

## Signal 1: TTFT (Time to First Token)

### What It Measures

TTFT is the elapsed time from when your server **receives** a request to when the **first token** is streamed back to the client.

```
User sends request                    First token arrives
       │                                      │
       ▼                                      ▼
       ├──── Queue Wait ────┼──── Prefill ────┤
       │                    │                  │
       │  Waiting for GPU   │  Processing the  │
       │  to be available   │  entire prompt   │
       │                    │  in parallel     │
       └────────────────────┴──────────────────┘
                    TTFT (total)
```

Think of it like a restaurant. TTFT is how long you wait before the first appetizer arrives — the time spent in the lobby (queue) plus the time the kitchen takes to prepare the first dish (prefill).

### Why It Matters

Users perceive TTFT as "thinking time." When you type a question to ChatGPT and see that blinking cursor before any text appears — that pause *is* the TTFT. Research shows:

- **< 100ms**: Feels instant, like the system "already knew"
- **100–300ms**: Noticeable but acceptable pause
- **300ms–1s**: User starts wondering "is it working?"
- **> 1s**: User considers refreshing the page
- **> 3s**: Significant fraction of users abandon

### The TTFT Formula

TTFT breaks down into two additive components:

$$TTFT \approx \text{queue delay} + \frac{\text{prompt tokens}}{\text{prefill throughput}}$$

Where:
- $\text{queue delay}$ = time spent waiting for a slot in the running batch
- $\text{prompt tokens}$ = number of tokens in the input prompt
- $\text{prefill throughput}$ = how fast the GPU processes prompt tokens (tok/s)

The prefill phase is **compute-bound** — the GPU processes all prompt tokens in parallel through the transformer layers, computing attention across the full sequence. This is the heavy matrix-multiplication work.

### Worked Example

Let's calculate TTFT for a typical request:

**Setup**: Llama-3-8B on a single H100, no queue contention

| Parameter | Value |
|---|---|
| Prompt length | 2,048 tokens |
| Prefill throughput (H100, 8B model) | ~40,000 tok/s |
| Queue delay (empty queue) | ~0 ms |

$$\text{prefill time} = \frac{2048}{40000} = 0.0512 \text{ seconds} = 51.2 \text{ ms}$$

$$TTFT = 0 + 51.2 = 51.2 \text{ ms}$$

That's excellent! But watch what happens as prompt length grows:

| Prompt Tokens | Prefill Time | Queue = 0ms | Queue = 200ms |
|---|---|---|---|
| 512 | 12.8 ms | 12.8 ms | 212.8 ms |
| 2,048 | 51.2 ms | 51.2 ms | 251.2 ms |
| 8,192 | 204.8 ms | 204.8 ms | 404.8 ms |
| 32,768 | 819.2 ms | 819.2 ms | 1,019.2 ms |
| 131,072 | 3,276.8 ms | 3,276.8 ms | 3,476.8 ms |

**Key insight**: For short prompts, queue delay dominates TTFT. For long prompts, prefill time dominates. Your optimization strategy depends on which regime you're in.

### SLO Thresholds (Typical Chat Application)

| Percentile | Threshold | Meaning |
|---|---|---|
| P50 | < 100 ms | Half of requests feel instant |
| P90 | < 300 ms | 90% of requests feel responsive |
| P99 | < 1,000 ms | Even the worst 1% stay under 1 second |

### What Causes TTFT Spikes

1. **Queue buildup**: Too many concurrent requests → requests wait in line
2. **Long prompts**: A 128K-token RAG prompt takes 100× longer to prefill than a 1K prompt
3. **No prefix caching**: Repeated system prompts get re-processed every time instead of being cached
4. **Prefill interference**: On systems without chunked prefill, a long prefill blocks all decode steps
5. **Model loading / cold start**: First request after scaling up hits model load latency

### Prometheus Metrics & PromQL

vLLM exposes TTFT as a histogram metric:

```
vllm:time_to_first_token_seconds_bucket
vllm:time_to_first_token_seconds_count
vllm:time_to_first_token_seconds_sum
```

**PromQL for P99 TTFT** (the single most important query for your dashboard):

```promql
histogram_quantile(0.99,
  rate(vllm:time_to_first_token_seconds_bucket[5m])
)
```

**PromQL for P50 TTFT**:

```promql
histogram_quantile(0.50,
  rate(vllm:time_to_first_token_seconds_bucket[5m])
)
```

**Alert rule** — fire if P99 TTFT exceeds 1 second for 2 minutes:

```yaml
- alert: HighP99TTFT
  expr: |
    histogram_quantile(0.99,
      rate(vllm:time_to_first_token_seconds_bucket[5m])
    ) > 1.0
  for: 2m
  labels:
    severity: warning
  annotations:
    summary: "P99 TTFT is {{ $value }}s (threshold: 1.0s)"
```

---

## Signal 2: ITL / TPOT (Inter-Token Latency / Time Per Output Token)

### What It Measures

ITL (Inter-Token Latency), also called TPOT (Time Per Output Token), measures the time between consecutive tokens being generated during the decode phase.

```
First token    Second token   Third token    Fourth token
    │               │              │              │
    ▼               ▼              ▼              ▼
    ├──── ITL ──────┼──── ITL ────┼──── ITL ─────┤
    │    ~30ms      │    ~30ms    │    ~30ms      │
```

If TTFT is how long you wait for the appetizer, ITL is how quickly the waiter brings each subsequent course. A low ITL means text streams smoothly — like someone typing fast. A high or variable ITL means text appears in stuttering bursts.

### Why It Matters

ITL directly determines the **perceived typing speed** of your LLM:

$$\text{tokens per second}_{per\ stream} = \frac{1000}{ITL_{ms}}$$

| ITL | Tokens/sec | User Experience |
|---|---|---|
| 20 ms | 50 tok/s | Impressively fast, hard to read along |
| 30 ms | 33 tok/s | Smooth, natural reading pace |
| 50 ms | 20 tok/s | Acceptable, slightly deliberate |
| 100 ms | 10 tok/s | Noticeably slow, feels laggy |
| 200 ms | 5 tok/s | Painfully slow, users get impatient |

The sweet spot is **20–40ms ITL** (25–50 tok/s), which matches comfortable human reading speed.

### The Memory-Bandwidth Ceiling

Here's a fundamental insight about decode performance: **each decode step must read the entire model's weights from GPU memory**. The GPU processes one token per sequence per decode step, meaning the arithmetic intensity is extremely low — it's a memory-bandwidth-bound operation.

This gives us a hard physical floor on ITL:

$$ITL_{min} = \frac{\text{model weight bytes}}{\text{memory bandwidth}}$$

This is the absolute minimum — you literally cannot read the weights faster than the memory bus allows.

### Worked Example: Llama-3-70B on H100

| Parameter | Value |
|---|---|
| Model parameters | 70 billion |
| Precision | BF16 (2 bytes per parameter) |
| Model weight size | $70 \times 10^9 \times 2 = 140$ GB |
| H100 HBM3 bandwidth | 3,350 GB/s |

$$ITL_{min} = \frac{140 \text{ GB}}{3350 \text{ GB/s}} = 0.0418 \text{ s} = 41.8 \text{ ms}$$

$$\text{max tokens per second} = \frac{1}{0.0418} \approx 23.9 \text{ tok/s per stream}$$

That's the theoretical maximum for a single stream — and it assumes:
- Perfect memory access patterns (no bank conflicts)
- Zero overhead for attention computation on the KV cache
- No interference from other operations

In practice, you'll see **18–22 tok/s** for a 70B BF16 model on a single H100. This is why quantization matters so much — a 70B model in INT4 (4-bit) is only 35 GB, halving the minimum ITL to ~10.4ms!

| Model | Precision | Weight Size | ITL Floor (H100) | Max tok/s |
|---|---|---|---|---|
| Llama-3-8B | BF16 | 16 GB | 4.8 ms | 209 tok/s |
| Llama-3-8B | INT4 (AWQ) | 4 GB | 1.2 ms | 833 tok/s |
| Llama-3-70B | BF16 | 140 GB | 41.8 ms | 23.9 tok/s |
| Llama-3-70B | INT4 (AWQ) | 35 GB | 10.4 ms | 96 tok/s |

### What Causes ITL Spikes

The #1 cause of ITL variability is **prefill interference**. In a naive continuous batching system, when a new request arrives and starts its prefill phase, all ongoing decode steps must wait:

```
Without Chunked Prefill:
─────────────────────────────────────────────────
Decode │████│         │████│████│████│
       │    │         │    │    │    │
Prefill│    │█████████│    │    │    │  ← Long prefill blocks decodes
─────────────────────────────────────────────────
             ↑ 200ms gap — ITL spike!

With Chunked Prefill:
─────────────────────────────────────────────────
Decode │████│████│████│████│████│████│
       │    │    │    │    │    │    │
Prefill│    │██  │██  │██  │    │    │  ← Prefill split into chunks
─────────────────────────────────────────────────
         Consistent ~30ms ITL throughout
```

**Chunked prefill** (enabled by default in modern vLLM) breaks long prefills into smaller chunks that interleave with decode steps, maintaining consistent ITL for all active streams.

### SLO Thresholds

| Percentile | Threshold | Rationale |
|---|---|---|
| P50 | < 30 ms | Smooth streaming experience |
| P90 | < 45 ms | Minor variation tolerated |
| P99 | < 60 ms | Even worst case doesn't stutter |

### Prometheus Metrics & PromQL

```
vllm:inter_token_latency_seconds_bucket
vllm:inter_token_latency_seconds_count
vllm:inter_token_latency_seconds_sum
```

**PromQL for P99 ITL**:

```promql
histogram_quantile(0.99,
  rate(vllm:inter_token_latency_seconds_bucket[5m])
)
```

---

## Signal 3: Throughput (Tokens per Second)

### What It Measures

Throughput measures the total productive work your inference system performs. But there's a crucial distinction: **system throughput** vs **per-request throughput**.

```
System Throughput (what the GPU produces in total):
═══════════════════════════════════════════════════
  Stream 1: ──tok──tok──tok──tok──tok──
  Stream 2: ──tok──tok──tok──tok──tok──
  Stream 3: ──tok──tok──tok──tok──tok──
  Stream 4: ──tok──tok──tok──tok──tok──
                                        
  System: 4 × 24 tok/s = 96 tok/s total
  Per-stream: still 24 tok/s each
```

Think of it like a highway. System throughput is the total number of cars passing through per hour (total lane capacity). Per-request throughput is how fast *your* car is going (speed per lane). Adding more lanes (batching more requests) increases total throughput without speeding up individual cars.

### Two Flavors of Throughput

| Metric | What It Counts | Why It Matters |
|---|---|---|
| **Prefill throughput** | Input tokens processed per second | Determines TTFT for new requests |
| **Decode (generation) throughput** | Output tokens generated per second | Determines streaming speed and system capacity |

### How Batching Increases System Throughput

This is one of the most beautiful properties of LLM decode. Remember, each decode step is memory-bandwidth-bound — you must read the model weights once regardless of batch size. So processing 1 token or 64 tokens costs nearly the same memory-read time:

$$\text{system decode throughput} \approx \frac{\text{batch size}}{ITL_{single}}$$

As long as you're memory-bandwidth bound (not compute bound), **adding more sequences to the batch is nearly free**.

### Worked Example: Llama-3-8B on Single H100

| Batch Size | Time per Decode Step | System tok/s | Per-Stream tok/s |
|---|---|---|---|
| 1 | ~4.8 ms | 209 | 209 |
| 8 | ~5.0 ms | 1,600 | 200 |
| 32 | ~6.5 ms | 4,923 | 154 |
| 64 | ~9.0 ms | 7,111 | 111 |
| 128 | ~14.0 ms | 9,143 | 71 |
| 256 | ~25.0 ms | 10,240 | 40 |

Notice the pattern: system throughput increases with batch size, but per-stream throughput *decreases* slightly as the operation transitions from memory-bound to compute-bound. There's a sweet spot — typically around batch size 64–128 for an 8B model on H100 — where you get maximum system throughput without degrading per-stream latency below acceptable SLOs.

For a 70B model, these batch sizes would be smaller because the KV cache per sequence is much larger, limiting how many concurrent sequences fit in GPU memory.

### Prometheus Metrics & PromQL

**Decode throughput** (total output tokens generated per second across all streams):

```promql
rate(vllm:generation_tokens_total[1m])
```

**Prefill throughput** (total input tokens processed per second):

```promql
rate(vllm:prompt_tokens_total[1m])
```

**Requests completed per second**:

```promql
rate(vllm:request_success_total[1m])
```

**Average output tokens per request** (useful for capacity planning):

```promql
rate(vllm:generation_tokens_total[5m]) / rate(vllm:request_success_total[5m])
```

---

## Signal 4: Saturation (Queue Depth & KV Cache Usage)

### What It Measures

Saturation answers the most critical operational question: **how close am I to falling off a cliff?**

Two metrics define saturation for LLM inference:

1. **Queue depth**: Number of requests waiting to be scheduled onto the GPU
2. **KV cache usage**: Percentage of GPU memory block pool occupied by active KV caches

```
GPU Memory Layout:
┌─────────────────────────────────────────────────┐
│  Model Weights (fixed)          │ 16 GB         │
├─────────────────────────────────┤               │
│  KV Cache Block Pool            │ 55 GB         │  80 GB total
│  ┌───────────────────────────┐  │               │  (H100)
│  │ █████████████████░░░░░░░░ │  │ 68% used      │
│  └───────────────────────────┘  │               │
├─────────────────────────────────┤               │
│  Activations + Overhead         │ 9 GB          │
└─────────────────────────────────────────────────┘

Queue: [req_12] [req_13] [req_14] [req_15]  ← 4 requests waiting
```

### The Cliff Point

LLM inference systems have a **non-linear** degradation pattern. They work beautifully up to about 85–90% KV cache usage, and then performance falls off a cliff:

```
Latency
  │
  │                                          ╱
  │                                        ╱
  │                                      ╱
  │                                   ╱
  │                                ╱   ← Cliff point (~90%)
  │              ─────────────╱
  │  ───────────╱
  │─╱
  └────────────────────────────────────── KV Cache %
  0%    20%    40%    60%   80%  90% 100%
```

Why? When the KV cache is full, the scheduler must **preempt** (evict) sequences to make room for new ones. Preempted sequences lose their KV cache and must re-prefill when rescheduled — creating a cascade of wasted compute:

1. New request arrives, KV cache is full
2. Scheduler evicts Request A's KV cache to make room
3. New request starts prefill
4. Request A gets rescheduled — must redo its entire prefill
5. Now *another* request gets evicted to make room for A's re-prefill
6. Repeat → **thrashing**

### Little's Law: The Queue Theory Connection

One of the most powerful tools for capacity planning is **Little's Law**, a theorem from queueing theory that holds under very general conditions:

$$L = \lambda \times W$$

Where:
- $L$ = average number of requests in the system (queue + being served)
- $\lambda$ = arrival rate (requests per second)
- $W$ = average time each request spends in the system

This is deceptively simple but profoundly useful.

### Worked Example: Capacity Planning with Little's Law

**Scenario**: Your inference server can complete 50 requests per second (service rate $\mu = 50$ req/s). Each request takes an average of $W = 2$ seconds to complete.

**Steady state queue length**:

$$L = \lambda \times W$$

| Arrival Rate ($\lambda$) | Avg Time ($W$) | Avg Requests in System ($L$) | Status |
|---|---|---|---|
| 30 req/s | 2.0 s | 60 | ✅ Comfortable headroom |
| 40 req/s | 2.0 s | 80 | ✅ Moderate load |
| 48 req/s | 2.1 s | 101 | ⚠️ Approaching capacity |
| 50 req/s | 2.5 s | 125 | 🔴 At capacity, latency rising |
| 55 req/s | 4.0 s | 220 | 💀 Overloaded, queue growing |

**Critical insight**: When $\lambda > \mu$ (arrival rate exceeds service rate), the queue grows **without bound**. There is no steady state. Your system will eventually crash. This is why saturation monitoring is essential — you need to detect the approach to $\lambda = \mu$ *before* it happens.

Here's the math for why this is catastrophic. If $\lambda = 60$ req/s but $\mu = 50$ req/s:

$$\text{queue growth rate} = \lambda - \mu = 60 - 50 = 10 \text{ req/s}$$

After 1 minute: queue = 600 requests. After 5 minutes: queue = 3,000 requests. Each request's wait time grows linearly, and your P99 latency rockets into the stratosphere.

### Prometheus Metrics & PromQL

**Queue depth** (requests waiting):

```promql
vllm:num_requests_waiting
```

**KV cache usage factor** (0.0 to 1.0):

```promql
vllm:gpu_cache_usage_factor
```

**Running requests** (actively being processed):

```promql
vllm:num_requests_running
```

### Critical Alert Rules

```yaml
# Alert when queue depth stays above 20 for 1 minute
- alert: HighQueueDepth
  expr: vllm:num_requests_waiting > 20
  for: 1m
  labels:
    severity: warning
  annotations:
    summary: "Queue depth is {{ $value }} (threshold: 20)"

# Alert when KV cache exceeds 85% for 2 minutes
- alert: KVCacheNearFull
  expr: vllm:gpu_cache_usage_factor > 0.85
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "KV cache usage is {{ $value | humanizePercentage }}"

# Alert when queue is growing (rate of change is positive for 3 minutes)
- alert: QueueGrowing
  expr: deriv(vllm:num_requests_waiting[5m]) > 2
  for: 3m
  labels:
    severity: critical
  annotations:
    summary: "Queue growing at {{ $value }} req/s — likely overloaded"
```

---

## Setting Up Prometheus + Grafana

### Step 1: Enable Metrics in vLLM

vLLM serves Prometheus metrics on its HTTP port by default. When you start vLLM:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3-8B-Instruct \
    --port 8000 \
    --enable-chunked-prefill
```

Metrics are available at `http://localhost:8000/metrics`.

### Step 2: Prometheus Scrape Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s       # Default for most targets
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'vllm-inference'
    scrape_interval: 5s       # Faster scraping for inference servers
    metrics_path: '/metrics'
    static_configs:
      - targets:
        - 'vllm-server-1:8000'
        - 'vllm-server-2:8000'
        - 'vllm-server-3:8000'
        - 'vllm-server-4:8000'
        labels:
          cluster: 'production'
          model: 'llama-3-8b'
```

> **Why 5-second scrape interval?** Inference workloads can spike and recover within seconds. A 15-second default interval might miss brief queue buildups entirely. For your inference fleet, 5s gives you the resolution to catch problems early without excessive overhead.

### Step 3: Key PromQL Expressions Dashboard

Here's every query you need for a comprehensive LLM inference dashboard:

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Inference Dashboard                      │
├──────────────────────────┬──────────────────────────────────────┤
│  P99 TTFT (seconds)      │  P99 ITL (seconds)                  │
│                          │                                      │
│  histogram_quantile(     │  histogram_quantile(                 │
│    0.99,                 │    0.99,                              │
│    rate(vllm:time_to_    │    rate(vllm:inter_token_            │
│    first_token_seconds_  │    latency_seconds_                   │
│    bucket[5m])           │    bucket[5m])                        │
│  )                       │  )                                    │
├──────────────────────────┼──────────────────────────────────────┤
│  Throughput (tok/s)      │  Queue Depth                         │
│                          │                                      │
│  rate(vllm:generation_   │  vllm:num_requests_waiting           │
│  tokens_total[1m])       │                                      │
├──────────────────────────┼──────────────────────────────────────┤
│  KV Cache Usage (%)      │  Request Rate (req/s)                │
│                          │                                      │
│  vllm:gpu_cache_usage_   │  rate(vllm:request_                  │
│  factor * 100            │  success_total[1m])                   │
├──────────────────────────┼──────────────────────────────────────┤
│  Error Rate (%)          │  Active Requests                     │
│                          │                                      │
│  rate(vllm:request_      │  vllm:num_requests_running           │
│  failure_total[1m]) /    │                                      │
│  rate(vllm:request_      │                                      │
│  total[1m]) * 100        │                                      │
└──────────────────────────┴──────────────────────────────────────┘
```

### Step 4: Alerting Configuration

For production systems, layer your alerts by severity:

```yaml
# alerting_rules.yml
groups:
  - name: llm_inference_alerts
    rules:
      # ─── WARNING TIER (Slack notification) ───
      - alert: TTFT_P99_Elevated
        expr: histogram_quantile(0.99, rate(vllm:time_to_first_token_seconds_bucket[5m])) > 0.5
        for: 3m
        labels:
          severity: warning
          team: ml-platform
        annotations:
          summary: "P99 TTFT elevated to {{ $value }}s on {{ $labels.instance }}"
          runbook: "https://wiki.internal/runbooks/ttft-elevated"

      # ─── CRITICAL TIER (PagerDuty) ───
      - alert: TTFT_P99_Critical
        expr: histogram_quantile(0.99, rate(vllm:time_to_first_token_seconds_bucket[5m])) > 2.0
        for: 1m
        labels:
          severity: critical
          team: ml-platform
        annotations:
          summary: "P99 TTFT critical at {{ $value }}s — user impact likely"

      - alert: KVCache_Thrashing
        expr: vllm:gpu_cache_usage_factor > 0.92 and deriv(vllm:num_requests_waiting[5m]) > 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "KV cache at {{ $value | humanizePercentage }}, queue growing — preemption thrashing likely"
```

**Alertmanager routing** (PagerDuty for critical, Slack for warning):

```yaml
# alertmanager.yml
route:
  group_by: ['alertname', 'cluster']
  receiver: 'slack-warnings'
  routes:
    - match:
        severity: critical
      receiver: 'pagerduty-critical'

receivers:
  - name: 'slack-warnings'
    slack_configs:
      - channel: '#ml-inference-alerts'
        send_resolved: true

  - name: 'pagerduty-critical'
    pagerduty_configs:
      - service_key: '<your-pagerduty-key>'
```

---

## The Art of SLO Design

### Start With User Experience

SLOs (Service Level Objectives) should flow from **user needs**, not system capabilities. Ask:

- "What latency makes our chat product feel responsive?" → TTFT < 500ms at P99
- "What streaming speed feels natural?" → ITL < 50ms at P99
- "What error rate is acceptable?" → < 0.1% of requests fail
- "How many requests must we handle?" → 500 req/s at peak

### Measure Baseline Percentiles First

Before setting SLOs, measure your actual performance under realistic load for at least a week. You need to understand your distribution:

```
TTFT Distribution (10,000 requests):
────────────────────────────────────
P50:   82 ms    ← Half your requests
P75:   145 ms
P90:   310 ms   ← 90% of requests
P95:   520 ms
P99:   890 ms   ← 99% of requests
P99.9: 2,100 ms ← One in a thousand
Max:   8,500 ms ← Single worst request
```

### Set SLOs at P99, Not Average

**Never** set SLOs on averages. The opening story of this doc showed why — averages hide tail latency. Here's the rule:

| Metric | SLO Percentile | Example Target |
|---|---|---|
| TTFT | P99 | < 500 ms |
| ITL | P99 | < 60 ms |
| Error rate | Total | < 0.1% |
| Throughput | Sustained | > 400 req/s |

### The Error Budget Concept

An error budget makes SLOs actionable. If your P99 TTFT SLO is 500ms, that means **1% of requests are allowed to exceed 500ms**. That 1% is your error budget.

$$\text{error budget} = 1 - \text{SLO target percentile}$$

For a P99 SLO: error budget = 1%

Over a 30-day window with 10 million requests:

$$\text{allowed violations} = 10{,}000{,}000 \times 0.01 = 100{,}000 \text{ requests}$$

You can track remaining budget in real time:

```promql
# Error budget remaining (fraction)
1 - (
  (
    sum(rate(vllm:time_to_first_token_seconds_bucket{le="0.5"}[30d]))
    /
    sum(rate(vllm:time_to_first_token_seconds_count[30d]))
  )
) / 0.01
```

**Error budget policy**: When you've consumed more than 80% of your monthly error budget, freeze non-critical deployments and focus on reliability. This turns SLOs from aspirational numbers into concrete engineering decisions.

```
Error Budget Burn Rate:
══════════════════════════════════════════════════
Day 1-5:    ████░░░░░░░░░░░░░░░░  20% consumed  ✅ Normal
Day 5-10:   ████████░░░░░░░░░░░░  40% consumed  ✅ Normal
Day 10-15:  ████████████░░░░░░░░  60% consumed  ⚠️ Watch
Day 15-18:  ████████████████░░░░  80% consumed  🔴 Freeze deploys
Day 18-20:  ██████████████████░░  90% consumed  🔴 Incident mode
Day 20-30:  ░░░░░░░░░░░░░░░░░░░░  Must stay flat
```

---

## Key Takeaways

1. **Monitor P99, not averages.** Average latency hides the pain of your worst-off users. A system with 50ms average and 8-second P99 is broken — the average just doesn't show it.

2. **LLM inference has TWO latency signals.** TTFT (time to first token) measures responsiveness; ITL (inter-token latency) measures streaming smoothness. Both matter independently.

3. **TTFT = queue_delay + prefill_time.** For short prompts, the queue dominates. For long prompts (RAG, document processing), prefill dominates. Know which regime you're optimizing for.

4. **ITL has a physics floor.** You cannot decode faster than $\frac{\text{model bytes}}{\text{memory bandwidth}}$. For Llama-3-70B BF16 on H100, that's 41.8ms minimum — about 24 tok/s per stream. Quantization is the lever to break through this floor.

5. **Saturation is the leading indicator.** Queue depth and KV cache usage predict problems *before* latency degrades. Alert on saturation early — once TTFT has spiked, users are already suffering.

6. **Little's Law is your capacity planning friend.** $L = \lambda \times W$. If arrival rate exceeds service rate, the queue grows without bound. No amount of queueing theory tricks can save you — you need more GPUs or fewer requests.

7. **The KV cache cliff is real.** Performance degrades non-linearly above ~90% KV cache usage due to preemption cascading. Treat 85% as your red line, not 100%.

8. **Set SLOs from user experience, not system capability.** Your users don't care about your GPU utilization. They care about how fast the first token appears and how smoothly text streams.

9. **Error budgets make SLOs actionable.** A P99 SLO at 500ms means 1% of requests can violate. Track this budget in real time, and freeze deployments when it's nearly spent.

10. **Five-second scrape intervals for inference.** The default 15-second Prometheus interval is too coarse for inference workloads. Queue depth can spike and thrash within seconds — you need the resolution to see it.
