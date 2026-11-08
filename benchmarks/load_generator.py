#!/usr/bin/env python3
"""
Apex-Infer Staff-Grade AsyncIO Load Generator (1 to 1,000+ Concurrency)
Simulates Poisson arrival traffic to OpenAI-compatible vLLM/SGLang streaming endpoints.
Accurately records TTFT (Time to First Token) and ITL (Inter-Token Latency) per stream.
"""

import argparse
import asyncio
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import List, Optional

try:
    import aiohttp
except ImportError:
    aiohttp = None


@dataclass
class StreamMetrics:
    request_id: str
    prompt_tokens: int
    completion_tokens: int = 0
    start_time: float = 0.0
    ttft_s: float = 0.0
    total_latency_s: float = 0.0
    token_timestamps: List[float] = field(default_factory=list)
    itl_list_ms: List[float] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class BenchmarkSummary:
    total_requests: int
    successful_requests: int
    failed_requests: int
    duration_s: float
    total_tokens_generated: int
    tokens_per_sec: float
    p50_ttft_ms: float
    p90_ttft_ms: float
    p95_ttft_ms: float
    p99_ttft_ms: float
    p50_itl_ms: float
    p90_itl_ms: float
    p99_itl_ms: float
    streams: List[dict] = field(default_factory=list)


def calculate_percentile(data: List[float], percentile: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    d0 = sorted_data[int(f)] * (c - k)
    d1 = sorted_data[int(c)] * (k - f)
    return d0 + d1


async def stream_request(
    session: "aiohttp.ClientSession",
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    request_id: str,
) -> StreamMetrics:
    metrics = StreamMetrics(request_id=request_id, prompt_tokens=len(prompt.split()))
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }

    metrics.start_time = time.perf_counter()
    first_chunk_received = False
    last_chunk_time = metrics.start_time

    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=180)) as resp:
            if resp.status != 200:
                metrics.error = f"HTTP {resp.status}: {await resp.text()}"
                return metrics

            async for line in resp.content:
                now = time.perf_counter()
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    break

                if not first_chunk_received:
                    metrics.ttft_s = now - metrics.start_time
                    first_chunk_received = True
                else:
                    metrics.itl_list_ms.append((now - last_chunk_time) * 1000.0)

                metrics.token_timestamps.append(now)
                last_chunk_time = now
                metrics.completion_tokens += 1

    except Exception as exc:
        metrics.error = str(exc)
    finally:
        metrics.total_latency_s = time.perf_counter() - metrics.start_time

    return metrics


async def worker_loop(
    worker_id: int,
    queue: asyncio.Queue,
    session: "aiohttp.ClientSession",
    url: str,
    model: str,
    max_tokens: int,
    results: List[StreamMetrics],
):
    while True:
        task = await queue.get()
        if task is None:
            queue.task_done()
            break
        req_id, prompt = task
        metric = await stream_request(session, url, model, prompt, max_tokens, req_id)
        results.append(metric)
        queue.task_done()


async def run_benchmark(
    host: str,
    port: int,
    model: str,
    concurrency: int,
    total_requests: int,
    prompt_len: int,
    max_tokens: int,
    arrival_rate: Optional[float],
) -> BenchmarkSummary:
    url = f"http://{host}:{port}/v1/chat/completions"
    test_prompt = ("Analyze the hardware roofline operational intensity ceiling " * (prompt_len // 8))[:prompt_len * 4]

    if aiohttp is None:
        raise RuntimeError("aiohttp is required to run the load generator. Run: pip install aiohttp")

    conn = aiohttp.TCPConnector(limit=concurrency + 50, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=conn) as session:
        queue: asyncio.Queue = asyncio.Queue()
        results: List[StreamMetrics] = []

        # Spawn workers
        workers = [
            asyncio.create_task(
                worker_loop(i, queue, session, url, model, max_tokens, results)
            )
            for i in range(concurrency)
        ]

        wall_start = time.perf_counter()

        # Enqueue requests with optional Poisson arrival spacing
        for i in range(total_requests):
            if arrival_rate:
                inter_arrival = random.expovariate(arrival_rate)
                await asyncio.sleep(inter_arrival)
            await queue.put((f"req-{i}", test_prompt))

        await queue.join()

        # Stop workers
        for _ in range(concurrency):
            await queue.put(None)
        await asyncio.gather(*workers)

        wall_duration = time.perf_counter() - wall_start

    # Compute statistics
    successful = [r for r in results if r.error is None and r.completion_tokens > 0]
    failed = [r for r in results if r.error is not None or r.completion_tokens == 0]

    ttft_ms_list = [r.ttft_s * 1000.0 for r in successful]
    all_itl_ms: List[float] = []
    for r in successful:
        all_itl_ms.extend(r.itl_list_ms)

    total_gen_tokens = sum(r.completion_tokens for r in successful)
    throughput = total_gen_tokens / wall_duration if wall_duration > 0 else 0.0

    return BenchmarkSummary(
        total_requests=total_requests,
        successful_requests=len(successful),
        failed_requests=len(failed),
        duration_s=wall_duration,
        total_tokens_generated=total_gen_tokens,
        tokens_per_sec=throughput,
        p50_ttft_ms=calculate_percentile(ttft_ms_list, 50),
        p90_ttft_ms=calculate_percentile(ttft_ms_list, 90),
        p95_ttft_ms=calculate_percentile(ttft_ms_list, 95),
        p99_ttft_ms=calculate_percentile(ttft_ms_list, 99),
        p50_itl_ms=calculate_percentile(all_itl_ms, 50),
        p90_itl_ms=calculate_percentile(all_itl_ms, 90),
        p99_itl_ms=calculate_percentile(all_itl_ms, 99),
        streams=[asdict(r) for r in results[:100]],  # Store sample streams
    )


def main():
    parser = argparse.ArgumentParser(description="Apex-Infer High Concurrency Load Generator")
    parser.add_argument("--host", type=str, default="localhost", help="Inference server hostname")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model name")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent client workers")
    parser.add_argument("--total-requests", type=int, default=50, help="Total requests to dispatch")
    parser.add_argument("--prompt-len", type=int, default=512, help="Prompt token length (approx)")
    parser.add_argument("--max-tokens", type=int, default=64, help="Tokens to generate per stream")
    parser.add_argument("--arrival-rate", type=float, default=None, help="Poisson lambda (requests/sec)")
    parser.add_argument("--output", type=str, default="benchmarks/results/latest_run.json", help="Output JSON path")
    args = parser.parse_args()

    print(f"=== Starting Load Generator ===")
    print(f"Target: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"Concurrency: {args.concurrency} | Total Requests: {args.total_requests} | Max Tokens: {args.max_tokens}")

    summary = asyncio.run(
        run_benchmark(
            host=args.host,
            port=args.port,
            model=args.model,
            concurrency=args.concurrency,
            total_requests=args.total_requests,
            prompt_len=args.prompt_len,
            max_tokens=args.max_tokens,
            arrival_rate=args.arrival_rate,
        )
    )

    print("\n=== Results Summary ===")
    print(f"Duration:       {summary.duration_s:.2f} s")
    print(f"Successful:     {summary.successful_requests}/{summary.total_requests}")
    print(f"Throughput:     {summary.tokens_per_sec:.2f} tokens/s")
    print(f"TTFT (P50/P90/P99): {summary.p50_ttft_ms:.1f} ms / {summary.p90_ttft_ms:.1f} ms / {summary.p99_ttft_ms:.1f} ms")
    print(f"ITL  (P50/P90/P99): {summary.p50_itl_ms:.1f} ms / {summary.p90_itl_ms:.1f} ms / {summary.p99_itl_ms:.1f} ms")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2)
    print(f"Metrics written to {args.output}")


if __name__ == "__main__":
    main()
