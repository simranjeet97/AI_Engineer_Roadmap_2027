#!/usr/bin/env python3
"""
Apex-Infer Automatic Prefix Caching (APC) Verification Harness
Validates TTFT speedup between cold cache and warm cache passes on identical prompts.
"""

import argparse
import json
import time
import urllib.request


SYSTEM_PROMPT = """You are the chief systems architect of an ultra-high performance distributed inference cluster.
Your responsibilities include auditing tensor parallelism, pipeline parallelism, continuous batching schedulers,
FlashAttention-3 forward passes, PagedAttention block table allocations, RadixAttention tree structures,
and FP8 quantization scaling factor calculations. When responding to queries, adhere strictly to the
laws of computer architecture, roofline operational intensity bounds, and little's law queueing theory.
""" * 20  # ~2,500 prompt tokens


def execute_request(url: str, model: str, system_text: str, user_query: str) -> float:
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_query}
        ],
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": True
    }).encode("utf-8")

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=60) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                ttft_ms = (time.perf_counter() - t0) * 1000.0
                return ttft_ms
    return (time.perf_counter() - t0) * 1000.0


def main():
    parser = argparse.ArgumentParser(description="Verify Automatic Prefix Caching TTFT Speedup")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    args = parser.parse_args()

    endpoint = f"http://{args.host}:{args.port}/v1/chat/completions"

    print("=== Automatic Prefix Caching (APC) Verification ===")
    print(f"System Prompt Length: ~{len(SYSTEM_PROMPT.split())} words")

    print("\n[Pass 1: Cold Cache] Sending fresh system prompt...")
    ttft_cold = execute_request(endpoint, args.model, SYSTEM_PROMPT, "Pass 1 query: Compute the H100 knee point.")
    print(f"Cold TTFT: {ttft_cold:.2f} ms")

    print("\n[Pass 2: Warm Cache] Sending identical prefix with new user query...")
    ttft_warm = execute_request(endpoint, args.model, SYSTEM_PROMPT, "Pass 2 query: What is the operational intensity?")
    print(f"Warm TTFT: {ttft_warm:.2f} ms")

    speedup = ttft_cold / max(ttft_warm, 0.001)
    print(f"\nPrefix Cache Speedup: {speedup:.2f}x")

    if speedup >= 2.0:
        print("[SUCCESS] Prefix caching verified. Substantial TTFT latency reduction achieved.")
    else:
        print("[NOTICE] Speedup was less than 2x. Ensure '--enable-prefix-caching' is enabled in engine launch flags.")


if __name__ == "__main__":
    main()
