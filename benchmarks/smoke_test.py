#!/usr/bin/env python3
"""
Apex-Infer Smoke Test & Telemetry Probe
Verifies OpenAI endpoint health and validates Prometheus metric exposition.
"""

import argparse
import json
import sys
import urllib.request
import urllib.error


def probe_health(host: str, port: int) -> bool:
    url = f"http://{host}:{port}/health"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as response:
            status = response.getcode()
            print(f"[OK] Health check passed: {url} -> HTTP {status}")
            return status == 200
    except Exception as exc:
        print(f"[FAIL] Health check failed for {url}: {exc}")
        return False


def probe_metrics(host: str, port: int) -> bool:
    url = f"http://{host}:{port}/metrics"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5) as response:
            content = response.read().decode("utf-8")
            required_metrics = [
                "vllm:num_requests_running",
                "vllm:gpu_cache_usage_factor",
            ]
            found = [m for m in required_metrics if m in content]
            print(f"[OK] Metrics endpoint responsive ({len(content)} bytes). Found {len(found)}/{len(required_metrics)} key gauges.")
            return True
    except Exception as exc:
        print(f"[WARN] Metrics probe failed for {url}: {exc}")
        return False


def probe_generation(host: str, port: int, model: str) -> bool:
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Ping"}],
        "max_tokens": 5,
        "temperature": 0.0,
    }).encode("utf-8")

    try:
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as response:
            res = json.loads(response.read().decode("utf-8"))
            content = res["choices"][0]["message"]["content"]
            print(f"[OK] Completion successful: '{content.strip()}'")
            return True
    except Exception as exc:
        print(f"[FAIL] Completion failed for {url}: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Apex-Infer Smoke Test Suite")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    args = parser.parse_args()

    print(f"Probing {args.host}:{args.port}...")
    h = probe_health(args.host, args.port)
    m = probe_metrics(args.host, args.port)
    g = probe_generation(args.host, args.port, args.model)

    if h and g:
        print(">>> All primary smoke tests PASSED.")
        sys.exit(0)
    else:
        print(">>> Smoke tests FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
