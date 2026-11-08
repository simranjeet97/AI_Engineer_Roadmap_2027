#!/usr/bin/env python3
"""
Apex-Infer Publication-Grade Latency Percentile & CDF Generator
Generates cumulative distribution functions (CDF) and throughput curves from benchmark JSON runs.
"""

import argparse
import json
import os
import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    plt = None
    np = None


def plot_benchmark_results(json_path: str, output_path: str):
    if plt is None or np is None:
        print("[WARN] matplotlib and numpy are required to render graphical plots.")
        print("Run: pip install matplotlib numpy")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    streams = data.get("streams", [])
    if not streams:
        print("[ERROR] No stream records found in benchmark JSON.")
        return

    ttft_vals = [s["ttft_s"] * 1000.0 for s in streams if s.get("ttft_s", 0) > 0]
    all_itls = []
    for s in streams:
        all_itls.extend(s.get("itl_list_ms", []))

    if not ttft_vals:
        print("[ERROR] No TTFT measurements found.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0f172a')

    for ax in axes:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        ax.spines['bottom'].set_color('#475569')
        ax.spines['top'].set_color('#475569')
        ax.spines['left'].set_color('#475569')
        ax.spines['right'].set_color('#475569')
        ax.grid(True, linestyle='--', alpha=0.3, color='#64748b')

    # TTFT CDF
    sorted_ttft = np.sort(ttft_vals)
    p_ttft = 1.0 * np.arange(len(sorted_ttft)) / (len(sorted_ttft) - 1) if len(sorted_ttft) > 1 else [1.0]

    axes[0].plot(sorted_ttft, p_ttft, color='#38bdf8', linewidth=2.5, label='TTFT CDF')
    axes[0].set_title('Time to First Token (TTFT) CDF', color='#f8fafc', fontsize=12, pad=12)
    axes[0].set_xlabel('Latency (ms)', color='#cbd5e1')
    axes[0].set_ylabel('Cumulative Probability', color='#cbd5e1')
    axes[0].axvline(np.percentile(sorted_ttft, 50), color='#22c55e', linestyle=':', label=f"P50: {np.percentile(sorted_ttft, 50):.1f}ms")
    axes[0].axvline(np.percentile(sorted_ttft, 99), color='#ef4444', linestyle=':', label=f"P99: {np.percentile(sorted_ttft, 99):.1f}ms")
    axes[0].legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0')

    # ITL Distribution / CDF
    if all_itls:
        sorted_itl = np.sort(all_itls)
        p_itl = 1.0 * np.arange(len(sorted_itl)) / (len(sorted_itl) - 1) if len(sorted_itl) > 1 else [1.0]
        axes[1].plot(sorted_itl, p_itl, color='#a855f7', linewidth=2.5, label='ITL CDF')
        axes[1].set_title('Inter-Token Latency (ITL) CDF', color='#f8fafc', fontsize=12, pad=12)
        axes[1].set_xlabel('Latency (ms)', color='#cbd5e1')
        axes[1].set_ylabel('Cumulative Probability', color='#cbd5e1')
        axes[1].axvline(np.percentile(sorted_itl, 50), color='#22c55e', linestyle=':', label=f"P50: {np.percentile(sorted_itl, 50):.1f}ms")
        axes[1].axvline(np.percentile(sorted_itl, 99), color='#ef4444', linestyle=':', label=f"P99: {np.percentile(sorted_itl, 99):.1f}ms")
        axes[1].legend(facecolor='#1e293b', edgecolor='#475569', labelcolor='#e2e8f0')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"[OK] Latency CDF chart saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate latency percentile curves")
    parser.add_argument("--input", default="benchmarks/results/sample_benchmark_run.json")
    parser.add_argument("--output", default="benchmarks/results/latency_percentiles.png")
    args = parser.parse_args()

    plot_benchmark_results(args.input, args.output)


if __name__ == "__main__":
    main()
