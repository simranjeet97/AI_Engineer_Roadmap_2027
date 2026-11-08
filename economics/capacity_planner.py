#!/usr/bin/env python3
"""
Apex-Infer Diurnal Traffic Cluster Sizing & Autoscaling Planner
Models 24-hour sinusoidal customer traffic profiles to determine static reserved vs dynamic spot capacity.
"""

import argparse
import math


def plan_cluster_capacity(
    peak_qps: float,
    tokens_per_req: int,
    gpu_capacity_tok_s: float = 350.0,
    peak_to_trough_ratio: float = 4.0,
    headroom_factor: float = 1.30,
):
    peak_tokens_per_sec = peak_qps * tokens_per_req
    provisioned_peak_tok_s = peak_tokens_per_sec * headroom_factor

    peak_gpus = math.ceil(provisioned_peak_tok_s / gpu_capacity_tok_s)
    trough_gpus = math.ceil(peak_gpus / peak_to_trough_ratio)

    print("=" * 75)
    print("Apex-Infer Diurnal Traffic Capacity Plan")
    print("=" * 75)
    print(f"Peak Traffic:               {peak_qps:.1f} req/s ({peak_tokens_per_sec:,.0f} tok/s)")
    print(f"Safety Headroom:            {(headroom_factor - 1.0) * 100:.0f}%")
    print(f"Per-GPU Throughput Target:  {gpu_capacity_tok_s:.0f} tok/s")
    print("-" * 75)
    print(f"Peak GPUs Required:         {peak_gpus} GPUs ({math.ceil(peak_gpus / 8)}x 8-GPU nodes)")
    print(f"Trough (Off-peak) GPUs:     {trough_gpus} GPUs ({math.ceil(trough_gpus / 8)}x 8-GPU nodes)")
    print("-" * 75)
    print("Recommended Autoscaling Strategy:")
    print(f" - Reserved Base Instances: {trough_gpus} GPUs (100% committed discount)")
    print(f" - Dynamic KEDA Elastic:    {peak_gpus - trough_gpus} GPUs (scaled on queue depth)")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Capacity Planner")
    parser.add_argument("--peak-qps", type=float, default=50.0, help="Peak requests per second")
    parser.add_argument("--tokens-per-req", type=int, default=1024, help="Average total tokens per request")
    args = parser.parse_args()
    plan_cluster_capacity(args.peak_qps, args.tokens_per_req)


if __name__ == "__main__":
    main()
