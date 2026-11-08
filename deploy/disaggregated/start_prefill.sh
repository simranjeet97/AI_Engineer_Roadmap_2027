#!/usr/bin/env bash
# ==============================================================================
# Disaggregated Prefill Node (Mooncake / Apex-Infer Split)
# Optimizations: Compute-bound GEMM, high batch token budgets, fast KV export
# ==============================================================================
set -euo pipefail

MODEL_ID=${MODEL_ID:-"meta-llama/Meta-Llama-3-70B-Instruct"}
PORT=${PORT:-8100}
TP_SIZE=${TP_SIZE:-4}
GPU_MEMORY_UTILIZATION=0.90
MAX_NUM_BATCHED_TOKENS=8192

echo "Starting Disaggregated PREFILL Worker on port ${PORT} with TP=${TP_SIZE}..."

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --disable-log-requests
