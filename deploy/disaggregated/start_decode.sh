#!/usr/bin/env bash
# ==============================================================================
# Disaggregated Decode Node (Mooncake / Apex-Infer Split)
# Optimizations: Memory-bandwidth bound, large KV cache block capacity, low latency
# ==============================================================================
set -euo pipefail

MODEL_ID=${MODEL_ID:-"meta-llama/Meta-Llama-3-70B-Instruct"}
PORT=${PORT:-8200}
TP_SIZE=${TP_SIZE:-4}
GPU_MEMORY_UTILIZATION=0.94
MAX_NUM_SEQS=512

echo "Starting Disaggregated DECODE Worker on port ${PORT} with TP=${TP_SIZE}..."

exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --enable-prefix-caching \
    --disable-log-requests
