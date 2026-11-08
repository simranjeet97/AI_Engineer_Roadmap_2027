#!/usr/bin/env bash
# ==============================================================================
# vLLM High-Throughput Production Launch Script
# Configured for: Chunked Prefill, Automatic Prefix Caching, and High Concurrency
# ==============================================================================
set -euo pipefail

MODEL_ID=${MODEL_ID:-"meta-llama/Meta-Llama-3-8B-Instruct"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.92}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-2048}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-256}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-"auto"}

echo "=========================================================="
echo "Starting Apex-Infer vLLM Production Node"
echo "Model:                  ${MODEL_ID}"
echo "Tensor Parallel:        ${TENSOR_PARALLEL_SIZE}"
echo "GPU Mem Utilization:    ${GPU_MEMORY_UTILIZATION}"
echo "Max Model Len:          ${MAX_MODEL_LEN}"
echo "Max Num Batched Tokens: ${MAX_NUM_BATCHED_TOKENS}"
echo "Max Sequences:          ${MAX_NUM_SEQS}"
echo "Host:Port:              ${HOST}:${PORT}"
echo "=========================================================="

# Exec into python module to preserve signal handling for PID 1 container shutdown
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --disable-log-requests \
    --trust-remote-code
