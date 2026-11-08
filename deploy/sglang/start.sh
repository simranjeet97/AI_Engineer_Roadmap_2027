#!/usr/bin/env bash
# ==============================================================================
# SGLang Production Launch Script
# Configured for: RadixAttention tree-based prefix caching & FlashInfer kernels
# ==============================================================================
set -euo pipefail

MODEL_ID=${MODEL_ID:-"meta-llama/Meta-Llama-3-8B-Instruct"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-30000}
TP_SIZE=${TP_SIZE:-1}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.90}
CONTEXT_LENGTH=${CONTEXT_LENGTH:-8192}
SCHEDULE_HEURISTIC=${SCHEDULE_HEURISTIC:-"lpm"}

echo "=========================================================="
echo "Starting Apex-Infer SGLang RadixAttention Server"
echo "Model:                ${MODEL_ID}"
echo "TP Size:              ${TP_SIZE}"
echo "Mem Fraction Static:  ${MEM_FRACTION_STATIC}"
echo "Context Length:       ${CONTEXT_LENGTH}"
echo "Schedule Heuristic:   ${SCHEDULE_HEURISTIC}"
echo "Host:Port:            ${HOST}:${PORT}"
echo "=========================================================="

exec python3 -m sglang.launch_server \
    --model-path "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tp "${TP_SIZE}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --context-length "${CONTEXT_LENGTH}" \
    --schedule-heuristic "${SCHEDULE_HEURISTIC}" \
    --enable-metrics \
    --trust-remote-code
