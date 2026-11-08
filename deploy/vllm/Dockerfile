# Staff-Grade Production Dockerfile for vLLM High-Throughput Inference Engine
FROM vllm/vllm-openai:v0.6.3.post1

LABEL maintainer="Simranjeet Singh <simranjeet97>"
LABEL description="Optimized production container for vLLM with chunked prefill, prefix caching, and Marlin kernels"

# Install production system utilities
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    jq \
    net-tools \
    numactl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Set high-performance runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    VLLM_ENGINE_ITERATION_TIMEOUT_S=60 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    NCCL_DEBUG=WARN \
    NCCL_IB_DISABLE=0 \
    NCCL_NET_GDR_LEVEL=3 \
    TORCH_CUDA_ARCH_LIST="8.0 8.9 9.0"

# Copy custom entrypoint script
COPY start.sh /opt/vllm/start.sh
RUN chmod +x /opt/vllm/start.sh

# Expose HTTP API & Prometheus metrics port
EXPOSE 8000

# Production Healthcheck
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

WORKDIR /opt/vllm
ENTRYPOINT ["/opt/vllm/start.sh"]
