# Staff-Grade Production Dockerfile for SGLang High-Performance Inference Engine
FROM lmsysorg/sglang:latest

LABEL maintainer="Simranjeet Singh <simranjeet97>"
LABEL description="SGLang server image configured for RadixAttention prefix caching and high-speed execution"

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    jq \
    procps \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    NCCL_DEBUG=WARN

COPY start.sh /opt/sglang/start.sh
RUN chmod +x /opt/sglang/start.sh

EXPOSE 30000

HEALTHCHECK --interval=10s --timeout=5s --start-period=45s --retries=3 \
  CMD curl -f http://localhost:30000/health || exit 1

WORKDIR /opt/sglang
ENTRYPOINT ["/opt/sglang/start.sh"]
