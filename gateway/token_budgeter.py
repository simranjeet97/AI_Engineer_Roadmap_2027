#!/usr/bin/env python3
"""
Apex-Infer Gateway Token Budgeter & Dynamic Output Clamp
Implements Leaky/Token Bucket rate limiting per tenant and dynamic output token clamping
based on engine KV cache saturation to prevent Out-Of-Memory (OOM) evictions.
"""

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class TokenBucket:
    capacity: float          # Maximum burst tokens
    refill_rate: float       # Tokens refilled per second
    tokens: float
    last_update: float

    def consume(self, requested: int) -> bool:
        now = time.perf_counter()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_update = now

        if self.tokens >= requested:
            self.tokens -= requested
            return True
        return False


class GatewayTokenBudgeter:
    """
    Enforces tenant token quotas and guards inference clusters against queue-induced OOMs
    by dynamically clamping max_tokens when GPU KV cache saturation exceeds thresholds.
    """
    def __init__(self, high_watermark_pct: float = 0.85, critical_watermark_pct: float = 0.95):
        self.high_watermark_pct = high_watermark_pct
        self.critical_watermark_pct = critical_watermark_pct
        self.tenant_buckets: Dict[str, TokenBucket] = {}

    def register_tenant(self, tenant_id: str, tokens_per_min: int, burst_tokens: int):
        self.tenant_buckets[tenant_id] = TokenBucket(
            capacity=float(burst_tokens),
            refill_rate=float(tokens_per_min) / 60.0,
            tokens=float(burst_tokens),
            last_update=time.perf_counter(),
        )

    def authorize_request(
        self,
        tenant_id: str,
        estimated_prompt_tokens: int,
        requested_max_tokens: int,
        gpu_kv_usage_pct: float,
    ) -> Tuple[bool, int, str]:
        """
        Returns: (is_allowed, clamped_max_tokens, reason)
        """
        bucket = self.tenant_buckets.get(tenant_id)
        if bucket is None:
            # Default unmetered tenant
            bucket = TokenBucket(100000.0, 10000.0, 100000.0, time.perf_counter())
            self.tenant_buckets[tenant_id] = bucket

        total_budget = estimated_prompt_tokens + requested_max_tokens
        if not bucket.consume(total_budget):
            return False, 0, f"Tenant '{tenant_id}' rate limit exceeded."

        # Dynamic output clamping to defend against memory exhaustion
        clamped_tokens = requested_max_tokens
        if gpu_kv_usage_pct >= self.critical_watermark_pct:
            # Critical: Clamp to 128 tokens or drop
            clamped_tokens = min(requested_max_tokens, 128)
            reason = f"CRITICAL KV cache saturation ({gpu_kv_usage_pct * 100:.1f}%). Clamped to 128 tokens."
        elif gpu_kv_usage_pct >= self.high_watermark_pct:
            # High watermark: Clamp to 512 tokens
            clamped_tokens = min(requested_max_tokens, 512)
            reason = f"HIGH KV cache saturation ({gpu_kv_usage_pct * 100:.1f}%). Clamped to 512 tokens."
        else:
            reason = "OK - unconstrained"

        return True, clamped_tokens, reason


def main():
    print("=== Testing Gateway Token Budgeter ===")
    budgeter = GatewayTokenBudgeter(high_watermark_pct=0.85, critical_watermark_pct=0.95)
    budgeter.register_tenant("tenant-tier1", tokens_per_min=600000, burst_tokens=50000)

    # Normal load
    allowed, clamped, reason = budgeter.authorize_request("tenant-tier1", 500, 2048, 0.40)
    print(f"Normal Load (40% KV): Allowed={allowed}, MaxTokens={clamped} ({reason})")

    # High saturation load
    allowed, clamped, reason = budgeter.authorize_request("tenant-tier1", 500, 2048, 0.88)
    print(f"High Load (88% KV):   Allowed={allowed}, MaxTokens={clamped} ({reason})")

    # Critical saturation load
    allowed, clamped, reason = budgeter.authorize_request("tenant-tier1", 500, 2048, 0.97)
    print(f"Critical (97% KV):    Allowed={allowed}, MaxTokens={clamped} ({reason})")


if __name__ == "__main__":
    main()
