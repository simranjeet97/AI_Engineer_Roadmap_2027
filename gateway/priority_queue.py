#!/usr/bin/env python3
"""
Apex-Infer Gateway Priority Scheduler & Load Shedder
Prioritizes VIP and low-latency interactive requests while gracefully shedding batch/background requests
during queue backlog spikes.
"""

import heapq
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional


class PriorityTier(IntEnum):
    CRITICAL_VIP = 0
    INTERACTIVE = 1
    BATCH_BACKGROUND = 2


@dataclass(order=True)
class QueuedRequest:
    priority: int
    arrival_time: float
    request_id: str = field(compare=False)
    tenant_id: str = field(compare=False)
    prompt_len: int = field(compare=False)
    max_tokens: int = field(compare=False)


class GatewayPriorityScheduler:
    def __init__(self, max_queue_capacity: int = 500, shedding_threshold: int = 400):
        self.max_queue_capacity = max_queue_capacity
        self.shedding_threshold = shedding_threshold
        self._heap: List[QueuedRequest] = []

    def enqueue(self, tier: PriorityTier, request_id: str, tenant_id: str, prompt_len: int, max_tokens: int) -> bool:
        current_len = len(self._heap)

        # Drop lowest priority traffic first when backlogged
        if current_len >= self.shedding_threshold and tier == PriorityTier.BATCH_BACKGROUND:
            print(f"[SHED] Shedding batch request {request_id} (queue depth {current_len} >= {self.shedding_threshold})")
            return False

        if current_len >= self.max_queue_capacity:
            print(f"[REJECT] Queue fully saturated ({current_len}/{self.max_queue_capacity}). Rejecting request {request_id}")
            return False

        req = QueuedRequest(
            priority=int(tier),
            arrival_time=time.perf_counter(),
            request_id=request_id,
            tenant_id=tenant_id,
            prompt_len=prompt_len,
            max_tokens=max_tokens,
        )
        heapq.heappush(self._heap, req)
        return True

    def dequeue(self) -> Optional[QueuedRequest]:
        if not self._heap:
            return None
        return heapq.heappop(self._heap)

    @property
    def depth(self) -> int:
        return len(self._heap)


def main():
    print("=== Testing Priority Queue Scheduling & Load Shedding ===")
    scheduler = GatewayPriorityScheduler(max_queue_capacity=10, shedding_threshold=5)

    # Fill queue to shedding point
    for i in range(5):
        scheduler.enqueue(PriorityTier.INTERACTIVE, f"chat-{i}", "user-std", 512, 64)

    # Attempt to add batch requests
    for i in range(3):
        scheduler.enqueue(PriorityTier.BATCH_BACKGROUND, f"batch-{i}", "crawler", 1024, 256)

    # Add VIP request
    scheduler.enqueue(PriorityTier.CRITICAL_VIP, "vip-query", "enterprise-corp", 256, 128)

    print(f"\nTotal Enqueued Items: {scheduler.depth}")
    print("Dequeuing requests in priority order:")
    while scheduler.depth > 0:
        req = scheduler.dequeue()
        print(f" -> Dequeued [{PriorityTier(req.priority).name}]: {req.request_id} (arrival: {req.arrival_time:.4f})")


if __name__ == "__main__":
    main()
