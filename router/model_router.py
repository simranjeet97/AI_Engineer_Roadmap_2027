#!/usr/bin/env python3
"""
Apex-Infer Adaptive Cost / Latency / Quality Model Router
Cascades queries across small local models (Llama-3-8B), large local models (Llama-3-70B),
and frontier API fallbacks based on real-time SLA deadlines, query complexity, and confidence scoring.
"""

import argparse
from dataclasses import dataclass
from typing import Dict, Tuple

try:
    from router.intent_classifier import IntentClassifier
except ImportError:
    from intent_classifier import IntentClassifier


@dataclass
class RouteDecision:
    target_engine: str
    target_model: str
    complexity_score: float
    estimated_cost_per_1k: float
    reason: str


class ApexModelRouter:
    def __init__(self):
        self.classifier = IntentClassifier()

    def route_query(
        self,
        query: str,
        max_latency_budget_ms: float = 500.0,
        enforce_cost_minimization: bool = False,
    ) -> RouteDecision:
        complexity = self.classifier.score_complexity(query)

        # Routing Policy:
        # Tier 1 (Low complexity < 0.35 or strict latency <= 100ms): Llama-3-8B-Instruct
        # Tier 2 (Medium complexity 0.35..0.75 and budget <= 600ms): Llama-3-70B-Instruct
        # Tier 3 (High complexity >= 0.75): Frontier Claude / GPT-4o API

        if enforce_cost_minimization or (complexity < 0.35 and max_latency_budget_ms >= 30.0):
            return RouteDecision(
                target_engine="local_vllm_8b",
                target_model="meta-llama/Meta-Llama-3-8B-Instruct",
                complexity_score=complexity,
                estimated_cost_per_1k=0.0001,
                reason="Low complexity heuristic match. Routed to 8B for maximal throughput.",
            )
        elif complexity < 0.75 and max_latency_budget_ms >= 150.0:
            return RouteDecision(
                target_engine="local_vllm_70b",
                target_model="meta-llama/Meta-Llama-3-70B-Instruct",
                complexity_score=complexity,
                estimated_cost_per_1k=0.0006,
                reason="Intermediate reasoning / code analysis. Routed to 70B.",
            )
        else:
            return RouteDecision(
                target_engine="frontier_cloud_api",
                target_model="claude-3-5-sonnet",
                complexity_score=complexity,
                estimated_cost_per_1k=0.0060,
                reason="High complexity mathematical derivation or multi-step logic. Routed to frontier API.",
            )


def main():
    router = ApexModelRouter()
    sample_queries = [
        "What is the capital of France?",
        "Write a Python script using asyncio to scrape web endpoints concurrently.",
        "Derive the exact proof of Little's Law for an M/M/1 queuing system and formulate the HBM memory bandwidth bound.",
    ]

    print("=== Apex-Infer Model Cascade Router ===")
    for q in sample_queries:
        decision = router.route_query(q)
        print(f"\nQuery: '{q[:60]}...'")
        print(f" -> Model: {decision.target_model} [{decision.target_engine}]")
        print(f" -> Complexity: {decision.complexity_score:.2f} | Cost: ${decision.estimated_cost_per_1k:.5f}/1k tok")
        print(f" -> Reason: {decision.reason}")


if __name__ == "__main__":
    main()
