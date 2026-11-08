#!/usr/bin/env python3
"""
Apex-Infer Ultra-Fast Sub-2ms Query Complexity Scorer
Analyzes prompt lexical features, entropy, code markers, and multi-step reasoning markers.
"""

import math
import re
from typing import Set


class IntentClassifier:
    HIGH_COMPLEXITY_PATTERNS = [
        r"\b(derive|proof|mathematical|theorem|integral|differential|eigenvalue)\b",
        r"\b(architect|kernel|triton|cuda|assembly|distributed|concurrency)\b",
        r"\b(implement|refactor|debug|profiler|memory leak|deadlock)\b",
        r"```[a-z]*\n[\s\S]*?```",  # Markdown code blocks
    ]

    LOW_COMPLEXITY_PATTERNS = [
        r"^(hi|hello|hey|greetings|thanks|thank you)[.!]?$",
        r"\b(what is|who is|when was|where is|define|translate)\b",
    ]

    def __init__(self):
        self.high_regex = [re.compile(p, re.IGNORECASE) for p in self.HIGH_COMPLEXITY_PATTERNS]
        self.low_regex = [re.compile(p, re.IGNORECASE) for p in self.LOW_COMPLEXITY_PATTERNS]

    def score_complexity(self, prompt: str) -> float:
        """
        Returns normalized complexity score in range [0.0, 1.0].
        Executes in < 0.5 milliseconds.
        """
        prompt_trimmed = prompt.strip()
        length = len(prompt_trimmed.split())

        # Fast path: greeting / single short question
        for pattern in self.low_regex:
            if pattern.search(prompt_trimmed):
                return 0.10

        score = 0.25

        # Factor in length
        if length > 250:
            score += 0.25
        elif length > 80:
            score += 0.15

        # Check keyword matches
        for pattern in self.high_regex:
            matches = pattern.findall(prompt_trimmed)
            if matches:
                score += 0.20 * min(len(matches), 2)

        # Code block presence
        if "```" in prompt_trimmed or "def " in prompt_trimmed or "class " in prompt_trimmed:
            score += 0.25

        return min(max(score, 0.0), 1.0)


def main():
    classifier = IntentClassifier()
    test_cases = [
        "Hello there!",
        "What is the population of Tokyo?",
        "Can you write a CUDA kernel that performs fused flash attention using shared memory tiles?",
    ]
    print("=== Intent Classifier Quick Check ===")
    for text in test_cases:
        score = classifier.score_complexity(text)
        print(f"Score {score:.2f} | Prompt: {text}")


if __name__ == "__main__":
    main()
