"""
Quick runner to exercise the DecompositionAgent with a crafted prompt.

Uses the configured LLM if available; otherwise falls back to a dummy LLM
that returns a plausible JSON array so we can test parsing and prints offline.
"""
from __future__ import annotations

import os
import sys
from typing import Any, List, Tuple

# Ensure repository root is on sys.path when running from any directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agents.decomposition_agent import DecompositionAgent
from config import get_llm


class _DummyResponse:
    def __init__(self, content: str):
        self.content = content


class DummyLLM:
    """Minimal stand-in for a chat model with an `invoke` method."""

    def __init__(self, mode: str = "multi"):
        self.mode = mode

    def invoke(self, messages: List[Tuple[str, str]], *args: Any, **kwargs: Any) -> _DummyResponse:
        # Craft responses that exercise the agent's parsing logic
        if self.mode == "multi":
            # Proper JSON array (no code fences)
            return _DummyResponse(
                '["Count distinct attacking IPs in last 30 days", '
                '"Find top 5 countries by number of attacks in last 30 days", '
                '"Determine the most common attack technique in last 30 days"]'
            )
        if self.mode == "fenced":
            # Code-fenced JSON, to test stripping
            return _DummyResponse(
                """```json
[
  "Total number of distinct IPs in 2024",
  "Top 3 techniques overall"
]
```"""
            )
        if self.mode == "prose":
            # Prose with an array embedded
            return _DummyResponse(
                "The tasks are as follows: [\"A\", \"B\"]. Please proceed."
            )
        # Worst case: not JSON at all — should fall back
        return _DummyResponse("From the analysis below, we might split...")


def main() -> None:
    # Crafted prompt that should trigger decomposition into multiple independent tasks
    default_message = (
        "Report three independent metrics for the past 30 days. "
        "Treat each as a separate task that will become an independent Cypher query: "
        "(1) total number of distinct attacking IPs, "
        "(2) top 5 countries by number of attacks, "
        "(3) most common attack technique overall."
    )
    crafted_message = " ".join(sys.argv[1:]).strip() or default_message

    # Default schema copied from main.py for convenience
    schema = (
        "Nodes: (:IP {address, botnet, pulse_ids}), (:Country {name}), (:ASN {number})\n"
        "Relationships: (ip:IP)-[:ATTACKS {technique, protocol, technology, industry, pulse_id, "
        "documented_create_date, documented_modified_date}]->(country:Country)\n"
        "Additional: (ip:IP)-[:FROM]->(country:Country), (ip:IP)-[:BELONGS_TO]->(asn:ASN)"
    )

    # Prefer real LLM if configured; otherwise use dummy
    llm = get_llm() or DummyLLM(mode="multi")
    agent = DecompositionAgent(llm)

    print("[test] crafted message:", crafted_message)
    print("[test] running decomposer...\n")
    subproblems = agent.decompose(crafted_message, schema)
    print("\n[test] final subproblems:")
    for i, s in enumerate(subproblems, 1):
        print(f"  {i}. {s}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
