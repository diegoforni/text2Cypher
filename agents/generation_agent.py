"""Generation agent that creates Cypher fragments for each subproblem."""
from __future__ import annotations

from typing import Any, List, Optional
import json

from langchain_core.language_models import BaseChatModel

from .langfuse_utils import start_span, finish_span


class GenerationAgent:
    """Generate Cypher query fragments using verified field values."""

    def __init__(self, llm: BaseChatModel, trace: Optional[Any] = None):
        self.llm = llm
        self.trace = trace

    def generate(self, subproblem: str, schema: str, pairs: List[dict]) -> str:
        span = start_span(self.trace, "generate", {"subproblem": subproblem, "pairs": pairs})
        system_message = (
            "You are a Cypher query expert. Use the provided database schema to solve the given subproblem. "
            "Always produce a complete Cypher fragment ending with a RETURN clause. "
            "Do not include explanations or commentary. "
            "CRITICAL: Do not nest MATCH inside WHERE clauses. "
            "Allowed clauses: MATCH, WHERE, WITH, RETURN, ORDER BY, LIMIT. "
            "Properties like 'technique' or 'protocol' are on relationships [:ATTACKS {technique: \"value\"}]. "
            "Attack relationships follow the pattern (ip:IP)-[:ATTACKS]->(country:Country). "
            "Use WITH or multiple MATCH statements for complex logic and preserve provided values exactly."
        )
        prompt = (
            f"Database schema:\n{schema}\n\n"
            f"Subproblem:\n{subproblem}\n\n"
            f"Verified field values:\n{json.dumps(pairs)}\n\n"
            "Return ONLY the Cypher fragment."
        )
        response = self.llm.invoke([
            ("system", system_message),
            ("user", prompt),
        ])
        fragment = response.content if hasattr(response, "content") else str(response)
        finish_span(span, {"fragment": fragment})
        return fragment
