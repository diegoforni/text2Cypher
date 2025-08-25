"""Expansion agent that clarifies and enriches user requests."""
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel

from .langfuse_utils import start_span, finish_span


class ExpansionAgent:
    """Ask clarifying questions and produce a schema-grounded description."""

    def __init__(self, llm: BaseChatModel, trace: Optional[Any] = None):
        self.llm = llm
        self.trace = trace

    def expand(self, request: str, schema: str) -> str:
        system_message = (
            "You are a data analysis expert specializing in graph databases and cybersecurity data. "
            "Your job is ONLY to clarify the request and capture context for later query generation. "
            "Do NOT output or suggest any Cypher syntax."
        )
        prompt = f"""
Analyze this question and enrich it with contextual details. Use natural language only.

Question: {request}
Schema: {schema}

Provide a JSON object with these keys:
1. INTENT – what is the user trying to find out?
2. KEY_ENTITIES – which nodes/relationships are relevant?
3. FILTERS – which conditions or values must be matched?
4. OUTPUT_FORMAT – how should results be presented?
5. COMPLEXITY_NOTES – any special considerations?
6. SUGGESTED_APPROACH – high-level strategy, no Cypher.
7. NOTE – respect given values exactly; values may span multiple words.

Do NOT include any query language in your response.
"""
        span = start_span(self.trace, "expand", {"request": request})
        response = self.llm.invoke([
            ("system", system_message),
            ("user", prompt),
        ])
        expanded = response.content if hasattr(response, "content") else str(response)
        finish_span(span, {"expanded": expanded})
        return expanded
