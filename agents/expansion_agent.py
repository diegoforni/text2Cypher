"""Expansion agent that clarifies and enriches user requests."""
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langfuse import Langfuse

from .langfuse_utils import start_span, finish_span


class ExpansionAgent:
    """Ask clarifying questions and produce a schema-grounded description."""

    def __init__(self, llm: BaseChatModel, langfuse: Optional[Langfuse] = None):
        self.llm = llm
        self.langfuse = langfuse

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
5. SUGGESTED_APPROACH – high-level strategy (no Cypher).
"""
        span = start_span(
            self.langfuse,
            "expand",
            {
                # Avoid duplicating large strings in telemetry; keep only final prompt
                "request": request,
                "system": system_message,
                "prompt": prompt,
            },
        )
        response = self.llm.invoke([
            ("system", system_message),
            ("user", prompt),
        ])
        expanded = response.content if hasattr(response, "content") else str(response)
        finish_span(span, {"expanded": expanded})
        return expanded
