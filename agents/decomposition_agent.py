"""Decomposition agent that splits expanded descriptions into subproblems."""
from typing import Any, List, Optional

from langchain_core.language_models import BaseChatModel

from .langfuse_utils import start_span, finish_span


class DecompositionAgent:
    """Break a problem description into independent subproblems."""

    def __init__(self, llm: BaseChatModel, trace: Optional[Any] = None):
        self.llm = llm
        self.trace = trace

    def decompose(self, description: str) -> List[str]:
        """Split an expanded description into discrete task statements."""
        span = start_span(self.trace, "decompose", {"description": description})
        system_message = (
            "You are a task planner for graph analysis. "
            "Break problems into natural-language tasks without writing any Cypher."
        )
        prompt = f"""
From the analysis below, list the distinct tasks required to build the final query.
Each item must be a concise natural-language description; do NOT include Cypher.

Analysis: {description}

Return a JSON array of task strings using the original values verbatim.
"""
        response = self.llm.invoke([
            ("system", system_message),
            ("user", prompt),
        ])
        text = response.content if hasattr(response, "content") else str(response)
        try:
            subproblems = [p.strip() for p in eval(text) if p.strip()]
        except Exception:
            subproblems = [text]
        finish_span(span, {"subproblems": subproblems})
        return subproblems
