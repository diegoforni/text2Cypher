"""Decomposition agent that splits expanded descriptions into subproblems."""
from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langfuse import Langfuse

from .langfuse_utils import start_span, finish_span


class DecompositionAgent:
    """Break a problem description into independent subproblems."""

    def __init__(self, llm: BaseChatModel, langfuse: Optional[Langfuse] = None):
        self.llm = llm
        self.langfuse = langfuse

    def decompose(self, description: str) -> List[str]:
        span = start_span(self.langfuse, "decompose", {"description": description})
        system_message = "You are a task decomposition assistant for Cypher queries."
        prompt = f"""
Based on the following analysis, break the request into independent Cypher subproblems that can be solved separately.

Analysis: {description}

Return a JSON list of subproblem strings.
Use complete values exactly as provided.
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
