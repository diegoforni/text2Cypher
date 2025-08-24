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
        prompt = (
            "Split the following description into independent Cypher subproblems.\n"
            "Return a JSON list of subproblem strings.\n"
            f"Description: {description}"
        )
        response = self.llm.invoke([
            ("system", "You are a task decomposition assistant."),
            ("user", prompt),
        ])
        text = response.content if hasattr(response, "content") else str(response)
        try:
            subproblems = [p.strip() for p in eval(text) if p.strip()]
        except Exception:
            subproblems = [text]
        finish_span(span, {"subproblems": subproblems})
        return subproblems
