"""Decomposition agent that splits expanded descriptions into subproblems."""
from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langfuse import Langfuse


class DecompositionAgent:
    """Break a problem description into independent subproblems."""

    def __init__(self, llm: BaseChatModel, langfuse: Optional[Langfuse] = None):
        self.llm = llm
        self.langfuse = langfuse

    def decompose(self, description: str) -> List[str]:
        if self.langfuse:
            span = self.langfuse.span("decompose")
            span.log_inputs({"description": description})
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
        if self.langfuse:
            span.log_outputs({"subproblems": subproblems})
            span.end()
        return subproblems
