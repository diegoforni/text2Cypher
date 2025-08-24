"""Expansion agent that clarifies and enriches user requests."""
from typing import Optional

from langchain_core.language_models import BaseChatModel
from langfuse import Langfuse


class ExpansionAgent:
    """Ask clarifying questions and produce a schema-grounded description."""

    def __init__(self, llm: BaseChatModel, langfuse: Optional[Langfuse] = None):
        self.llm = llm
        self.langfuse = langfuse

    def expand(self, request: str, schema: str) -> str:
        prompt = (
            "You are an assistant that expands short user requests.\n"
            "Schema information: {schema}\n"
            "If the request is ambiguous, ask for clarification.\n"
            "Return a detailed, schema-grounded description.".format(schema=schema)
        )
        if self.langfuse:
            span = self.langfuse.span("expand")
            span.log_inputs({"request": request})
        response = self.llm.invoke([
            ("system", "You expand user requests for Cypher queries."),
            ("user", f"Request: {request}\n{prompt}"),
        ])
        expanded = response.content if hasattr(response, "content") else str(response)
        if self.langfuse:
            span.log_outputs({"expanded": expanded})
            span.end()
        return expanded
