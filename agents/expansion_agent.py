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
        prompt = (
            "You are an assistant that expands short user requests.\n"
            "Schema information: {schema}\n"
            "If the request is ambiguous, ask for clarification.\n"
            "Return a detailed, schema-grounded description.".format(schema=schema)
        )
        span = start_span(self.langfuse, "expand", {"request": request})
        response = self.llm.invoke([
            ("system", "You expand user requests for Cypher queries."),
            ("user", f"Request: {request}\n{prompt}"),
        ])
        expanded = response.content if hasattr(response, "content") else str(response)
        finish_span(span, {"expanded": expanded})
        return expanded
