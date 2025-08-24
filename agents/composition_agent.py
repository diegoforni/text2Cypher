"""Composition agent that assembles validated fragments into a final query."""
from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langfuse import Langfuse

from .langfuse_utils import start_trace, finish_trace


class CompositionAgent:
    """Compose final Cypher statement and provide explanation."""

    def __init__(self, llm: BaseChatModel, langfuse: Optional[Langfuse] = None):
        self.llm = llm
        self.langfuse = langfuse

    def compose(self, fragments: List[str]) -> str:
        query = "\n".join(fragments)
        trace = start_trace(self.langfuse, "compose", {"fragments": fragments})
        finish_trace(trace, {"query": query})
        return query

    def explain(self, query: str, schema: str) -> str:
        system_message = (
            "You are a cybersecurity-focused expert who explains Cypher queries clearly and concisely."
        )
        prompt = f"""
Schema: {schema}
Query: {query}

Explain in a short sentence what this query does.
"""
        trace = start_trace(self.langfuse, "explain", {"query": query, "schema": schema})
        response = self.llm.invoke([
            ("system", system_message),
            ("user", prompt),
        ])
        explanation = response.content if hasattr(response, "content") else str(response)
        finish_trace(trace, {"explanation": explanation})
        return explanation
