"""Composition agent that assembles validated fragments into a final query."""
from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langfuse import Langfuse

from .langfuse_utils import start_span, finish_span


class CompositionAgent:
    """Compose final Cypher statement and provide explanation."""

    def __init__(self, llm: BaseChatModel, langfuse: Optional[Langfuse] = None):
        self.llm = llm
        self.langfuse = langfuse

    def compose(self, fragments: List[str]) -> str:
        query = "\n".join(fragments)
        span = start_span(self.langfuse, "compose", {"fragments": fragments})
        finish_span(span, {"query": query})
        return query

    def explain(self, query: str, schema: str) -> str:
        prompt = (
            "Schema: {schema}\n"
            "Query: {query}\n"
            "Explain in a short sentence what this query does.".format(
                schema=schema, query=query
            )
        )
        response = self.llm.invoke([
            ("system", "You explain Cypher queries."),
            ("user", prompt),
        ])
        return response.content if hasattr(response, "content") else str(response)
