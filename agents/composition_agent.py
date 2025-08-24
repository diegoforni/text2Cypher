"""Composition agent that assembles validated fragments into a final query."""
from typing import List, Optional

from langchain_core.language_models import BaseChatModel
from langfuse import Langfuse


class CompositionAgent:
    """Compose final Cypher statement and provide explanation."""

    def __init__(self, llm: BaseChatModel, langfuse: Optional[Langfuse] = None):
        self.llm = llm
        self.langfuse = langfuse

    def compose(self, fragments: List[str]) -> str:
        query = "\n".join(fragments)
        if self.langfuse:
            span = self.langfuse.span("compose")
            span.log_inputs({"fragments": fragments})
            span.log_outputs({"query": query})
            span.end()
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
