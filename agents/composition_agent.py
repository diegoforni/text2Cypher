"""Composition agent that assembles validated fragments into a final query."""
from typing import List, Optional, Any

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

    def refine(self, fragments: List[str], previous_query: str, error: str, schema: str) -> str:
        """Use the LLM to repair a composed query given a database error.

        The prompt includes the schema, the validated fragments, the previous
        composed query, and the error returned by Neo4j. The model should
        return a single corrected Cypher statement without commentary.
        """
        system_message = (
            "You fix Cypher queries. Given validated fragments and a previous composed query "
            "that failed to execute, output a corrected single Cypher query. Do not include "
            "explanations or code fences. Preserve semantics and RETURN clause."
        )
        prompt = f"""
Database schema:
{schema}

Validated fragments (already executed successfully individually):
{fragments}

Previous composed query:
{previous_query}

Database error:
{error}

Produce a corrected single Cypher query that resolves the error. No commentary.
"""
        span = start_span(
            self.langfuse,
            "compose_refine",
            {"fragments": fragments, "previous_query": previous_query, "error": error},
        )
        response = self.llm.invoke([
            ("system", system_message),
            ("user", prompt),
        ])
        raw = response.content if hasattr(response, "content") else response

        def _coerce_text(value: Any) -> str:
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                parts: list[str] = []
                for v in value:
                    if isinstance(v, str):
                        parts.append(v)
                    elif isinstance(v, dict):
                        t = v.get("text") or v.get("content")
                        if isinstance(t, str):
                            parts.append(t)
                    else:
                        t = getattr(v, "text", None)
                        if isinstance(t, str):
                            parts.append(t)
                return "".join(parts) if parts else str(value)
            return str(value)

        candidate = _coerce_text(raw)
        cleaned = candidate.strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("cypher"):
                cleaned = cleaned[6:].strip()
        finish_span(span, {"query": cleaned})
        return cleaned

    def explain(self, query: str, schema: str) -> str:
        system_message = (
            "You are a cybersecurity-focused expert who explains Cypher queries clearly and concisely."
        )
        prompt = f"""
Schema: {schema}
Query: {query}

Explain in a short sentence what this query does.
"""
        span = start_span(
            self.langfuse,
            "explain",
            {
                # Avoid duplicating large strings in telemetry; keep only final prompt
                "query": query,
                "system": system_message,
                "prompt": prompt,
            },
        )
        response = self.llm.invoke([
            ("system", system_message),
            ("user", prompt),
        ])
        raw = response.content if hasattr(response, "content") else response

        def _coerce_text2(value: Any) -> str:
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                parts: list[str] = []
                for v in value:
                    if isinstance(v, str):
                        parts.append(v)
                    elif isinstance(v, dict):
                        t = v.get("text") or v.get("content")
                        if isinstance(t, str):
                            parts.append(t)
                    else:
                        t = getattr(v, "text", None)
                        if isinstance(t, str):
                            parts.append(t)
                return "".join(parts) if parts else str(value)
            return str(value)

        explanation = _coerce_text2(raw)
        finish_span(span, {"explanation": explanation})
        return explanation
