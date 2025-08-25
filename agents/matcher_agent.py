"""Matcher agent that resolves literal values against the database."""
from __future__ import annotations

from typing import Dict, List, Optional
import json

from langchain_core.language_models import BaseChatModel
from neo4j import Driver
from langfuse import Langfuse

from config import NEO4J_DB
from .langfuse_utils import start_span, finish_span


class MatcherAgent:
    """Extract field-value pairs and match them to database values."""

    def __init__(self, llm: BaseChatModel, driver: Driver, langfuse: Optional[Langfuse] = None):
        self.llm = llm
        self.driver = driver
        self.langfuse = langfuse

    def _extract_pairs(self, description: str, schema: str) -> List[Dict[str, str]]:
        """Use the LLM to extract label/property/value triples from text."""
        system_message = (
            "You extract field-value pairs for a Neo4j graph. "
            "Return JSON array objects with keys: label, property, value."
        )
        prompt = f"""
Schema:\n{schema}\n
Text:\n{description}\n
List every literal value that must appear in the query with its node label and property name.
Return a JSON array like:
[
  {{"label": "Person", "property": "name", "value": "Tom"}}
]
Use the exact input value.
"""
        response = self.llm.invoke([
            ("system", system_message),
            ("user", prompt),
        ])
        text = response.content if hasattr(response, "content") else str(response)
        try:
            pairs = json.loads(text)
            if not isinstance(pairs, list):  # ensure list
                raise ValueError
            return [
                {"label": p.get("label", ""), "property": p.get("property", ""), "value": p.get("value", "")}
                for p in pairs
                if p.get("value")
            ]
        except Exception:
            return []

    def _exact_match(self, label: str, prop: str, value: str) -> str:
        """Fetch all values for label.prop and return exact match if present."""
        if not label or not prop:
            return value
        cypher = f"MATCH (n:`{label}`) WHERE n.{prop} IS NOT NULL RETURN DISTINCT toString(n.{prop}) AS val"
        with self.driver.session(database=NEO4J_DB) as session:
            results = [r["val"] for r in session.run(cypher)]
        for candidate in results:
            if str(candidate).lower() == value.lower():
                return str(candidate)
        return value

    def match(self, description: str, schema: str) -> List[Dict[str, str]]:
        """Extract and resolve field-value pairs in description."""
        span = start_span(self.langfuse, "match", {"description": description})
        pairs = self._extract_pairs(description, schema)
        resolved: List[Dict[str, str]] = []
        for pair in pairs:
            matched = self._exact_match(pair["label"], pair["property"], pair["value"])
            resolved.append({"label": pair["label"], "property": pair["property"], "value": matched})
        finish_span(span, {"pairs": resolved})
        return resolved
