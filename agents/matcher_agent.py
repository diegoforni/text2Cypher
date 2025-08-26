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
            "For each literal value, identify the node label or relationship type and property where it belongs. "
            "Return JSON objects with keys: kind, label, property, value. "
            "Kind must be 'node' or 'relationship'."
        )
        prompt = f"""
Schema:\n{schema}\n
Text:\n{description}\n
List every literal value that must appear in the query with its element type, label and property name.
Return a JSON array like:
[
  {{"kind": "node", "label": "Person", "property": "name", "value": "Tom"}}
]
Use the exact input value.
"""
        span = start_span(
            self.langfuse,
            "match.extract_pairs",
            {"system": system_message, "prompt": prompt},
        )
        response = self.llm.invoke([
            ("system", system_message),
            ("user", prompt),
        ])
        text = response.content if hasattr(response, "content") else str(response)
        print("[matcher] LLM response:", text)
        try:
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()
            start, end = cleaned.find("["), cleaned.rfind("]")
            if start != -1 and end != -1:
                cleaned = cleaned[start : end + 1]
            print("[matcher] cleaned response:", cleaned)
            pairs = json.loads(cleaned)
            if not isinstance(pairs, list):
                raise ValueError("expected list")
            parsed = [
                {
                    "kind": p.get("kind", ""),
                    "label": p.get("label", ""),
                    "property": p.get("property", ""),
                    "value": p.get("value", ""),
                }
                for p in pairs
                if p.get("value")
            ]
            print("[matcher] extracted pairs:", parsed)
            finish_span(span, {"response": text, "pairs": parsed})
            return parsed
        except Exception as e:
            print("[matcher] failed to parse pairs:", e)
            finish_span(span, {"response": text}, e)
            return []

    def _match_value(self, kind: str, label: str, prop: str, value: str) -> str:
        """Return the closest database value for ``label.prop`` to ``value``."""
        if not label or not prop:
            return value

        def fetch(cypher: str) -> List[str]:
            print(f"[matcher] running query: {cypher}")
            with self.driver.session(database=NEO4J_DB) as session:
                result = session.run(cypher)
                values = [r["val"] for r in result]
            print(f"[matcher] first rows: {values[:5]}")
            return values

        if kind.lower() == "relationship":
            cypher = (
                f"MATCH ()-[r:`{label}`]-() WHERE r.{prop} IS NOT NULL "
                f"RETURN DISTINCT toString(r.{prop}) AS val"
            )
        else:
            cypher = (
                f"MATCH (n:`{label}`) WHERE n.{prop} IS NOT NULL "
                f"RETURN DISTINCT toString(n.{prop}) AS val"
            )
        results = fetch(cypher)

        if not results:
            print(f"[matcher] no candidates found for {label}.{prop}")
            return value

        for candidate in results:
            if str(candidate).lower() == value.lower():
                print(f"[matcher] exact match for {value} -> {candidate}")
                return str(candidate)

        def lcs_length(a: str, b: str) -> int:
            a, b = a.lower(), b.lower()
            m, n = len(a), len(b)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            longest = 0
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if a[i - 1] == b[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                        if dp[i][j] > longest:
                            longest = dp[i][j]
            return longest

        scored = [
            (lcs_length(str(candidate), value), str(candidate))
            for candidate in results
        ]
        best_len, best_val = max(scored, key=lambda x: x[0])
        print(
            f"[matcher] best match for '{value}' -> '{best_val}' (LCS length {best_len})",
        )
        return best_val if best_len > 0 else value

    def match(self, description: str, schema: str) -> List[Dict[str, str]]:
        """Extract and resolve field-value pairs in description."""
        span = start_span(self.langfuse, "match", {"description": description, "schema": schema})
        pairs = self._extract_pairs(description, schema)
        print("[matcher] pairs after extraction:", pairs)
        resolved: List[Dict[str, str]] = []
        for pair in pairs:
            matched = self._match_value(
                pair["kind"], pair["label"], pair["property"], pair["value"]
            )
            resolved.append(
                {
                    "kind": pair["kind"],
                    "label": pair["label"],
                    "property": pair["property"],
                    "value": matched,
                }
            )
        print("[matcher] resolved pairs:", resolved)
        finish_span(span, {"extracted": pairs, "pairs": resolved})
        return resolved
