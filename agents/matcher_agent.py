"""Matcher agent that resolves literal values against the database."""
from __future__ import annotations

from typing import Dict, List, Optional
import json
from concurrent.futures import ThreadPoolExecutor

from langchain_core.language_models import BaseChatModel
from neo4j import Driver
from langfuse import Langfuse

from config import NEO4J_DB
from .langfuse_utils import start_span, finish_span


class MatcherAgent:
    """Extract literal values from Cypher and match them to database values."""

    def __init__(self, llm: BaseChatModel, driver: Driver, langfuse: Optional[Langfuse] = None):
        self.llm = llm
        self.driver = driver
        self.langfuse = langfuse

    def _extract_pairs(self, query: str, schema: str) -> List[Dict[str, str]]:
        """Use the LLM to extract label/property/value triples from a Cypher query."""
        system_message = (
            "You are given a Cypher query for a Neo4j database. "
            "Identify literal string or numeric values that appear directly in node or relationship property comparisons. "
            "Ignore numbers used for LIMIT, SKIP, or unrelated arithmetic. "
            "For each literal, return the element kind ('node' or 'relationship'), its label or relationship type, "
            "the property name, and the exact value from the query. "
            "If the query has no such literals, return an empty JSON array."
        )
        prompt = f"""
Schema:\n{schema}\n
Query:\n{query}\n
List every literal value used in a property comparison in the query with its element type, label or relationship type, and property name. If none are present, return [].
Return a JSON array like:
[
  {{"kind": "node", "label": "Person", "property": "name", "value": "Tom"}}
]
Use the exact value from the query.
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

    def match(self, query: str, schema: str) -> List[Dict[str, str]]:
        """Extract and resolve literal property values in ``query``.

        Returns a list of dictionaries with the original extracted value under
        ``original`` and the database-resolved value under ``value``.
        """
        span = start_span(
            self.langfuse, "match", {"query": query, "schema": schema}
        )
        extracted = self._extract_pairs(query, schema)
        print("[matcher] pairs after extraction:", extracted)

        def resolve(pair: Dict[str, str]) -> Dict[str, str]:
            matched = self._match_value(
                pair["kind"], pair["label"], pair["property"], pair["value"]
            )
            return {
                "kind": pair["kind"],
                "label": pair["label"],
                "property": pair["property"],
                "original": pair["value"],
                "value": matched,
            }

        with ThreadPoolExecutor(max_workers=len(extracted) or 1) as executor:
            resolved = list(executor.map(resolve, extracted))
        print("[matcher] resolved pairs:", resolved)
        finish_span(span, {"extracted": extracted, "pairs": resolved})
        return resolved
