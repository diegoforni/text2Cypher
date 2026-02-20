"""Matcher agent that resolves literal values against the database."""
from __future__ import annotations

from typing import Dict, List, Optional, Any
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
        import sys
        print("=" * 80, file=sys.stderr)
        print("[MATCHER] _extract_pairs called", file=sys.stderr)
        print(f"[MATCHER] Query: {query[:200]}...", file=sys.stderr)
        print(f"[MATCHER] Schema: {schema[:200]}...", file=sys.stderr)

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
        print("[MATCHER] Calling LLM...", file=sys.stderr)
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

        text = _coerce_text(raw)
        print(f"[MATCHER] LLM raw response: {text}", file=sys.stderr)
        try:
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()
            start, end = cleaned.find("["), cleaned.rfind("]")
            if start != -1 and end != -1:
                cleaned = cleaned[start : end + 1]
            print(f"[MATCHER] Cleaned JSON: {cleaned}", file=sys.stderr)
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
            print(f"[MATCHER] ✓ Extracted {len(parsed)} pairs: {parsed}", file=sys.stderr)
            finish_span(span, {"response": text, "pairs": parsed})
            print("=" * 80, file=sys.stderr)
            return parsed
        except Exception as e:
            print(f"[MATCHER] ✗ Failed to parse pairs: {e}", file=sys.stderr)
            print(f"[MATCHER] Raw text was: {text[:500]}", file=sys.stderr)
            finish_span(span, {"response": text}, e)
            print("=" * 80, file=sys.stderr)
            return []

    def _match_value(self, kind: str, label: str, prop: str, value: str) -> str:
        """Return the closest database value for ``label.prop`` to ``value``."""
        import sys
        print(f"[MATCHER] _match_value called: kind={kind}, label={label}, prop={prop}, value={value}", file=sys.stderr)

        if not label or not prop:
            print(f"[MATCHER] No label or prop, returning original value", file=sys.stderr)
            return value

        def fetch(cypher: str) -> List[str]:
            print(f"[MATCHER] Running query: {cypher}", file=sys.stderr)
            with self.driver.session(database=NEO4J_DB) as session:
                result = session.run(cypher)
                values = [r["val"] for r in result]
            print(f"[MATCHER] Query returned {len(values)} values", file=sys.stderr)
            if values:
                print(f"[MATCHER] First 5 values: {values[:5]}", file=sys.stderr)
            return values

        if kind.lower() == "relationship":
            cypher = (
                f"MATCH ()-[r:`{label}`]-() WHERE r.{prop} IS NOT NULL "
                f"RETURN DISTINCT r.{prop} AS val"
            )
        else:
            cypher = (
                f"MATCH (n:`{label}`) WHERE n.{prop} IS NOT NULL "
                f"RETURN DISTINCT n.{prop} AS val"
            )
        results = fetch(cypher)

        # Flatten list-valued properties into individual candidates and
        # normalize everything to strings for matching.
        import sys
        flat: List[str] = []
        for v in results:
            if isinstance(v, list):
                flat.extend([str(x) for x in v])
            else:
                flat.append(str(v))
        # Deduplicate while preserving order
        seen = set()
        deduped: List[str] = []
        for s in flat:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        results = deduped

        if not results:
            print(f"[MATCHER] ✗ No candidates found for {label}.{prop}", file=sys.stderr)
            return value

        print(f"[MATCHER] Looking for exact match (case-insensitive): '{value}' in {len(results)} candidates", file=sys.stderr)
        for candidate in results:
            if str(candidate).lower() == value.lower():
                print(f"[MATCHER] ✓ Exact match found: '{value}' -> '{candidate}'", file=sys.stderr)
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

        import sys
        print(f"[MATCHER] No exact match, trying LCS matching...", file=sys.stderr)
        scored = [
            (lcs_length(str(candidate), value), str(candidate))
            for candidate in results
        ]
        best_len, best_val = max(scored, key=lambda x: x[0])
        print(f"[MATCHER] LCS best match: '{value}' -> '{best_val}' (LCS length {best_len})", file=sys.stderr)
        if best_len > 0:
            print(f"[MATCHER] ✓ Using LCS match: '{best_val}'", file=sys.stderr)
        else:
            print(f"[MATCHER] ✗ No good match found, using original value: '{value}'", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        return best_val if best_len > 0 else value

    def match(self, query: str, schema: str) -> List[Dict[str, str]]:
        """Extract and resolve literal property values in ``query``.

        Returns a list of dictionaries with the original extracted value under
        ``original`` and the database-resolved value under ``value``.
        """
        import sys
        print("=" * 80, file=sys.stderr)
        print("[MATCHER] match() called", file=sys.stderr)
        print(f"[MATCHER] Query: {query[:200]}...", file=sys.stderr)
        span = start_span(
            self.langfuse, "match", {"query": query, "schema": schema}
        )
        extracted = self._extract_pairs(query, schema)
        print(f"[MATCHER] Extraction complete. Got {len(extracted)} pairs", file=sys.stderr)

        def resolve(pair: Dict[str, str]) -> Dict[str, str]:
            print(f"[MATCHER] Resolving pair: {pair}", file=sys.stderr)
            matched = self._match_value(
                pair["kind"], pair["label"], pair["property"], pair["value"]
            )
            result = {
                "kind": pair["kind"],
                "label": pair["label"],
                "property": pair["property"],
                "original": pair["value"],
                "value": matched,
            }
            print(f"[MATCHER] Resolved -> {result}", file=sys.stderr)
            return result

        with ThreadPoolExecutor(max_workers=len(extracted) or 1) as executor:
            resolved = list(executor.map(resolve, extracted))
        print(f"[MATCHER] All pairs resolved: {resolved}", file=sys.stderr)
        print(f"[MATCHER] match() complete, returning {len(resolved)} pairs", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        finish_span(span, {"extracted": extracted, "pairs": resolved})
        return resolved
