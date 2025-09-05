"""Generation agent that creates Cypher fragments for each subproblem."""
from __future__ import annotations

from typing import List, Optional, Any
import json

from langchain_core.language_models import BaseChatModel
from langfuse import Langfuse

from .langfuse_utils import start_span, finish_span


class GenerationAgent:
    """Generate Cypher query fragments using verified field values."""

    def __init__(self, llm: BaseChatModel, langfuse: Optional[Langfuse] = None):
        self.llm = llm
        self.langfuse = langfuse

    def generate(
        self, subproblem: str, schema: str, pairs: Optional[List[dict]] = None
    ) -> str:
        """Produce a Cypher fragment solving ``subproblem`` using ``schema`` and ``pairs``.

        ``pairs`` is optional and defaults to an empty list when no values have been
        pre-verified.
        """
        pairs = pairs or []
        system_message = (
            "You are a Cypher query expert. Use the provided database schema to solve the given subproblem. "
            "Always produce a complete Cypher fragment ending with a RETURN clause. "
            "Do not include explanations or commentary. "
            "Do not wrap your answer in code fences. "
            "CRITICAL: Do not nest MATCH inside WHERE clauses. "
            "Allowed clauses: MATCH, WHERE, WITH, RETURN, ORDER BY, LIMIT. Do NOT use APOC. "
            "Properties like 'technique' or 'protocol' are on relationships [:ATTACKS {technique: \"value\"}]. "
            "Attack relationships follow the pattern (ip:IP)-[:ATTACKS]->(country:Country). "
            "Use WITH or multiple MATCH statements for complex logic and preserve provided values exactly. "
            "When the subproblem text mentions a literal value, look it up in the verified field values list "
            "and use the verified form in the Cypher fragment. "
            "TEMPORAL RULES: Many date properties (e.g., a.documented_create_date) are arrays of strings. "
            "To derive calendar year-month (YYYY-MM), always use substring(toString(<prop>[0]),0,7). "
            "Do NOT call datetime() on an array value and do NOT use left(); prefer substring()+toString() as shown. "
            "When comparing months across two relationships, compare substring(toString(r1.documented_create_date[0]),0,7) "
            "to substring(toString(r2.documented_create_date[0]),0,7). Name the grouped field year_month. "
            "SCOPING RULES: After a WITH clause only referenced variables remain in scope; if you didn't carry a variable forward, "
            "use count(*) instead of count(r). "
            "PAIR GENERATION: When forming ASN pairs within a group, first collect DISTINCT ASNs per group, then UNWIND ranges "
            "to create combinations (avoid Cartesian products)."
        )
        # Sanitize subproblem to avoid leaking prior fragments/errors or fenced code
        cleaned_subproblem = subproblem or ""
        # Remove common noise blocks
        for marker in ["Previous fragment:", "Error:"]:
            if marker in cleaned_subproblem:
                cleaned_subproblem = cleaned_subproblem.split(marker, 1)[0].strip()
        # Drop fenced code blocks
        cleaned_subproblem = cleaned_subproblem.replace("```cypher", "```")
        while "```" in cleaned_subproblem:
            pre, _sep, rest = cleaned_subproblem.partition("```")
            _code, _sep2, post = rest.partition("```")
            if _sep2:
                cleaned_subproblem = (pre + post).strip()
            else:
                # unmatched fence; break to avoid infinite loop
                break
        # Remove stray inline schema headings leaking from upstream prompts
        for heading in ["Database schema:", "Schema:"]:
            if heading in cleaned_subproblem:
                cleaned_subproblem = cleaned_subproblem.replace(heading, "").strip()

        # Build the prompt without repeating optional sections
        prompt_parts = [
            f"Database schema:\n{schema}",
            f"Subproblem:\n{cleaned_subproblem}",
        ]
        if pairs:  # include only when non-empty to avoid redundant input
            prompt_parts.append(f"Verified field values:\n{json.dumps(pairs)}")
        prompt = "\n\n".join(prompt_parts)
        span = start_span(
            self.langfuse,
            "generate",
            {
                # Avoid duplicating large strings in telemetry; keep only final prompt
                "system": system_message,
                "prompt": prompt,
            },
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

        fragment = _coerce_text(raw)
        # Strip code fences if the model ignored instructions
        cleaned = fragment.strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("cypher"):
                cleaned = cleaned[6:].strip()
        fragment = cleaned
        finish_span(span, {"fragment": fragment})
        return fragment
