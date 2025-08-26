"""Decomposition agent that splits expanded descriptions into subproblems."""
from typing import List, Optional
import json
import re

from langchain_core.language_models import BaseChatModel
from langfuse import Langfuse

from .langfuse_utils import start_span, finish_span


class DecompositionAgent:
    """Break a problem description into independent subproblems."""

    def __init__(self, llm: BaseChatModel, langfuse: Optional[Langfuse] = None):
        self.llm = llm
        self.langfuse = langfuse

    def decompose(self, description: str, schema: str) -> List[str]:
        system_message = (
            "You are a task planner for graph analysis. "
            "Use the provided database schema to reason about the minimal set of "
            "independent natural-language tasks needed to build the final Cypher query. "
            "Only split the problem when multiple queries must be composed together."
        )
        prompt = f"""
Schema:\n{schema}\n\n"""
        prompt += (
            "From the analysis below, determine whether more than one independent query is required.\n"
            "If so, list each query as a concise natural-language task. Otherwise, return a single-item list with the original problem.\n\n"
            f"Analysis: {description}\n\n"
            "Return a JSON array of task strings using the original values verbatim."
        )
        span = start_span(
            self.langfuse,
            "decompose",
            {"description": description, "schema": schema, "system": system_message, "prompt": prompt},
        )
        response = self.llm.invoke([
            ("system", system_message),
            ("user", prompt),
        ])
        text = response.content if hasattr(response, "content") else str(response)
        cleaned = re.sub(r"^```[a-zA-Z]*\n", "", text.strip())
        cleaned = re.sub(r"\n```$", "", cleaned).strip()
        try:
            data = json.loads(cleaned)
            subproblems = [p.strip() for p in data if isinstance(p, str) and p.strip()]
        except Exception:
            subproblems = [cleaned]
        finish_span(span, {"response": text, "subproblems": subproblems})
        return subproblems

