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
            "Only split the problem when multiple queries must be composed together. "
            "Return only a JSON array. Do not use code fences."
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
            {
                # Avoid duplicating large strings in telemetry; keep only final prompt
                "description": description,
                "system": system_message,
                "prompt": prompt,
            },
        )
        response = self.llm.invoke([
            ("system", system_message),
            ("user", prompt),
        ])
        text = response.content if hasattr(response, "content") else str(response)
        # Debug: show raw model output (trimmed)
        try:
            preview = text if len(text) < 1000 else text[:1000] + "... [truncated]"
            print("[decompose] LLM response:", preview)
        except Exception:
            pass

        # Robust parsing: strip code fences, then try to extract the JSON array
        cleaned = text.strip()
        cleaned = re.sub(r"^```[a-zA-Z]*\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned).strip()

        # Some models wrap the array in prose; extract the outermost brackets if present
        start, end = cleaned.find("["), cleaned.rfind("]")
        candidate = cleaned[start : end + 1] if start != -1 and end != -1 else cleaned
        try:
            preview = candidate if len(candidate) < 1000 else candidate[:1000] + "... [truncated]"
            print("[decompose] candidate JSON text:", preview)
        except Exception:
            pass

        subproblems: List[str]
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                subproblems = [p.strip() for p in data if isinstance(p, str) and p.strip()]
            else:
                # Not a list; fall back to original description
                subproblems = [description]
        except Exception:
            # If the model echoed the instruction or failed to return JSON, use the original description
            if "From the analysis below" in cleaned:
                subproblems = [description]
            else:
                # Last resort: split by lines/bullets, otherwise keep as single task
                lines = [ln.strip("- •\t ") for ln in cleaned.splitlines() if ln.strip()]
                subproblems = lines or [description]

        # Guard: if the model echoed the instruction inside a parsed list or produced nothing, fallback
        INSTRUCTION_PHRASE = "From the analysis below"
        if (
            not subproblems
            or all(INSTRUCTION_PHRASE.lower() in s.lower() for s in subproblems)
        ):
            subproblems = [description]

        try:
            print("[decompose] parsed subproblems:", subproblems)
        except Exception:
            pass

        finish_span(span, {"response": text, "subproblems": subproblems})
        return subproblems
