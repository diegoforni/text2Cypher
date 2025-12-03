"""Expansion agent that clarifies and enriches user requests."""
from typing import Optional, Tuple, Any
import json
import re

from langchain_core.language_models import BaseChatModel
from langfuse import Langfuse

from .langfuse_utils import start_span, finish_span


class ExpansionAgent:
    """Ask clarifying questions and produce a schema-grounded description.

    The constructor accepts `ask_when_ambiguous` for compatibility with
    `main.build_app`, which may request the agent prompt clarifying
    questions when the expanded result appears ambiguous.
    """

    def __init__(self, llm: BaseChatModel, langfuse: Optional[Langfuse] = None, ask_when_ambiguous: bool = False):
        self.llm = llm
        self.langfuse = langfuse
        # Whether the agent should actively ask clarifying questions when
        # it detects ambiguity. Currently reserved for higher-level logic;
        # stored for potential future use.
        self.ask_when_ambiguous = bool(ask_when_ambiguous)

    def expand(self, request: str, schema: str) -> Tuple[str, bool, str, bool, str]:
        system_message = (
            "You are a data analysis expert specializing in graph databases and cybersecurity data. "
            "Your job is ONLY to clarify the request and capture context for later query generation. "
            "Do NOT output or suggest any Cypher syntax."
        )
        prompt = f"""
Analyze this question and enrich it with contextual details. Use natural language only.

Question: {request}
Schema: {schema}

Provide a single JSON object with these keys (no code fences):
1. INTENT – what is the user trying to find out?
2. KEY_ENTITIES – which nodes/relationships are relevant?
3. FILTERS – which conditions or values must be matched?
4. OUTPUT_FORMAT – how should results be presented?
5. SUGGESTED_APPROACH – high-level strategy (no Cypher).
6. NEEDS_DECOMPOSITION – boolean. True only if multiple independent Cypher queries must be generated and composed; false if a single query suffices.
7. SINGLE_TASK – a concise single-sentence task to pass directly to the generator when NEEDS_DECOMPOSITION is false.
"""
        span = start_span(
            self.langfuse,
            "expand",
            {
                # Avoid duplicating large strings in telemetry; keep only final prompt
                "request": request,
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

        # Robustly parse JSON while keeping the original text as expanded output
        expanded_text = _coerce_text(raw).strip()
        # Strip code fences if any
        expanded_text = re.sub(r"^```[a-zA-Z]*\n", "", expanded_text)
        expanded_text = re.sub(r"\n```$", "", expanded_text).strip()

        # Try to extract the JSON object
        needs_decomposition = False
        single_task = request.strip() if isinstance(request, str) else ""
        obj_text = expanded_text
        start, end = obj_text.find("{"), obj_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj_text = obj_text[start : end + 1]
        try:
            data = json.loads(obj_text)
            # Extract fields with safe fallbacks
            nd_key = next((k for k in data.keys() if str(k).lower() == "needs_decomposition"), None)
            if nd_key is not None:
                needs_decomposition = bool(data.get(nd_key))
            st_key = next((k for k in data.keys() if str(k).lower() == "single_task"), None)
            if st_key is not None and isinstance(data.get(st_key), str) and data.get(st_key).strip():
                single_task = data.get(st_key).strip()
            else:
                # Fallback: reconstruct a sensible single-task from INTENT and FILTERS
                intent_key = next((k for k in data.keys() if str(k).lower() == "intent"), None)
                filters_key = next((k for k in data.keys() if str(k).lower() == "filters"), None)
                intent_txt = str(data.get(intent_key)).strip() if intent_key else ""
                filters_txt = data.get(filters_key)
                if isinstance(filters_txt, (list, tuple)):
                    filters_txt = ", ".join(str(x) for x in filters_txt if str(x).strip())
                filters_txt = str(filters_txt).strip() if filters_txt else ""
                parts = [p for p in [intent_txt, filters_txt] if p]
                if parts:
                    single_task = "; ".join(parts)
        except Exception:
            # If parsing fails, keep defaults
            pass

        finish_span(span, {
            "expanded": expanded_text,
            "needs_decomposition": needs_decomposition,
            "single_task": single_task,
        })
        # For backward/forward compatibility the expand method returns a
        # 5-tuple: (expanded_text, needs_decomposition, single_task,
        # clarification_needed, clarification_question).
        # The last two values default to False/"" unless the agent
        # explicitly detects a need for clarification.
        clarification_needed = False
        clarification_question = ""

        return (
            expanded_text,
            needs_decomposition,
            single_task,
            clarification_needed,
            clarification_question,
        )
