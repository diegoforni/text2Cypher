"""Console entry point for the multi-agent Cypher system."""
from typing import Any, Callable, List, Optional, TypedDict
import json
import logging
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langgraph.graph import StateGraph, END
from langfuse import Langfuse
from neo4j import GraphDatabase

from agents.expansion_agent import ExpansionAgent
from agents.decomposition_agent import DecompositionAgent
from agents.generation_agent import GenerationAgent
from agents.matcher_agent import MatcherAgent
from agents.validation_agent import ValidationAgent
from agents.composition_agent import CompositionAgent
from agents.langfuse_utils import start_span, finish_span
from config import (
    get_llm,
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_HOST,
    MODEL_PROVIDER,
    GEMINI_API_KEY,
)


logger = logging.getLogger(__name__)


class TokenCountingLLM:
    """Wrapper to accumulate token usage from LLM responses."""

    def __init__(self, llm: object):
        self._llm = llm
        self.input_tokens = 0
        self.output_tokens = 0

    def invoke(self, *args, **kwargs):
        response = self._llm.invoke(*args, **kwargs)
        metadata = getattr(response, "response_metadata", {}) or {}
        # LangChain providers vary: OpenAI uses usage_metadata {input_tokens, output_tokens},
        # some use token_usage {prompt_tokens, completion_tokens}, Gemini can differ.
        usage = metadata.get("token_usage") or metadata.get("usage") or {}
        if not usage:
            # LangChain also exposes a top-level usage_metadata on the message
            top_level_usage = getattr(response, "usage_metadata", None) or getattr(response, "usage", None)
            if isinstance(top_level_usage, dict):
                usage = top_level_usage

        if not usage and MODEL_PROVIDER == "gemini":
            try:
                import google.generativeai as genai
                # Prefer the actual key used by the underlying LLM (if provided)
                active_key = getattr(self._llm, "last_api_key", None) or GEMINI_API_KEY
                if active_key:
                    genai.configure(api_key=active_key)
                model_name = getattr(self._llm, "model", "gemini-1.5-pro")
                model = genai.GenerativeModel(model_name)

                prompt_data = args[0] if args else kwargs.get("messages") or kwargs.get("prompt") or ""
                if isinstance(prompt_data, list):
                    def _to_text(m):
                        if isinstance(m, tuple) and len(m) == 2:
                            return str(m[1])
                        if hasattr(m, "content"):
                            return str(m.content)
                        return str(m)

                    prompt_text = "\n".join(_to_text(m) for m in prompt_data)
                else:
                    prompt_text = str(prompt_data)

                output_text = getattr(response, "content", str(response))

                self.input_tokens += model.count_tokens(prompt_text).total_tokens
                self.output_tokens += model.count_tokens(output_text).total_tokens
                return response
            except Exception:
                usage = {}

        # Collect tokens across possible key names
        in_tok = (
            usage.get("prompt_tokens")
            or usage.get("prompt_token_count")
            or usage.get("input_tokens")
            or 0
        )
        out_tok = (
            usage.get("completion_tokens")
            or usage.get("candidates_token_count")
            or usage.get("output_tokens")
            or 0
        )
        # As a last resort, if only total_tokens is present, attribute them to input
        if not in_tok and not out_tok:
            total_only = usage.get("total_tokens")
            if isinstance(total_only, (int, float)):
                in_tok = int(total_only)
                out_tok = 0

        self.input_tokens += int(in_tok)
        self.output_tokens += int(out_tok)
        return response

    def __getattr__(self, name: str):
        return getattr(self._llm, name)


class GraphState(TypedDict, total=False):
    """State container passed between workflow nodes."""

    request: str
    schema: str
    expanded: str
    needs_decomposition: bool
    clarification_needed: bool
    clarification_question: str
    subproblems: List[str]
    fragments: List[str]
    generation_trace: List[dict]
    final_query: str
    results: Any
    explanation: str
    error: str
    final_validate_trace: List[dict]
    awaiting_clarification: bool


def build_app(
    llm: object,
    langfuse: Langfuse | None,
    progress_callback: Optional[Callable[[str, str, Optional[dict]], None]] = None,
    clarification_enabled: bool = False,
):
    """Create the LangGraph application wiring all agents."""
    if llm is None:
        raise RuntimeError(
            "LLM provider or API key not configured. Set MODEL_PROVIDER and API key environment variables."
        )
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    expander = ExpansionAgent(llm, langfuse, ask_when_ambiguous=clarification_enabled)
    decomposer = DecompositionAgent(llm, langfuse)
    matcher = MatcherAgent(llm, driver, langfuse)
    validator = ValidationAgent(driver, langfuse)
    composer = CompositionAgent(llm, langfuse)

    def emit(phase: str, status: str, payload: Optional[dict] = None) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback(phase, status, payload)
        except Exception:
            logger.debug("Progress callback failed for phase %s", phase, exc_info=True)

    def expand_node(state: GraphState):
        """LLM expansion step for the original request.

        Also decides whether decomposition is necessary. When decomposition is
        not needed, pre-populate a single subproblem to allow routing directly
        to generation.
        """
        emit("expand", "start", {"request": state.get("request")})
        (
            expanded,
            needs_decomp,
            single_task,
            clarification_needed,
            clarification_question,
        ) = expander.expand(
            state["request"], state["schema"]
        )
        out: GraphState = {
            "expanded": expanded,
            "needs_decomposition": needs_decomp,
            "clarification_needed": clarification_needed,
            "clarification_question": clarification_question,
        }
        if not needs_decomp:
            out["subproblems"] = [single_task or state["request"]]
            emit(
                "decompose",
                "skipped",
                {"reason": "Single task", "subproblems": out["subproblems"]},
            )
        emit(
            "expand",
            "complete",
            {
                "expanded": expanded,
                "needs_decomposition": needs_decomp,
                "clarification_needed": clarification_needed,
                "clarification_question": clarification_question,
            },
        )
        return out

    def clarification_node(state: GraphState):
        """Pause the workflow when clarification is required before continuing."""
        question = state.get("clarification_question")
        emit(
            "clarify",
            "start",
            {
                "clarification_needed": state.get("clarification_needed", False),
                "clarification_question": question,
            },
        )
        message = "Clarification needed before generating Cypher. Please answer the follow-up question."
        return {
            "awaiting_clarification": True,
            "error": message,
        }

    def decompose_node(state: GraphState):
        """Break the expanded request into subproblems."""
        emit("decompose", "start", {"expanded": state.get("expanded")})
        subproblems = decomposer.decompose(state["expanded"], state["schema"])
        emit("decompose", "complete", {"subproblems": subproblems})
        return {"subproblems": subproblems}

    def generate_node(state: GraphState):
        """Create and validate fragments for each subproblem in parallel.

        Optimization:
        - Only invoke the matcher when the generated fragment appears to use
          specific literal values (e.g., quoted strings in property filters).
        """

        import re

        _IDENT = r"(?:`[^`]+`|[A-Za-z_][A-Za-z0-9_`]*)"
        _EXACT_VALUE_PATTERN = re.compile(
            rf"""
            (?P<alias>{_IDENT})\.(?P<prop>{_IDENT})\s*=\s*
            (
                (?P<quote>['\"])(?P<literal>[^'\"$]+)(?P=quote)
                |
                (?P<number>-?\d+(?:\.\d+)?)
            )
            """,
            flags=re.IGNORECASE | re.VERBOSE,
        )
        _RELATIONSHIP_MAP_PATTERN = re.compile(
            r"\[(?P<alias>{id})?\s*:(?P<label>{id})\s*\{{(?P<body>[^\}]*)\}\]".replace("{id}", _IDENT),
            flags=re.IGNORECASE,
        )
        _NODE_MAP_PATTERN = re.compile(
            r"\((?P<alias>{id})?\s*:(?P<label>{id})\s*\{{(?P<body>[^\}]*)\}\)".replace("{id}", _IDENT),
            flags=re.IGNORECASE,
        )
        _MAP_ENTRY_PATTERN = re.compile(
            r"(?P<prop>{id})\s*:\s*(?P<quote>['\"])(?P<literal>[^'\"]+)(?P=quote)".replace("{id}", _IDENT),
            flags=re.IGNORECASE,
        )

        def _normalize_identifier(value: str) -> str:
            return value.replace("`", "").lower()

        def _find_exact_value_literals(fragment: str) -> list[dict[str, object]]:
            if not fragment:
                return []
            matches: list[dict[str, object]] = []
            for m in _EXACT_VALUE_PATTERN.finditer(fragment):
                literal = m.group("literal")
                number = m.group("number")
                value = literal if literal is not None else number
                if value is None:
                    continue
                alias = m.group("alias")
                prop = m.group("prop")
                if literal is not None:
                    start, end = m.start("literal"), m.end("literal")
                    quote = m.group("quote") or ""
                else:
                    start, end = m.start("number"), m.end("number")
                    quote = ""
                matches.append(
                    {
                        "alias": alias,
                        "alias_norm": _normalize_identifier(alias),
                        "property": prop,
                        "property_norm": _normalize_identifier(prop),
                        "literal": value,
                        "quote": quote,
                        "start": start,
                        "end": end,
                        "label": "",
                        "label_norm": "",
                        "kind": "comparison",
                    }
                )

            def _collect_map_matches(pattern: re.Pattern, kind: str) -> None:
                for match in pattern.finditer(fragment):
                    body = match.group("body") or ""
                    if not body:
                        continue
                    alias = match.group("alias") or ""
                    label = match.group("label") or ""
                    base_norm = _normalize_identifier(alias) or _normalize_identifier(label)
                    label_norm = _normalize_identifier(label)
                    body_start = match.start("body")
                    for entry in _MAP_ENTRY_PATTERN.finditer(body):
                        literal = entry.group("literal")
                        if literal is None:
                            continue
                        quote = entry.group("quote") or ""
                        prop = entry.group("prop") or ""
                        start = body_start + entry.start("literal")
                        end = body_start + entry.end("literal")
                        matches.append(
                            {
                                "alias": alias,
                                "alias_norm": base_norm,
                                "label": label,
                                "label_norm": label_norm,
                                "property": prop,
                                "property_norm": _normalize_identifier(prop),
                                "literal": literal,
                                "quote": quote,
                                "start": start,
                                "end": end,
                                "kind": kind,
                            }
                        )

            _collect_map_matches(_NODE_MAP_PATTERN, "node")
            _collect_map_matches(_RELATIONSHIP_MAP_PATTERN, "relationship")
            return matches

        def _apply_exact_replacements(fragment: str, replacements: list[tuple[dict[str, object], str]]) -> str:
            updated = fragment
            # Replace from the end to keep earlier indices valid
            for match_info, new_value in sorted(replacements, key=lambda item: int(item[0]["start"]), reverse=True):
                start = int(match_info["start"])
                end = int(match_info["end"])
                updated = updated[:start] + new_value + updated[end:]
            return updated

        subproblems = state.get("subproblems") or [state.get("expanded") or state.get("request", "")]
        emit("generate", "start", {"subproblems": subproblems})
        previous = [""] * len(subproblems)
        errors = [""] * len(subproblems)
        fragments: List[str | None] = [None] * len(subproblems)
        verified_pairs: List[List[dict]] = [[] for _ in subproblems]
        gen_trace: List[dict] = [
            {"subproblem": sub, "attempts": []} for sub in subproblems
        ]
        # If we are in single-task mode, capture the validated rows to avoid re-validation
        single_mode = len(subproblems) == 1
        single_rows: Any | None = None

        for _ in range(3):
            pending = [i for i, f in enumerate(fragments) if f is None]
            if not pending:
                break

            def generate(idx: int) -> tuple[int, str, List[dict]]:
                sub = subproblems[idx]
                local_generator = GenerationAgent(llm, langfuse)
                prompt = sub
                if previous[idx]:
                    prompt += (
                        f"\nPrevious fragment:\n{previous[idx]}\n"
                        f"Error: {errors[idx]}\n"
                        "Please fix and regenerate."
                    )
                fragment = local_generator.generate(
                    prompt, state["schema"], pairs=verified_pairs[idx]
                )
                # After generation, verify any literal values only if clearly needed
                literal_matches = _find_exact_value_literals(fragment)
                pairs: List[dict] = []
                if literal_matches:
                    matched_pairs = matcher.match(fragment, state["schema"])
                    replacements: list[tuple[dict[str, object], str]] = []
                    selected_pairs: List[dict] = []

                    for literal_match in literal_matches:
                        for pair in matched_pairs:
                            prop_norm = _normalize_identifier(pair.get("property", ""))
                            label_norm = _normalize_identifier(pair.get("label", ""))
                            pair_original = str(pair.get("original", ""))
                            literal_value = str(literal_match["literal"])
                            labels_match = True
                            match_label_norm = str(literal_match.get("label_norm", ""))
                            if match_label_norm and label_norm:
                                labels_match = label_norm == match_label_norm
                            if (
                                prop_norm == literal_match["property_norm"]
                                and pair_original.strip() == literal_value.strip()
                                and labels_match
                            ):
                                replacements.append((literal_match, str(pair.get("value", literal_value))))
                                if pair not in selected_pairs:
                                    selected_pairs.append(pair)
                                break

                    if replacements:
                        fragment = _apply_exact_replacements(fragment, replacements)
                        pairs = selected_pairs
                return idx, fragment, pairs

            with ThreadPoolExecutor(max_workers=len(pending)) as executor:
                generated = list(executor.map(generate, pending))

            indices = [i for i, _, _ in generated]
            candidates = [frag for _, frag, _ in generated]
            pair_lists = [p for _, _, p in generated]
            results = validator.validate_many(candidates)

            for idx, fragment, pairs, (ok, result) in zip(
                indices, candidates, pair_lists, results
            ):
                # Record attempt trace
                gen_trace[idx]["attempts"].append({
                    "fragment": fragment,
                    "matched_pairs": pairs,
                    "ok": bool(ok),
                    "error": None if ok else str(result),
                    "rows_preview": result[:3] if ok and isinstance(result, list) else None,
                })
                if ok:
                    fragments[idx] = fragment
                    if single_mode:
                        single_rows = result
                else:
                    previous[idx] = fragment
                    errors[idx] = str(result)
                    if pairs:
                        verified_pairs[idx] = pairs

        final_fragments = [f for f in fragments if f]
        out: GraphState = {"fragments": final_fragments, "generation_trace": gen_trace}
        # Optimization: if there is only one validated fragment, we already have its rows;
        # surface them now so we can skip the final validation pass.
        if single_mode and len(final_fragments) == 1 and single_rows is not None:
            out["results"] = single_rows
            emit(
                "generate",
                "complete",
                {
                    "fragments": final_fragments,
                    "validated_rows": len(single_rows) if isinstance(single_rows, list) else None,
                },
            )
        else:
            emit("generate", "complete", {"fragments": final_fragments})
        return out

    def compose_node(state: GraphState):
        """Combine validated fragments into a final query.

        Optimization: if only one fragment exists, bypass composer.
        """
        emit("compose", "start", {"fragments": state.get("fragments")})
        frags = state["fragments"]
        if len(frags) == 1:
            emit("compose", "complete", {"final_query": frags[0], "reason": "Single fragment"})
            return {"final_query": frags[0]}
        query = composer.compose(frags)
        emit("compose", "complete", {"final_query": query})
        return {"final_query": query}

    def final_validate_node(state: GraphState):
        """Run a final validation pass over the composed query with retries.

        If single-task mode already produced rows, skip re-validating. Otherwise,
        attempt to validate the composed query; on failure, invoke the composer to
        refine the query using the database error and retry up to 2 more times.
        """
        if state.get("results") is not None:
            emit(
                "final_validate",
                "skipped",
                {"reason": "Results already validated", "results": state.get("results")},
            )
            return {}
        query = state["final_query"]
        fragments = state.get("fragments", [])
        last_error: str | None = None
        fv_trace: List[dict] = []
        emit("final_validate", "start", {"query": query})
        for _ in range(3):
            ok, res = validator.validate(query)
            fv_trace.append({
                "query": query,
                "ok": bool(ok),
                "error": None if ok else str(res),
                "rows_preview": res[:3] if ok and isinstance(res, list) else None,
            })
            if ok:
                # Succeed with potentially refined query
                emit(
                    "final_validate",
                    "success",
                    {
                        "query": query,
                        "row_count": len(res) if isinstance(res, list) else None,
                        "results": res,
                    },
                )
                return {"results": res, "final_query": query, "final_validate_trace": fv_trace}
            last_error = str(res)
            # Try refining only if there are fragments and we have an error
            try:
                query = composer.refine(fragments, query, last_error, state["schema"]) if fragments else query
            except Exception:
                # If refine fails (e.g., LLM issues), break and surface original error
                break
        # If we reach here, we failed all attempts
        emit(
            "final_validate",
            "error",
            {"query": query, "error": last_error or "Validation failed"},
        )
        return {"error": last_error or "Validation failed", "final_validate_trace": fv_trace}

    def explain_node(state: GraphState):
        """Generate a human-readable explanation of the final query."""
        if "error" in state:
            return {}
        emit("explain", "start", {"query": state.get("final_query")})
        explanation = composer.explain(state["final_query"], state["schema"])
        emit("explain", "complete", {"explanation": explanation})
        return {"explanation": explanation}

    workflow = StateGraph(GraphState)
    workflow.add_node("expand", expand_node)
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("decompose", decompose_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("compose", compose_node)
    workflow.add_node("final_validate", final_validate_node)
    workflow.add_node("explain", explain_node)

    # Route from expand -> decompose or generate based on expander's decision
    def _route_after_expand(s: GraphState) -> str:
        if s.get("clarification_needed"):
            return "clarification"
        return "decompose" if s.get("needs_decomposition") else "generate"

    # langgraph supports conditional edges; map route keys to node names
    workflow.add_conditional_edges(
        "expand",
        _route_after_expand,
        {"clarification": "clarification", "decompose": "decompose", "generate": "generate"},
    )
    workflow.add_edge("clarification", END)
    workflow.add_edge("decompose", "generate")
    workflow.add_edge("generate", "compose")
    workflow.add_edge("compose", "final_validate")
    workflow.add_edge("final_validate", "explain")
    workflow.set_entry_point("expand")
    workflow.add_edge("explain", END)

    return workflow.compile()


def save_run(question: str, result: GraphState, llm: object) -> None:
    """Save the question, generated Cypher, and database response to JSON."""
    metadata = {
        "time": datetime.now(timezone.utc).isoformat(),
        "model": getattr(llm, "model_name", getattr(llm, "model", None)),
        "provider": MODEL_PROVIDER,
        "commit": None,
    }
    try:
        metadata["commit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        )
    except Exception:
        pass
    usage = {
        "input": getattr(llm, "input_tokens", 0),
        "output": getattr(llm, "output_tokens", 0),
    }
    usage["total"] = usage["input"] + usage["output"]
    agents = {
        "expansion": {
            "expanded": result.get("expanded"),
            "needs_decomposition": result.get("needs_decomposition"),
            "clarification_needed": result.get("clarification_needed"),
            "clarification_question": result.get("clarification_question"),
        },
        "decomposition": {
            "subproblems": result.get("subproblems"),
        },
        "generation": {
            "fragments": result.get("fragments"),
            "trace": result.get("generation_trace"),
        },
        "final_validation": {
            "trace": result.get("final_validate_trace"),
            "error": result.get("error"),
            "results_preview": (result.get("results")[:3] if isinstance(result.get("results"), list) else None),
        },
        "explanation": {
            "text": result.get("explanation"),
        },
    }
    payload = {
        "question": question,
        "cypher": result.get("final_query"),
        "response": result.get("results"),
        "error": result.get("error"),
        "agents": agents,
        "token_usage": usage,
        "metadata": metadata,
        "clarification_needed": result.get("clarification_needed"),
        "clarification_question": result.get("clarification_question"),
    }
    payload_text = json.dumps(payload, indent=2)

    targets = [Path("last_run.json")]
    fallback_dir = os.getenv("LAST_RUN_DIR")
    if fallback_dir:
        targets.append(Path(fallback_dir) / "last_run.json")
    targets.append(Path(tempfile.gettempdir()) / "text2cypher_last_run.json")

    for idx, path in enumerate(targets):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Parent may already exist or cannot be created; continue to attempt write
            pass
        try:
            path.write_text(payload_text, encoding="utf-8")
            if idx > 0:
                logger.warning("last_run.json not writable; saved fallback copy at %s", path)
            break
        except PermissionError:
            if idx == len(targets) - 1:
                logger.error("Unable to persist last_run.json due to permission errors")
        except OSError as exc:
            if idx == len(targets) - 1:
                logger.error("Unable to persist last_run.json: %s", exc)


def run(
    question: str,
    schema: str,
    progress_callback: Optional[Callable[[str, str, Optional[dict]], None]] = None,
    clarification_enabled: bool = False,
) -> GraphState:
    """Execute the agent workflow for ``question`` against ``schema``."""
    llm = get_llm()
    if llm is None:
        raise RuntimeError(
            "LLM provider or API key not configured. Set MODEL_PROVIDER and API key environment variables."
        )
    llm = TokenCountingLLM(llm)
    langfuse_client = None
    trace = None
    if LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY:
        langfuse_client = Langfuse(
            secret_key=LANGFUSE_SECRET_KEY,
            public_key=LANGFUSE_PUBLIC_KEY,
            host=LANGFUSE_HOST,
        )
        trace = start_span(langfuse_client, "run", {"request": question, "schema": schema})
    app = build_app(llm, trace, progress_callback, clarification_enabled=clarification_enabled)
    inputs: GraphState = {"request": question, "schema": schema}
    try:
        result = app.invoke(inputs)
        save_run(question, result, llm)
        finish_span(trace, {"result": result})
        return result
    except Exception as e:  # pragma: no cover - simple passthrough
        save_run(question, {"error": str(e)}, llm)
        finish_span(trace, error=e)
        raise


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit(1)

    schema = (
        "Nodes: (:IP {address, botnet, pulse_ids}), (:Country {name}), (:ASN {number})\n"
        "Relationships: (ip:IP)-[:ATTACKS {technique, protocol, technology, industry, pulse_id, "
        "documented_create_date, documented_modified_date}]->(country:Country)\n"
        "Additional: (ip:IP)-[:FROM]->(country:Country), (ip:IP)-[:BELONGS_TO]->(asn:ASN)"
    )
    run(sys.argv[1], schema)
