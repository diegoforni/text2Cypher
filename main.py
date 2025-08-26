"""Console entry point for the multi-agent Cypher system."""
from typing import List, TypedDict, Any
import json
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

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
)


class TokenCountingLLM:
    """Wrapper to accumulate token usage from LLM responses."""

    def __init__(self, llm: object):
        self._llm = llm
        self.input_tokens = 0
        self.output_tokens = 0

    def invoke(self, *args, **kwargs):
        response = self._llm.invoke(*args, **kwargs)
        metadata = getattr(response, "response_metadata", {}) or {}
        usage = metadata.get("token_usage") or metadata.get("usage_metadata") or {}
        self.input_tokens += usage.get("prompt_tokens") or usage.get("prompt_token_count") or 0
        self.output_tokens += (
            usage.get("completion_tokens")
            or usage.get("candidates_token_count")
            or 0
        )
        return response

    def __getattr__(self, name: str):
        return getattr(self._llm, name)


class GraphState(TypedDict, total=False):
    """State container passed between workflow nodes."""

    request: str
    schema: str
    expanded: str
    subproblems: List[str]
    fragments: List[str]
    final_query: str
    results: Any
    explanation: str
    error: str


def build_app(llm: object, langfuse: Langfuse | None):
    """Create the LangGraph application wiring all agents."""
    if llm is None:
        raise RuntimeError(
            "LLM provider or API key not configured. Set MODEL_PROVIDER and API key environment variables."
        )
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    expander = ExpansionAgent(llm, langfuse)
    decomposer = DecompositionAgent(llm, langfuse)
    matcher = MatcherAgent(llm, driver, langfuse)
    validator = ValidationAgent(driver, langfuse)
    composer = CompositionAgent(llm, langfuse)

    def expand_node(state: GraphState):
        """LLM expansion step for the original request."""
        expanded = expander.expand(state["request"], state["schema"])
        return {"expanded": expanded}

    def decompose_node(state: GraphState):
        """Break the expanded request into subproblems."""
        subproblems = decomposer.decompose(state["expanded"])
        return {"subproblems": subproblems}

    def generate_node(state: GraphState):
        """Create and validate fragments for each subproblem in parallel."""

        def handle_subproblem(sub: str) -> str | None:
            local_generator = GenerationAgent(llm, langfuse)
            pairs = matcher.match(sub, state["schema"])
            previous_fragment = ""
            error_message = ""
            for _ in range(3):
                prompt = sub
                if previous_fragment:
                    prompt += (
                        f"\nPrevious fragment:\n{previous_fragment}\n"
                        f"Error: {error_message}\n"
                        "Please fix and regenerate."
                    )
                fragment = local_generator.generate(prompt, state["schema"], pairs)
                ok, result = validator.validate(fragment)
                if ok:
                    return fragment
                previous_fragment = fragment
                error_message = str(result)
            return None

        with ThreadPoolExecutor(max_workers=len(state["subproblems"])) as executor:
            results = list(executor.map(handle_subproblem, state["subproblems"]))
        fragments = [r for r in results if r]
        return {"fragments": fragments}

    def compose_node(state: GraphState):
        """Combine validated fragments into a final query."""
        query = composer.compose(state["fragments"])
        return {"final_query": query}

    def final_validate_node(state: GraphState):
        """Run a final validation pass over the composed query."""
        ok, res = validator.validate(state["final_query"])
        if ok:
            return {"results": res}
        return {"error": res}

    def explain_node(state: GraphState):
        """Generate a human-readable explanation of the final query."""
        if "error" in state:
            return {}
        explanation = composer.explain(state["final_query"], state["schema"])
        return {"explanation": explanation}

    workflow = StateGraph(GraphState)
    workflow.add_node("expand", expand_node)
    workflow.add_node("decompose", decompose_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("compose", compose_node)
    workflow.add_node("final_validate", final_validate_node)
    workflow.add_node("explain", explain_node)

    workflow.add_edge("expand", "decompose")
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
        "time": datetime.utcnow().isoformat() + "Z",
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
    payload = {
        "question": question,
        "cypher": result.get("final_query"),
        "response": result.get("results"),
        "error": result.get("error"),
        "token_usage": usage,
        "metadata": metadata,
    }
    with open("last_run.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run(question: str, schema: str) -> GraphState:
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
    app = build_app(llm, trace)
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
