"""Console entry point for the multi-agent Cypher system."""
from typing import List, TypedDict, Any

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
)


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


def build_app(langfuse: Langfuse | None):
    """Create the LangGraph application wiring all agents."""
    llm = get_llm()
    if llm is None:
        raise RuntimeError(
            "LLM provider or API key not configured. Set MODEL_PROVIDER and API key environment variables."
        )
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    expander = ExpansionAgent(llm, langfuse)
    decomposer = DecompositionAgent(llm, langfuse)
    matcher = MatcherAgent(llm, driver, langfuse)
    generator = GenerationAgent(llm, langfuse)
    validator = ValidationAgent(driver, langfuse)
    composer = CompositionAgent(llm, langfuse)

    def expand_node(state: GraphState):
        """LLM expansion step for the original request."""
        print("[expand] request:", state["request"])
        expanded = expander.expand(state["request"], state["schema"])
        print("[expand] expanded:", expanded)
        return {"expanded": expanded}

    def decompose_node(state: GraphState):
        """Break the expanded request into subproblems."""
        print("[decompose] expanded:", state["expanded"])
        subproblems = decomposer.decompose(state["expanded"])
        print("[decompose] subproblems:", subproblems)
        return {"subproblems": subproblems}

    def generate_node(state: GraphState):
        """Create and validate fragments for each subproblem."""
        print("[generate] subproblems:", state["subproblems"])
        fragments: List[str] = []
        for sub in state["subproblems"]:
            pairs = matcher.match(sub, state["schema"])
            previous_fragment = ""
            error_message = ""
            for _ in range(3):  # initial + 2 retries
                prompt = sub
                if previous_fragment:
                    prompt += (
                        f"\nPrevious fragment:\n{previous_fragment}\n"
                        f"Error: {error_message}\n"
                        "Please fix and regenerate."
                    )
                fragment = generator.generate(prompt, state["schema"], pairs)
                ok, result = validator.validate(fragment)
                if ok:
                    fragments.append(fragment)
                    break
                previous_fragment = fragment
                error_message = str(result)
        print("[generate] fragments:", fragments)
        return {"fragments": fragments}

    def compose_node(state: GraphState):
        """Combine validated fragments into a final query."""
        print("[compose] fragments:", state["fragments"])
        query = composer.compose(state["fragments"])
        print("[compose] final_query:", query)
        return {"final_query": query}

    def final_validate_node(state: GraphState):
        """Run a final validation pass over the composed query."""
        print("[final_validate] final_query:", state["final_query"])
        ok, res = validator.validate(state["final_query"])
        print("[final_validate] ok:", ok, "result:", res)
        if ok:
            return {"results": res}
        return {"error": res}

    def explain_node(state: GraphState):
        """Generate a human-readable explanation of the final query."""
        if "error" in state:
            print("[explain] skipping due to error")
            return {}
        explanation = composer.explain(state["final_query"], state["schema"])
        print("[explain] explanation:", explanation)
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


def run(question: str, schema: str) -> GraphState:
    """Execute the agent workflow for ``question`` against ``schema``."""
    langfuse_client = None
    trace = None
    if LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY:
        print("[run] Initializing Langfuse client")
        langfuse_client = Langfuse(
            secret_key=LANGFUSE_SECRET_KEY,
            public_key=LANGFUSE_PUBLIC_KEY,
            host=LANGFUSE_HOST,
        )
        trace = start_span(langfuse_client, "run", {"request": question, "schema": schema})
    else:
        print("[run] Langfuse credentials not provided")
    app = build_app(trace)
    inputs: GraphState = {"request": question, "schema": schema}
    print("[run] inputs:", inputs)
    try:
        result = app.invoke(inputs)
        print("[run] result:", result)
        finish_span(trace, {"result": result})
        return result
    except Exception as e:  # pragma: no cover - simple passthrough
        finish_span(trace, error=e)
        raise


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python main.py 'your question'")
        raise SystemExit(1)

    schema = (
        "Nodes: (:IP {address, botnet, pulse_ids}), (:Country {name}), (:ASN {number})\n"
        "Relationships: (ip:IP)-[:ATTACKS {technique, protocol, technology, industry, pulse_id, "
        "documented_create_date, documented_modified_date}]->(country:Country)\n"
        "Additional: (ip:IP)-[:FROM]->(country:Country), (ip:IP)-[:BELONGS_TO]->(asn:ASN)"
    )
    output = run(sys.argv[1], schema)
    if output.get("error"):
        print("Validation failed:", output["error"])
    else:
        print("Cypher query:\n", output["final_query"])
        print("Explanation:", output.get("explanation"))
        print("Results:", output.get("results"))
