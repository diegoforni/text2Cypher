"""Console entry point for the multi-agent Cypher system."""
from typing import List, TypedDict, Any

from langgraph.graph import StateGraph, END
from langfuse import Langfuse
from neo4j import GraphDatabase

from agents.expansion_agent import ExpansionAgent
from agents.decomposition_agent import DecompositionAgent
from agents.generation_agent import GenerationAgent
from agents.validation_agent import ValidationAgent
from agents.composition_agent import CompositionAgent
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
    llm = get_llm()
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    expander = ExpansionAgent(llm, langfuse)
    decomposer = DecompositionAgent(llm, langfuse)
    generator = GenerationAgent(llm, langfuse)
    validator = ValidationAgent(driver, langfuse)
    composer = CompositionAgent(llm, langfuse)

    def expand_node(state: GraphState):
        return {"expanded": expander.expand(state["request"], state["schema"]) }

    def decompose_node(state: GraphState):
        return {"subproblems": decomposer.decompose(state["expanded"])}

    def generate_node(state: GraphState):
        fragments: List[str] = []
        for sub in state["subproblems"]:
            feedback = ""
            for _ in range(3):  # initial + 2 retries
                fragment = generator.generate(f"{sub}\n{feedback}", state["schema"])
                ok, result = validator.validate(fragment)
                if ok:
                    fragments.append(fragment)
                    break
                feedback = f"Previous error: {result}. Please fix and regenerate."
        return {"fragments": fragments}

    def compose_node(state: GraphState):
        query = composer.compose(state["fragments"])
        return {"final_query": query}

    def final_validate_node(state: GraphState):
        ok, res = validator.validate(state["final_query"])
        if ok:
            return {"results": res}
        return {"error": res}

    def explain_node(state: GraphState):
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


def run(question: str, schema: str) -> GraphState:
    langfuse = None
    if LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY:
        langfuse = Langfuse(
            secret_key=LANGFUSE_SECRET_KEY,
            public_key=LANGFUSE_PUBLIC_KEY,
            host=LANGFUSE_HOST,
        )
    app = build_app(langfuse)
    inputs: GraphState = {"request": question, "schema": schema}
    return app.invoke(inputs)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python main.py 'your question'")
        raise SystemExit(1)

    schema = """Nodes and relationships are provided by the Neo4j database schema."""
    output = run(sys.argv[1], schema)
    if output.get("error"):
        print("Validation failed:", output["error"])
    else:
        print("Cypher query:\n", output["final_query"])
        print("Explanation:", output.get("explanation"))
        print("Results:", output.get("results"))
