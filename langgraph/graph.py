"""Minimal shim of `langgraph.graph` used by text2Cypher.

This provides a very small, synchronous StateGraph implementation with
conditional edges sufficient for running the example workflow in
`main.py`. It's intentionally lightweight and not a full replacement for
the original library.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple

END = "__LG_END__"


class StateGraph:
    def __init__(self, _state_type: Optional[type] = None):
        # nodes: name -> callable(state) -> dict
        self._nodes: Dict[str, Callable[[dict], dict]] = {}
        # edges: from -> list[to]
        self._edges: Dict[str, List[str]] = {}
        # conditional edges: name -> (chooser_callable, mapping_dict)
        self._conditional: Dict[str, Tuple[Callable[[dict], str], Dict[str, str]]] = {}
        self._entry: Optional[str] = None

    def add_node(self, name: str, func: Callable[[dict], dict]) -> None:
        self._nodes[name] = func

    def add_edge(self, a: str, b: str) -> None:
        self._edges.setdefault(a, []).append(b)

    def add_conditional_edges(self, name: str, chooser: Callable[[dict], str], mapping: Dict[str, str]) -> None:
        self._conditional[name] = (chooser, mapping)

    def set_entry_point(self, name: str) -> None:
        self._entry = name

    def compile(self) -> "CompiledWorkflow":
        if not self._entry:
            raise RuntimeError("Entry point not set on StateGraph")
        return CompiledWorkflow(self._nodes, self._edges, self._conditional, self._entry)


class CompiledWorkflow:
    def __init__(self, nodes: Dict[str, Callable[[dict], dict]], edges: Dict[str, List[str]],
                 conditional: Dict[str, Tuple[Callable[[dict], str], Dict[str, str]]], entry: str):
        self._nodes = nodes
        self._edges = edges
        self._conditional = conditional
        self._entry = entry

    def invoke(self, state: dict) -> dict:
        # We expect state to be a dict-like object; mutate and return it.
        current = self._entry
        # Defensive copy in case callers re-use the same dict
        if state is None:
            state = {}
        while True:
            if current == END or current == "END" or current is None:
                break
            node = self._nodes.get(current)
            if node is None:
                raise RuntimeError(f"Unknown node '{current}' in workflow")
            # Call node with current state
            out = node(state) or {}
            if not isinstance(out, dict):
                # Support nodes that return other types (treat as no-op)
                out = {}
            # Merge node output into state
            state.update(out)

            # Decide next node
            if current in self._conditional:
                chooser, mapping = self._conditional[current]
                key = chooser(state)
                next_name = mapping.get(key)
                if next_name is None:
                    # If chooser returned an unmapped key, try to stop
                    break
            else:
                nexts = self._edges.get(current, [])
                if not nexts:
                    break
                next_name = nexts[0]

            if next_name == END or next_name == "END":
                break
            current = next_name

        return state
