# Agents

Coordinated agents transform a natural‑language question into an executable, validated Cypher query.

1. **ExpansionAgent**: clarifies the request and captures context (no Cypher in output).
2. **DecompositionAgent**: splits the expanded description into independent sub‑tasks (JSON array of strings).
3. **GenerationAgent**: produces a Cypher fragment per sub‑task, honoring verified values and schema rules.
4. **ValidationAgent**: executes fragments against Neo4j and returns either rows or an error.
5. **CompositionAgent**: concatenates validated fragments into the final query and can produce a short explanation.
6. **MatcherAgent**: extracts literal values from generated Cypher and resolves them against actual DB values. It is invoked only when the generated fragment contains explicit equality comparisons (e.g., `n.prop = "Value"` or `[:REL {prop: "Value"}]`), ensuring matcher calls are focused on exact lookups.

Design highlights:
- Single, focused public method per agent for clarity and testability.
- Optional Langfuse spans around all steps for traceability.
- Inputs avoid repeated content: optional sections (e.g., verified values) are only included when present to reduce token usage.
