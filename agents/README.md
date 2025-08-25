# Agents

This package contains the coordinated agents that transform a natural
language question into an executable Cypher query.

1. **ExpansionAgent** – clarifies the user request and enriches it with
   domain context.
2. **DecompositionAgent** – splits the enriched description into smaller
   independent tasks.
3. **MatcherAgent** – extracts literal field/value pairs and resolves them
   against existing values in the Neo4j database.
4. **GenerationAgent** – produces a Cypher fragment for each task while
   respecting the verified values and schema.
5. **ValidationAgent** – runs the fragment and returns either rows or the
   error that occurred during execution.
6. **CompositionAgent** – combines validated fragments into the final
   query and optionally generates a short explanation.

Each agent has a single public method representing its responsibility,
which keeps the pipeline easy to follow and test. The agents share a
minimal tracing interface so callers can observe intermediate steps.
