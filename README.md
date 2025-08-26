# Text2Cypher

Multi‑agent pipeline that converts natural‑language questions into validated Neo4j Cypher, with optional explanation and tracing.

## Project Structure

- `agents/`: pipeline components (see `agents/README.md`).
- `config.py`: environment + model selection.
- `main.py`: LangGraph workflow entrypoint and CLI.
- `proof_of_concept.py`: early reference script.

## Setup

- Install dependencies: `pip install -r requirements.txt`
- Create a `.env` (see `.env.example`) with:
  - `MODEL_PROVIDER`: `openai` or `gemini`
  - `OPENAI_API_KEY` or `GEMINI_API_KEY`
  - `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DB`
  - Optional Langfuse: `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_HOST`

Default chat models (as configured in `config.py`):
- OpenAI: `gpt-5-nano`
- Gemini: `gemini-2.5-flash`

## Usage

Ensure Neo4j is running and credentials are correct. Then run:

```bash
python main.py "Who attacked Country X?"
```

What happens:
- Expands and decomposes the request, generates candidate fragments, validates against Neo4j, composes a final query, and optionally explains it.
- Persists the last run to `last_run.json` (final Cypher, results/error, token usage, metadata).
- If Langfuse env vars are set, spans are emitted for each agent step.

## Notes

- The schema used for `main.py` is inlined for convenience; swap it with yours or load dynamically as needed.
- Inputs to each agent avoid redundant sections (e.g., optional blocks only included when populated) to minimize token usage without losing context.
