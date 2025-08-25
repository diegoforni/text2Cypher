# Text2Cypher

This project uses multiple agents to translate natural language questions into Neo4j Cypher queries with optional validation and explanation.

## Project structure

- `agents/` – multi-step pipeline components (see [agents/README.md](agents/README.md)).
- `config.py` – configuration helpers for environment and models.
- `main.py` – command-line entry point that orchestrates the agents.
- `proof_of_concept.py` – early script retained for reference.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure environment variables in a `.env` file (an example is provided in `.env.example`):
   ```bash
   MODEL_PROVIDER=openai  # or gemini
   OPENAI_API_KEY=your-openai-key
   GEMINI_API_KEY=your-gemini-key
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=neo4j
   NEO4J_DB=neo4j
   LANGFUSE_SECRET_KEY=
   LANGFUSE_PUBLIC_KEY=
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

## Usage

Ensure your Neo4j instance is running and the `.env` values match your environment. Run the console entry point with a question:

```bash
python main.py "Who attacked Country X?"
```

The system will attempt to generate and validate a Cypher query, printing the query, explanation, and results.
