"""
Run end-to-end evaluations for all prompts in queries.json using the full agent.

For each prompt, this script:
- Invokes the full agent pipeline (Expansion → Decomposition → Generation → Validation → Composition → Final validation → Explain)
- Captures the final Cypher, DB results or error, token usage, timing, and metadata
- Sleeps 60 seconds between queries if MODEL_PROVIDER == 'gemini' (to respect rate limits)
- Uses Langfuse tracing automatically when LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY are set in .env
- Writes a single JSON file with all results

Usage:
  python tools/evaluator.py [--output evaluation_results.json] [--runs N]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Ensure project root is importable when running from tools/
import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env early so model provider, Langfuse and Neo4j creds are visible
load_dotenv()

from config import MODEL_PROVIDER  # type: ignore
from main import run as agent_run  # type: ignore


# Default schema matching main.py CLI usage
SCHEMA = (
    "Nodes: (:IP {address, botnet, pulse_ids}), (:Country {name}), (:ASN {number})\n"
    "Relationships: (ip:IP)-[:ATTACKS {technique, protocol, technology, industry, pulse_id, "
    "documented_create_date, documented_modified_date}]->(country:Country)\n"
    "Additional: (ip:IP)-[:FROM]->(country:Country), (ip:IP)-[:BELONGS_TO]->(asn:ASN)"
)


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        )
    except Exception:
        return None


def _sanitize_filename_part(value: str) -> str:
    """Make a string safe to use in filenames.

    Keep letters, numbers, dash and underscore; replace others with '-'.
    Collapse consecutive dashes.
    """
    import re

    safe = re.sub(r"[^A-Za-z0-9_-]+", "-", value or "")
    safe = re.sub(r"-+", "-", safe).strip("-")
    return safe or "model"


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--output",
        default="evaluation_results.json",
        help="Output JSON file path (used when --runs=1)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of times to run the full evaluation (default: 1)",
    )
    args = parser.parse_args()

    queries_path = Path("queries.json")
    if not queries_path.exists():
        print("queries.json not found")
        return 1
    prompts: List[Dict[str, Any]] = json.loads(queries_path.read_text())
    total_runs = max(1, int(args.runs))

    for run_idx in range(1, total_runs + 1):
        started = datetime.now(timezone.utc)
        commit = _git_commit()
        model_name: Optional[str] = None

        results: List[Dict[str, Any]] = []
        for i, item in enumerate(prompts, start=1):
            category = item.get("category")
            prompt = item.get("prompt")
            expected_cypher = item.get("cypher")
            print(f"\n(run {run_idx}/{total_runs}) [{i}/{len(prompts)}] {category} :: {prompt}")

            # Time the agent run
            t0 = time.perf_counter()
            t0_wall = datetime.now(timezone.utc)
            try:
                state = agent_run(prompt, SCHEMA)
            except Exception as e:
                # Attempt to pull token usage and metadata from last_run.json even if agent failed
                state = {"error": str(e)}
            t1_wall = datetime.now(timezone.utc)
            duration = time.perf_counter() - t0

            # Load last_run.json produced by main.run for token usage and metadata
            last_run_file = Path("last_run.json")
            token_usage: Dict[str, Any] | None = None
            metadata: Dict[str, Any] | None = None
            if last_run_file.exists():
                try:
                    lr = json.loads(last_run_file.read_text())
                    token_usage = lr.get("token_usage")
                    metadata = lr.get("metadata")
                    # Capture model name once for filename suffix
                    if not model_name:
                        model_name = (metadata or {}).get("model")
                except Exception:
                    pass

            # Assemble per-agent trace for downstream analysis
            agents: Dict[str, Any] = {
                "expansion": {
                    "expanded": state.get("expanded"),
                    "needs_decomposition": state.get("needs_decomposition"),
                },
                "decomposition": {
                    "subproblems": state.get("subproblems"),
                },
                "generation": {
                    "fragments": state.get("fragments"),
                    "trace": state.get("generation_trace"),
                },
                "composition": {
                    "final_query": state.get("final_query"),
                },
                "final_validation": {
                    "trace": state.get("final_validate_trace"),
                    "error": state.get("error"),
                    "results_preview": (state.get("results")[:3] if isinstance(state.get("results"), list) else None),
                },
                "explanation": {
                    "text": state.get("explanation"),
                },
            }

            result_row: Dict[str, Any] = {
                "index": i,
                "category": category,
                "prompt": prompt,
                "expected_cypher": expected_cypher,
                "agent_cypher": state.get("final_query"),
                "response": state.get("results"),
                "error": state.get("error"),
                "agents": agents,
                "token_usage": token_usage,
                "duration_seconds": round(duration, 3),
                "started_at": t0_wall.isoformat(),
                "ended_at": t1_wall.isoformat(),
                "metadata": metadata,
            }
            # Ensure commit/model/provider are present even if last_run metadata missing
            if not result_row.get("metadata"):
                result_row["metadata"] = {
                    "time": t1_wall.isoformat(),
                    "provider": MODEL_PROVIDER,
                    "commit": commit,
                }
            else:
                result_row["metadata"].setdefault("provider", MODEL_PROVIDER)
                result_row["metadata"].setdefault("commit", commit)

            # Indicate whether Langfuse tracing is enabled and which host
            langfuse_enabled = bool(os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY"))
            result_row["metadata"]["langfuse"] = {
                "enabled": langfuse_enabled,
                "host": os.getenv("LANGFUSE_HOST"),
            }

            results.append(result_row)

            # Sleep for Gemini to respect rate limits
            if MODEL_PROVIDER.lower() == "gemini" and i < len(prompts):
                print("Model provider is Gemini. Sleeping 60 seconds before next prompt...\n")
                time.sleep(60)

        ended = datetime.now(timezone.utc)

        # Determine output filename
        if total_runs == 1:
            out_path = Path(args.output)
        else:
            # Prefer model name from metadata; fall back to provider
            model_part = _sanitize_filename_part(model_name or MODEL_PROVIDER)
            out_path = Path(f"ev_res_{model_part}_{run_idx}.json")

        # Write all results as a single JSON array to keep parity with queries.json
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nSaved evaluation results to {out_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
