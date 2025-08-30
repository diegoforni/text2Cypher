"""
Run end-to-end evaluations for all prompts in queries.json using the full agent.

For each prompt, this script:
- Invokes the full agent pipeline (Expansion → Decomposition → Generation → Validation → Composition → Final validation → Explain)
- Captures the final Cypher, DB results or error, token usage, timing, and metadata
- Sleeps 60 seconds between queries if MODEL_PROVIDER == 'gemini' (to respect rate limits)
- Uses Langfuse tracing automatically when LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY are set in .env
- Writes a single JSON file with all results

Usage:
  python tools/evaluator.py [--output evaluation_results.json]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

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


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--output",
        default="evaluation_results.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()

    queries_path = Path("queries.json")
    if not queries_path.exists():
        print("queries.json not found")
        return 1
    prompts: List[Dict[str, Any]] = json.loads(queries_path.read_text())

    started = datetime.now(timezone.utc)
    commit = _git_commit()

    results: List[Dict[str, Any]] = []
    for i, item in enumerate(prompts, start=1):
        category = item.get("category")
        prompt = item.get("prompt")
        expected_cypher = item.get("cypher")
        print(f"\n[{i}/{len(prompts)}] Running agent for: {category} :: {prompt}")

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
            except Exception:
                pass

        result_row: Dict[str, Any] = {
            "index": i,
            "category": category,
            "prompt": prompt,
            "expected_cypher": expected_cypher,
            "agent_cypher": state.get("final_query"),
            "response": state.get("results"),
            "error": state.get("error"),
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
            print("Model provider is Gemini. Sleeping 60 seconds before next prompt...")
            time.sleep(60)

    ended = datetime.now(timezone.utc)

    # Write all results as a single JSON array to keep parity with queries.json
    out_path = Path(args.output)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved evaluation results to {out_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
