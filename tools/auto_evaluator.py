"""
Auxiliary runner to repeatedly evaluate all prompts in `queries.json`.

Features:
- Enforces at most one run per hour per query.
- Stops when token budget (in tokens) is exhausted or when each query has been run N times (default 50).
- Persists per-query state in `tools/auto_state.json` and writes run records to `tools/auto_evaluation_results.json`.
- Supports a --dry-run mode to validate behavior without calling the LLM or Neo4j.

Usage (dry-run):
  python tools/auto_evaluator.py --dry-run

Usage (real run, with token budget):
  TOTAL_TOKEN_BUDGET=100000 python tools/auto_evaluator.py --max-per-query 50

Notes:
- This script calls `main.run` from the project which expects MODEL_PROVIDER and API keys to be configured.
  If those are not set, prefer --dry-run for testing.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Ensure project root importable
import sys
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv()

from config import MODEL_PROVIDER  # type: ignore
from main import run as agent_run  # type: ignore


STATE_FILE = Path("tools/auto_state.json")
RESULTS_FILE = Path("tools/auto_evaluation_results.json")
LAST_RUN_FILE = Path("last_run.json")
QUERIES_FILE = Path("queries.json")


def load_queries() -> List[Dict[str, Any]]:
    if not QUERIES_FILE.exists():
        raise FileNotFoundError("queries.json not found")
    return json.loads(QUERIES_FILE.read_text(encoding="utf-8"))


def load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"per_query": {}, "summary": {}}


def save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def append_result(entry: Dict[str, Any]) -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    existing: List[Dict[str, Any]] = []
    if RESULTS_FILE.exists():
        try:
            existing = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    existing.append(entry)
    RESULTS_FILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-query", type=int, default=50, help="Max runs per query")
    parser.add_argument("--budget-tokens", type=int, default=int(os.getenv("TOTAL_TOKEN_BUDGET") or 0), help="Total token budget (tokens). 0 means unlimited if --dry-run is not used cautiously.")
    parser.add_argument("--dry-run", action="store_true", help="Do everything except call the LLM/DB; useful for testing state and schedule logic")
    parser.add_argument("--hourly-window-minutes", type=int, default=60, help="Min minutes between runs of the same query (default 60)")
    parser.add_argument("--poll-seconds", type=int, default=60, help="Seconds to wait between scheduling checks when no query is eligible")
    args = parser.parse_args()

    try:
        prompts = load_queries()
    except Exception as e:
        print("Error loading queries.json:", e)
        return 2

    state = load_state()
    per_query = state.get("per_query", {})

    # Initialize per-query slots
    for idx, item in enumerate(prompts, start=1):
        key = str(idx)
        if key not in per_query:
            per_query[key] = {"count": 0, "last_run": None}

    state["per_query"] = per_query
    save_state(state)

    remaining_tokens = int(args.budget_tokens) if args.budget_tokens and args.budget_tokens > 0 else None
    max_per_query = int(args.max_per_query)
    hourly_window = timedelta(minutes=args.hourly_window_minutes)

    print(f"Starting auto-evaluator: dry_run={args.dry_run} max_per_query={max_per_query} budget_tokens={remaining_tokens}")

    # Primary loop
    while True:
        all_done = True
        any_eligible = False
        for idx, item in enumerate(prompts, start=1):
            key = str(idx)
            meta = per_query.get(key, {})
            count = int(meta.get("count", 0))
            last_run_iso = meta.get("last_run")
            last_run_dt = parse_iso(last_run_iso)

            if count >= max_per_query:
                continue
            all_done = False

            # Check hourly constraint
            now = datetime.now(timezone.utc)
            if last_run_dt and (now - last_run_dt) < hourly_window:
                continue

            # If token budget is exhausted, exit
            if remaining_tokens is not None and remaining_tokens <= 0:
                print("Token budget exhausted; stopping.")
                save_state(state)
                return 0

            any_eligible = True
            prompt = item.get("prompt")
            category = item.get("category")
            expected = item.get("cypher")

            print(f"Running query [{idx}] (count={count+1}/{max_per_query}) category={category} prompt={prompt}")

            started = iso_now()
            result_entry: Dict[str, Any] = {
                "index": idx,
                "category": category,
                "prompt": prompt,
                "expected_cypher": expected,
                "started_at": started,
                "metadata": {"provider": MODEL_PROVIDER},
            }

            if args.dry_run:
                # Simulate a run without invoking LLM/DB
                result_entry.update({
                    "agent_cypher": None,
                    "response": None,
                    "error": "dry-run",
                    "token_usage": {"input": 0, "output": 0, "total": 0},
                    "duration_seconds": 0,
                    "ended_at": iso_now(),
                })
                append_result(result_entry)
                per_query[key]["count"] = count + 1
                per_query[key]["last_run"] = iso_now()
                save_state(state)
                # Short pause between prompts in dry-run to avoid tight loops
                time.sleep(0.1)
                continue

            # Real run: call agent_run
            try:
                # Call the agent which will save last_run.json
                state_result = agent_run(prompt, expected)
            except Exception as e:
                ended = iso_now()
                result_entry.update({
                    "agent_cypher": None,
                    "response": None,
                    "error": str(e),
                    "token_usage": None,
                    "duration_seconds": None,
                    "ended_at": ended,
                })
                append_result(result_entry)
                per_query[key]["count"] = count + 1
                per_query[key]["last_run"] = iso_now()
                save_state(state)
                # If provider is Gemini, respect the 60s sleep like evaluator.py did
                if MODEL_PROVIDER and MODEL_PROVIDER.lower() == "gemini":
                    print("Sleeping 60s for Gemini rate limits...")
                    time.sleep(60)
                continue

            # After successful run, attempt to read last_run.json for token usage
            token_usage = None
            if LAST_RUN_FILE.exists():
                try:
                    lr = json.loads(LAST_RUN_FILE.read_text(encoding="utf-8"))
                    token_usage = lr.get("token_usage") or lr.get("tokenUsage") or {}
                except Exception:
                    token_usage = None

            ended = iso_now()
            duration = None
            try:
                # If state_result returned timing info, compute approximate durations
                duration = None
            except Exception:
                duration = None

            result_entry.update({
                "agent_cypher": state_result.get("final_query"),
                "response": state_result.get("results"),
                "error": state_result.get("error"),
                "agents": {
                    "expansion": {"expanded": state_result.get("expanded")},
                    "generation": {"fragments": state_result.get("fragments")},
                    "final_validation": {"trace": state_result.get("final_validate_trace")},
                },
                "token_usage": token_usage,
                "duration_seconds": duration,
                "ended_at": ended,
            })

            append_result(result_entry)

            # Deduct tokens from remaining budget if present
            if remaining_tokens is not None and token_usage:
                tok_total = None
                if isinstance(token_usage, dict):
                    tok_total = token_usage.get("total") or token_usage.get("input") or 0
                    # If input/output present, sum them
                    if token_usage.get("input") is not None and token_usage.get("output") is not None:
                        tok_total = int(token_usage.get("input", 0)) + int(token_usage.get("output", 0))
                try:
                    tok_val = int(tok_total or 0)
                except Exception:
                    tok_val = 0
                remaining_tokens -= tok_val
                print(f"Consumed {tok_val} tokens; remaining budget={remaining_tokens}")

            per_query[key]["count"] = count + 1
            per_query[key]["last_run"] = iso_now()
            save_state(state)

            # Model-specific backoff
            if MODEL_PROVIDER and MODEL_PROVIDER.lower() == "gemini":
                print("Sleeping 60s for Gemini rate limits...")
                time.sleep(60)

        if all_done:
            print("All queries reached max runs. Exiting.")
            break

        if not any_eligible:
            print(f"No eligible queries right now. Sleeping {args.poll_seconds}s...")
            time.sleep(args.poll_seconds)
            continue

    print("Auto-evaluator finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
