#!/usr/bin/env python3
"""
Run a single evaluation with per-query timeout.

This modifies the evaluator to:
1. Run each query with a 10-minute timeout
2. On timeout, save partial results (last generated Cypher)
3. Continue to the next query instead of failing the entire run
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import multiprocessing
import queue as _queue

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from config import MODEL_PROVIDER
from main import run as agent_run, Neo4jEncoder

SCHEMA = (
    "Nodes: (:IP {address, botnet, pulse_ids}), (:Country {name}), (:ASN {number})\n"
    "Relationships: (ip:IP)-[:ATTACKS {technique, protocol, technology, industry, pulse_id, "
    "documented_create_date, documented_modified_date}]->(country:Country)\n"
    "Additional: (ip:IP)-[:FROM]->(country:Country), (ip:IP)-[:BELONGS_TO]->(asn:ASN)"
)


def _git_commit() -> str | None:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def run_query_with_timeout(prompt: str, schema: str, timeout_seconds: int = 600) -> Dict[str, Any]:
    """Run a single query with timeout and return state or partial results."""
    def _run_child(prompt_inner: str, schema_inner: str, out_q: multiprocessing.Queue) -> None:
        try:
            st = agent_run(prompt_inner, schema_inner)
            out_q.put({"ok": True, "state": st})
        except Exception as exc:
            try:
                out_q.put({"ok": False, "error": str(exc)})
            except Exception:
                pass

    ctx = multiprocessing.get_context("fork")
    q: multiprocessing.Queue = ctx.Queue()
    p = ctx.Process(target=_run_child, args=(prompt, schema, q))

    start_time = time.time()
    p.start()
    p.join(timeout_seconds)

    elapsed = time.time() - start_time

    if p.is_alive():
        # Timeout - try to get partial results from last_run.json
        print(f"    Query timeout after {elapsed:.1f}s. Attempting to recover partial results...")
        try:
            p.terminate()
        except Exception:
            pass
        p.join(5)

        # Try to read partial state from last_run.json
        last_run_file = PROJECT_ROOT / "last_run.json"
        partial_state = {"error": f"Timeout after {timeout_seconds}s"}

        if last_run_file.exists():
            try:
                lr_data = json.loads(last_run_file.read_text())
                # Extract whatever we have
                partial_state["final_query"] = lr_data.get("state", {}).get("final_query")
                partial_state["results"] = lr_data.get("state", {}).get("results")
                partial_state["error"] = f"Timeout after {timeout_seconds}s (partial Cypher recovered)"
                partial_state["partial"] = True
                print(f"    Recovered partial Cypher: {partial_state.get('final_query', 'None')[:100]}...")
            except Exception as e:
                print(f"    Could not recover partial results: {e}")

        return partial_state

    # Process finished - get result
    try:
        res = q.get_nowait()
    except _queue.Empty:
        return {"error": "Process exited without returning state"}

    if isinstance(res, dict) and res.get("ok"):
        return res.get("state") or {}
    else:
        error = res.get("error") if isinstance(res, dict) else str(res)
        return {"error": error}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="evaluation_results.json", help="Output file")
    parser.add_argument("--queries", default="queries.json", help="Queries JSON file")
    parser.add_argument("--query-timeout", type=int, default=600, help="Timeout per query in seconds")
    args = parser.parse_args()

    queries_path = PROJECT_ROOT / args.queries
    if not queries_path.exists():
        print("queries.json not found")
        return 1

    prompts: List[Dict[str, Any]] = json.loads(queries_path.read_text())
    started = datetime.now(timezone.utc)
    commit = _git_commit()

    results = []
    print(f"Starting evaluation with {len(prompts)} queries")
    print(f"Per-query timeout: {args.query_timeout}s ({args.query_timeout // 60} minutes)")
    print(f"Model: {MODEL_PROVIDER}\n")

    for i, item in enumerate(prompts, start=1):
        category = item.get("category")
        prompt = item.get("prompt")
        expected_cypher = item.get("cypher")

        print(f"[{i}/{len(prompts)}] {category} :: {prompt}")
        t0 = time.perf_counter()

        # Run query with timeout
        state = run_query_with_timeout(prompt, SCHEMA, timeout_seconds=args.query_timeout)
        duration = time.perf_counter() - t0

        # Load token usage from last_run.json
        last_run_file = PROJECT_ROOT / "last_run.json"
        token_usage = None
        metadata = None
        if last_run_file.exists():
            try:
                lr = json.loads(last_run_file.read_text())
                token_usage = lr.get("token_usage")
                metadata = lr.get("metadata")
            except Exception:
                pass

        result_row = {
            "index": i,
            "category": category,
            "prompt": prompt,
            "expected_cypher": expected_cypher,
            "agent_cypher": state.get("final_query"),
            "response": state.get("results"),
            "error": state.get("error"),
            "partial": state.get("partial", False),
            "token_usage": token_usage,
            "duration_seconds": round(duration, 3),
            "metadata": metadata or {
                "provider": MODEL_PROVIDER,
                "commit": commit,
            },
        }

        results.append(result_row)

        status = "TIMEOUT (partial)" if state.get("partial") else "ERROR" if state.get("error") else "SUCCESS"
        print(f"  -> {status} in {duration:.1f}s\n")

    ended = datetime.now(timezone.utc)

    # Calculate summary
    total_tokens = sum(r.get("token_usage", {}).get("total", 0) or 0 for r in results)
    summary = {
        "total_queries": len(results),
        "successful": sum(1 for r in results if not r.get("error")),
        "errors": sum(1 for r in results if r.get("error")),
        "timeouts": sum(1 for r in results if r.get("partial")),
        "total_tokens": total_tokens,
        "total_duration_seconds": (ended - started).total_seconds(),
        "model": MODEL_PROVIDER,
        "commit": commit,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
    }

    output_data = {
        "summary": summary,
        "results": results
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(output_data, indent=2, cls=Neo4jEncoder))

    print(f"\n{'='*80}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total queries: {summary['total_queries']}")
    print(f"Successful: {summary['successful']}")
    print(f"Errors: {summary['errors']}")
    print(f"Timeouts (partial): {summary['timeouts']}")
    print(f"Total tokens: {summary['total_tokens']:,}")
    print(f"Duration: {summary['total_duration_seconds'] / 60:.1f} minutes")
    print(f"\nResults saved to: {output_path}")
    print(f"{'='*80}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
