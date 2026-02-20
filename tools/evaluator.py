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
import multiprocessing
import queue as _queue

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
from main import run as agent_run, Neo4jEncoder  # type: ignore


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


def _classify_error(error_msg: str | None) -> str:
    """Classify the type of error for better analysis."""
    if not error_msg:
        return "NONE"
    
    error_lower = error_msg.lower()
    
    if "json serializable" in error_lower or "date" in error_lower or "datetime" in error_lower:
        return "JSON_SERIALIZATION_ERROR"
    elif "empty query" in error_lower:
        return "EMPTY_QUERY"
    elif "parameter" in error_lower and "missing" in error_lower:
        return "MISSING_PARAMETER"
    elif "memory" in error_lower or "out of memory" in error_lower:
        return "OUT_OF_MEMORY"
    elif "neo4j" in error_lower:
        return "NEO4J_ERROR"
    else:
        return "OTHER"


def _extract_generation_details(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract detailed information about query generation attempts."""
    gen_trace = state.get("generation_trace", [])
    
    details = {
        "total_subproblems": len(gen_trace),
        "total_attempts": 0,
        "successful_attempts": 0,
        "failed_attempts": 0,
        "attempt_details": []
    }
    
    for subproblem_trace in gen_trace:
        subproblem = subproblem_trace.get("subproblem", "")[:100]
        attempts = subproblem_trace.get("attempts", [])
        details["total_attempts"] += len(attempts)
        
        for i, attempt in enumerate(attempts, 1):
            attempt_info = {
                "subproblem": subproblem,
                "attempt_number": i,
                "fragment": attempt.get("fragment", "")[:200] if attempt.get("fragment") else None,
                "ok": attempt.get("ok", False),
                "error": attempt.get("error"),
                "has_preview": bool(attempt.get("rows_preview"))
            }
            
            if attempt.get("ok"):
                details["successful_attempts"] += 1
            else:
                details["failed_attempts"] += 1
                
            details["attempt_details"].append(attempt_info)
    
    return details


def _analyze_query_result(state: Dict[str, Any], error: str | None) -> Dict[str, Any]:
    """Analyze the query result and provide detailed status information."""
    final_query = state.get("final_query")
    results = state.get("results")
    
    analysis = {
        "has_query": bool(final_query and final_query.strip()),
        "query_length": len(final_query) if final_query else 0,
        "has_error": bool(error),
        "error_type": _classify_error(error),
        "has_results": results is not None,
        "result_type": type(results).__name__ if results is not None else "None",
        "result_count": len(results) if isinstance(results, list) else None,
        "status": "UNKNOWN"
    }
    
    # Determine overall status
    if error:
        analysis["status"] = "ERROR"
        analysis["status_detail"] = f"Failed with {analysis['error_type']}"
    elif not analysis["has_query"]:
        analysis["status"] = "NO_QUERY_GENERATED"
        analysis["status_detail"] = "Agent failed to generate a valid query"
        # Add generation attempt details
        analysis["generation_info"] = _extract_generation_details(state)
    elif isinstance(results, list):
        if len(results) == 0:
            analysis["status"] = "SUCCESS_ZERO_ROWS"
            analysis["status_detail"] = "Query executed successfully but returned no data"
        else:
            analysis["status"] = "SUCCESS"
            analysis["status_detail"] = f"Query executed successfully and returned {len(results)} rows"
    elif results is None:
        analysis["status"] = "EXECUTION_FAILED"
        analysis["status_detail"] = "Query could not be executed or results could not be retrieved"
    
    return analysis


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
            def _run_agent_child(prompt_inner: str, schema_inner: str, out_q: multiprocessing.Queue) -> None:
                """Child process target: run the agent and put the result into the queue.

                We wrap exceptions and always put a dictionary so the parent can inspect status.
                """
                try:
                    st = agent_run(prompt_inner, schema_inner)
                    out_q.put({"ok": True, "state": st})
                except Exception as exc:  # pragma: no cover - best-effort capture
                    try:
                        out_q.put({"ok": False, "error": str(exc)})
                    except Exception:
                        # If putting to the queue fails for any reason, there's nothing we can do.
                        pass

            def run_agent_with_timeout(prompt_inner: str, schema_inner: str, timeout_seconds: int = 600, max_retries: int = 3) -> Dict[str, Any]:
                """Run agent_run in a separate process with timeout and retry logic.

                - timeout_seconds: how long to wait for the agent before considering it stuck
                - max_retries: number of attempts (including the first)
                Returns the state dict from the agent, or a dict with an "error" key on failure.
                """
                ctx = multiprocessing.get_context("fork")
                attempt = 0
                last_error: Optional[str] = None
                while attempt < max_retries:
                    attempt += 1
                    print(f"  Attempt {attempt}/{max_retries} for query [{i}/{len(prompts)}]...")
                    q: multiprocessing.Queue = ctx.Queue()
                    p = ctx.Process(target=_run_agent_child, args=(prompt_inner, schema_inner, q))
                    start_ts = time.perf_counter()
                    p.start()
                    p.join(timeout_seconds)
                    if p.is_alive():
                        # Timed out
                        print(f"    Timeout after {timeout_seconds} seconds. Terminating child process...")
                        try:
                            p.terminate()
                        except Exception:
                            pass
                        p.join(5)
                        last_error = f"Timeout after {timeout_seconds} seconds"
                        # Try again unless we've exhausted retries
                        if attempt < max_retries:
                            print("    Retrying...")
                            time.sleep(1)
                            continue
                        else:
                            return {"error": last_error}
                    # Process finished within timeout; try to read result
                    try:
                        res = q.get_nowait()
                    except _queue.Empty:
                        # Nothing in queue: child crashed silently
                        last_error = "Agent process exited without returning state"
                        if attempt < max_retries:
                            print("    No result returned from child process. Retrying...")
                            time.sleep(1)
                            continue
                        return {"error": last_error}

                    # Interpret result
                    if isinstance(res, dict) and res.get("ok"):
                        return res.get("state") or {}
                    else:
                        last_error = res.get("error") if isinstance(res, dict) else str(res)
                        print(f"    Agent raised an exception: {last_error}")
                        if attempt < max_retries:
                            print("    Retrying after error...")
                            time.sleep(1)
                            continue
                        return {"error": last_error}

            # Run agent with timeout and retry
            state = run_agent_with_timeout(prompt, SCHEMA, timeout_seconds=400, max_retries=1)
            t1_wall = datetime.now(timezone.utc)
            duration = time.perf_counter() - t0

            # Load last_run.json produced by main.run for token usage and metadata
            last_run_file = Path("last_run.json")
            token_usage: Dict[str, Any] | None = None
            metadata: Dict[str, Any] | None = None
            partial_cypher: str | None = None
            if last_run_file.exists():
                try:
                    lr = json.loads(last_run_file.read_text())
                    token_usage = lr.get("token_usage")
                    metadata = lr.get("metadata")
                    # If agent failed, try to extract partial Cypher from last_run.json
                    if state.get("error") and lr.get("cypher"):
                        partial_cypher = lr.get("cypher")
                        print(f"    Extracted partial Cypher from last_run.json ({len(partial_cypher)} chars)")
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

            # Perform comprehensive analysis of the query result
            analysis = _analyze_query_result(state, state.get("error"))

            result_row: Dict[str, Any] = {
                "index": i,
                "category": category,
                "prompt": prompt,
                "expected_cypher": expected_cypher,
                "agent_cypher": state.get("final_query") or partial_cypher,  # Use partial Cypher if final_query is missing
                "response": state.get("results"),
                "error": state.get("error"),
                "analysis": analysis,  # Comprehensive analysis including status, error type, generation details
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

        # Calculate summary statistics
        summary = {
            "total_queries": len(results),
            "run_duration_seconds": (ended - started).total_seconds(),
            "model": model_name or MODEL_PROVIDER,
            "provider": MODEL_PROVIDER,
            "commit": commit,
            "started_at": started.isoformat(),
            "ended_at": ended.isoformat(),
            "status_breakdown": {},
            "error_types": {},
            "total_tokens": 0,
            "total_duration_seconds": 0,
        }
        
        # Aggregate statistics from all results
        for result in results:
            # Count by status
            status = result.get("analysis", {}).get("status", "UNKNOWN")
            summary["status_breakdown"][status] = summary["status_breakdown"].get(status, 0) + 1
            
            # Count by error type
            error_type = result.get("analysis", {}).get("error_type", "NONE")
            if error_type != "NONE":
                summary["error_types"][error_type] = summary["error_types"].get(error_type, 0) + 1
            
            # Sum tokens
            token_usage = result.get("token_usage", {})
            if token_usage:
                summary["total_tokens"] += token_usage.get("total", 0)
            
            # Sum duration
            summary["total_duration_seconds"] += result.get("duration_seconds", 0)
        
        # Calculate success rate
        success_count = summary["status_breakdown"].get("SUCCESS", 0)
        summary["success_rate"] = round(success_count / len(results) * 100, 2) if results else 0
        
        # Add summary to results structure
        output_data = {
            "summary": summary,
            "results": results
        }

        # Determine output filename
        if total_runs == 1:
            out_path = Path(args.output)
        else:
            # Prefer model name from metadata; fall back to provider
            model_part = _sanitize_filename_part(model_name or MODEL_PROVIDER)
            out_path = Path(f"ev_res_{model_part}_{run_idx}.json")

        # Write results with summary
        out_path.write_text(json.dumps(output_data, indent=2, cls=Neo4jEncoder))
        
        # Print summary to console
        print(f"\n{'='*80}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total queries: {summary['total_queries']}")
        print(f"Model: {summary['model']}")
        print(f"Duration: {summary['run_duration_seconds']:.2f} seconds")
        print(f"Total tokens: {summary['total_tokens']:,}")
        print(f"\nStatus Breakdown:")
        for status, count in sorted(summary["status_breakdown"].items()):
            pct = (count / summary['total_queries'] * 100) if summary['total_queries'] > 0 else 0
            emoji = "✅" if status == "SUCCESS" else "🟡" if status == "SUCCESS_ZERO_ROWS" else "🔴"
            print(f"  {emoji} {status:25s}: {count:3d} ({pct:5.1f}%)")
        
        if summary["error_types"]:
            print(f"\nError Types:")
            for error_type, count in sorted(summary["error_types"].items()):
                print(f"  🔴 {error_type:30s}: {count:3d}")
        
        print(f"\nSuccess Rate: {summary['success_rate']}%")
        print(f"\nResults saved to: {out_path}")
        print(f"{'='*80}\n")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
