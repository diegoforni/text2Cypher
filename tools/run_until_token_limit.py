#!/usr/bin/env python3
"""
Run evaluations continuously until reaching a token limit.

This script runs the evaluator multiple times, tracking actual token usage
from the OpenAI API, and stops when approaching the specified token limit.

Usage:
  python tools/run_until_token_limit.py --model gpt-5.2 --max-tokens 9000000 --output-dir gpt-5.2_results
"""

import argparse
import sys

# Unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR_SCRIPT = PROJECT_ROOT / "tools" / "run_evaluator_with_query_timeout.py"
PYTHON_PATH = PROJECT_ROOT / ".venv" / "bin" / "python3.12"


def load_token_results(json_file: Path) -> Dict[str, Any]:
    """Load and parse results JSON to extract token usage."""
    try:
        data = json.loads(json_file.read_text())
        summary = data.get("summary", {})
        return {
            "total_tokens": summary.get("total_tokens", 0),
            "run_count": len(data.get("results", [])),
            "model": summary.get("model", "unknown"),
        }
    except Exception as e:
        print(f"Warning: Could not parse {json_file}: {e}")
        return {"total_tokens": 0, "run_count": 0, "model": "unknown"}


def run_single_evaluation(
    model: str,
    run_number: int,
    output_dir: Path,
    timeout: int = 600,
) -> Optional[Dict[str, Any]]:
    """Run a single evaluation and return token usage info."""
    output_file = output_dir / f"{model}_run{run_number}.json"

    env = os.environ.copy()
    env["MODEL_PROVIDER"] = "openai"
    env["OPENAI_MODEL"] = model

    print(f"\n{'='*80}")
    print(f"[START] {model} run {run_number} -> {output_file.name}")
    print(f"{'='*80}")
    start_time = time.time()

    try:
        process = subprocess.Popen(
            [str(PYTHON_PATH), str(EVALUATOR_SCRIPT), "--output", str(output_file), "--queries", "queries.json"],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            elapsed = time.time() - start_time

            if process.returncode != 0:
                print(f"[FAIL] Exit code {process.returncode} ({elapsed:.1f}s)")
                if stderr:
                    print(f"STDERR:\n{stderr}")
                return None

            # Parse token usage from output file
            token_info = load_token_results(output_file)
            tokens_used = token_info["total_tokens"]

            print(f"[DONE] Completed in {elapsed:.1f}s")
            print(f"TOKENS: {tokens_used:,} total tokens used in this run")

            return {
                "run_number": run_number,
                "output_file": str(output_file),
                "tokens_used": tokens_used,
                "duration_seconds": elapsed,
                "success": True,
            }

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            print(f"[TIMEOUT] Exceeded {timeout}s ({elapsed:.1f}s)")
            process.kill()
            process.wait()
            return None

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[ERROR] {e} ({elapsed:.1f}s)")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run evaluations until reaching a token limit"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2",
        help="OpenAI model to use (default: gpt-5.2)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=9000000,
        help="Maximum total tokens to consume (default: 9,000,000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=14400,
        help="Timeout per evaluation run in seconds (default: 14400 = 4 hours for all queries)",
    )
    parser.add_argument(
        "--buffer",
        type=int,
        default=500000,
        help="Buffer tokens to stop before reaching limit (default: 500,000)",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tracking
    total_tokens = 0
    token_limit = args.max_tokens - args.buffer  # Stop before hitting actual limit
    run_number = 1
    results_summary = []
    started = datetime.now(timezone.utc)

    print(f"{'='*80}")
    print(f"Token-Limited Evaluation Runner")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Token limit: {args.max_tokens:,} (buffer: {args.buffer:,})")
    print(f"Will stop at: {token_limit:,} tokens")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")

    try:
        while total_tokens < token_limit:
            # Check if we're close to the limit
            if total_tokens > 0:
                remaining = token_limit - total_tokens
                print(f"\n[CURRENT STATUS]")
                print(f"  Total tokens used: {total_tokens:,}")
                print(f"  Remaining until limit: {remaining:,}")
                print(f"  Completed runs: {run_number - 1}")

                # If we're very close, stop
                if remaining < 100000:  # Less than 100K buffer
                    print(f"\n[STOP] Approaching token limit. Stopping.")
                    break

            # Run a single evaluation
            result = run_single_evaluation(
                model=args.model,
                run_number=run_number,
                output_dir=output_dir,
                timeout=args.timeout,
            )

            if result is None or not result.get("success"):
                print(f"\n[ERROR] Run {run_number} failed. Waiting 10s before retry...")
                time.sleep(10)
                continue

            # Update totals
            tokens_used = result["tokens_used"]
            total_tokens += tokens_used
            results_summary.append(result)
            run_number += 1

            # Check if we've exceeded the limit
            if total_tokens >= token_limit:
                print(f"\n[STOP] Token limit reached!")
                break

        # Final summary
        ended = datetime.now(timezone.utc)
        duration = (ended - started).total_seconds()

        print(f"\n{'='*80}")
        print(f"FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"Model: {args.model}")
        print(f"Total runs completed: {run_number - 1}")
        print(f"Total tokens used: {total_tokens:,}")
        print(f"Token limit: {args.max_tokens:,}")
        print(f"Percentage of limit: {(total_tokens / args.max_tokens * 100):.1f}%")
        print(f"Duration: {duration / 3600:.2f} hours ({duration / 60:.1f} minutes)")
        print(f"Avg tokens per run: {total_tokens / (run_number - 1):,.0f}" if run_number > 1 else "N/A")
        print(f"Avg time per run: {duration / (run_number - 1) / 60:.1f} minutes" if run_number > 1 else "N/A")
        print(f"Results directory: {output_dir}")

        # Save summary
        summary_file = output_dir / "RUN_SUMMARY.json"
        summary_data = {
            "model": args.model,
            "max_tokens": args.max_tokens,
            "buffer": args.buffer,
            "actual_limit": token_limit,
            "total_runs": run_number - 1,
            "total_tokens_used": total_tokens,
            "percentage_of_limit": round(total_tokens / args.max_tokens * 100, 2),
            "started_at": started.isoformat(),
            "ended_at": ended.isoformat(),
            "duration_seconds": duration,
            "runs": results_summary,
        }
        summary_file.write_text(json.dumps(summary_data, indent=2))
        print(f"Summary saved to: {summary_file}")
        print(f"{'='*80}\n")

        return 0

    except KeyboardInterrupt:
        print(f"\n\n[INTERRUPTED] Stopped by user")
        ended = datetime.now(timezone.utc)
        duration = (ended - started).total_seconds()
        print(f"Total tokens used: {total_tokens:,}")
        print(f"Runs completed: {run_number - 1}")
        print(f"Duration: {duration / 60:.1f} minutes")
        return 1


if __name__ == "__main__":
    sys.exit(main())
