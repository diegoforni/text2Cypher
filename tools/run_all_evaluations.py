#!/usr/bin/env python3
"""
Run evaluations for multiple OpenAI models, each model twice.

This script runs evaluator.py for each specified model with the proper
environment variables set, saving results to eval_results/ directory.

Usage:
  python tools/run_all_evaluations.py [--workers N] [--timeout SECONDS] [--retries N]
  
  --workers N:  Number of parallel workers (default: 3)
                Recommended: 2-3 for Tier 1-2, 4-5 for Tier 3+
  --timeout S:  Timeout per evaluation run in seconds (default: 300 = 5 minutes)
  --retries N:  Number of retries on timeout/failure (default: 2)
"""

import subprocess
import sys
import os
import argparse
import signal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Tuple, Optional
import threading
import time

# Models to evaluate (remaining models that need to be run)
# Note: codex-mini-latest removed - requires organization verification
# Single model to evaluate (we run this model multiple times)
MODELS = ["gpt-5-mini"]

# Total number of runs across the single model (can be overridden via CLI)
DEFAULT_TOTAL_RUNS = 6

# Timeout and retry settings (can be overridden via CLI)
DEFAULT_TIMEOUT = 600  # 10 minutes (increased for complex queries)
DEFAULT_RETRIES = 2

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results"
EVALUATOR_SCRIPT = PROJECT_ROOT / "tools" / "evaluator.py"
PYTHON_PATH = PROJECT_ROOT / ".venv" / "bin" / "python3.12"

# Thread-safe print lock
print_lock = threading.Lock()


@dataclass
class EvalResult:
    model: str
    run_num: int
    success: bool
    message: str
    attempts: int = 1


def safe_print(*args, **kwargs):
    """Thread-safe print."""
    with print_lock:
        print(*args, **kwargs)


def run_evaluation_with_timeout(model: str, run_num: int, timeout: int, _max_retries: int) -> EvalResult:
    """Run a single evaluation for a model one time with timeout.

    This function performs a single attempt only. Retry orchestration is handled
    at a higher level in `main` so the runner can re-submit only failed tasks.
    """
    output_file = EVAL_RESULTS_DIR / f"{model}_run{run_num}.json"

    env = os.environ.copy()
    env["MODEL_PROVIDER"] = "openai"
    env["OPENAI_MODEL"] = model

    safe_print(f"[START] {model} run {run_num} -> {output_file.name}")
    start_time = time.time()

    try:
        process = subprocess.Popen(
            [str(PYTHON_PATH), str(EVALUATOR_SCRIPT), "--output", str(output_file)],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid if sys.platform != 'win32' else None,
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            elapsed = time.time() - start_time

            if process.returncode != 0:
                safe_print(f"[FAIL] {model} run {run_num} - exit code {process.returncode} ({elapsed:.1f}s)")
                return EvalResult(model, run_num, False, f"exit code {process.returncode}", 1)

            safe_print(f"[DONE] {model} run {run_num} ✓ ({elapsed:.1f}s)")
            return EvalResult(model, run_num, True, "completed", 1)

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            safe_print(f"[TIMEOUT] {model} run {run_num} - exceeded {timeout}s timeout ({elapsed:.1f}s)")

            # Kill the process group
            try:
                if sys.platform != 'win32':
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    time.sleep(2)
                    if process.poll() is None:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                else:
                    process.kill()
            except Exception as kill_err:
                safe_print(f"[WARN] Error killing process: {kill_err}")

            process.wait()
            return EvalResult(model, run_num, False, f"timeout after {timeout}s", 1)

    except Exception as e:
        elapsed = time.time() - start_time
        safe_print(f"[ERROR] {model} run {run_num} - {e} ({elapsed:.1f}s)")
        return EvalResult(model, run_num, False, str(e), 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=3,
        help="Number of parallel workers (default: 3). Recommended: 2-3 for Tier 1-2, 4-5 for Tier 3+"
    )
    parser.add_argument(
        "--total-runs", "-n",
        type=int,
        default=DEFAULT_TOTAL_RUNS,
        dest="total_runs",
        help=f"Total number of runs to execute for the single model (default: {DEFAULT_TOTAL_RUNS})"
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout per evaluation run in seconds (default: {DEFAULT_TIMEOUT} = 5 minutes)"
    )
    parser.add_argument(
        "--retries", "-r",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"Number of retries on timeout/failure (default: {DEFAULT_RETRIES})"
    )
    args = parser.parse_args()
    
    timeout = args.timeout
    max_retries = args.retries
    
    # Ensure output directory exists
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Build list of all (model, run_num) tasks for the single model
    tasks: List[Tuple[str, int]] = []
    total_runs = args.total_runs
    model = MODELS[0]
    for run_num in range(1, total_runs + 1):
        tasks.append((model, run_num))

    workers = min(args.workers, total_runs)  # Don't use more workers than tasks
    
    print(f"{'='*60}")
    print(f"Parallel Evaluation Runner")
    print(f"{'='*60}")
    print(f"Model: {MODELS[0]}")
    print(f"Total runs: {total_runs}")
    print(f"Parallel workers: {workers}")
    print(f"Timeout per run: {timeout}s ({timeout//60}m)")
    print(f"Max retries: {max_retries}")
    print(f"Results directory: {EVAL_RESULTS_DIR}")
    print(f"{'='*60}\n")
    
    # We'll orchestrate retries at the main level so we can re-submit only failed
    # tasks. `run_evaluation_with_timeout` performs one attempt per call.
    results_by_task = {}  # type: ignore[var-annotated]

    remaining_tasks = list(tasks)
    attempts_map = {task: 0 for task in tasks}

    for attempt_num in range(1, max_retries + 1):
        if not remaining_tasks:
            break

        safe_print(f"[PASS] Retry pass {attempt_num}/{max_retries} - {len(remaining_tasks)} task(s) to run")

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_task = {
                executor.submit(run_evaluation_with_timeout, model, run_num, timeout, max_retries): (model, run_num)
                for model, run_num in remaining_tasks
            }

            next_remaining = []

            for future in as_completed(future_to_task):
                task = future_to_task[future]
                result = future.result()

                # Update attempt count for this task
                attempts_map[task] = attempts_map.get(task, 0) + 1
                result.attempts = attempts_map[task]

                if result.success:
                    results_by_task[task] = result
                else:
                    # keep last failure result for reporting, and mark for retry
                    results_by_task[task] = result
                    next_remaining.append(task)

                completed = sum(1 for r in results_by_task.values() if r.success)
                failed = sum(1 for r in results_by_task.values() if not r.success)
                safe_print(f"Progress: {len(results_by_task)}/{total_runs} (completed: {completed}, failed: {failed})")

        if next_remaining and attempt_num < max_retries:
            safe_print(f"[RETRY] {len(next_remaining)} failed task(s) will be retried after 5s...")
            time.sleep(5)
            remaining_tasks = next_remaining
        else:
            # either no failures or we've exhausted retries
            remaining_tasks = next_remaining
            break

    # Collect final results in a list preserving order of tasks
    results: List[EvalResult] = [results_by_task.get(task, EvalResult(task[0], task[1], False, "not run", attempts_map.get(task, 0))) for task in tasks]
    
    # Final summary
    completed = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total runs: {total_runs}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print(f"\nFailed runs:")
        for r in results:
            if not r.success:
                print(f"  - {r.model} run {r.run_num}: {r.message} (after {r.attempts} attempts)")
    
    print(f"\nResults saved to: {EVAL_RESULTS_DIR}")
    
    # List generated files
    print(f"\nGenerated files:")
    for f in sorted(EVAL_RESULTS_DIR.glob("*.json")):
        print(f"  - {f.name}")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
