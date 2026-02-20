"""
LLM-as-a-judge evaluation and HTML report generator.

Usage:
  python tools/evaluate_results.py --input-dir ./runs --out-dir ./report [--use-llm]

Reads all JSON files in --input-dir that contain arrays of evaluation entries
with keys similar to those in evaluation_results.json. For each entry, compares
`agent_cypher` vs `expected_cypher` and produces a similarity score using an
LLM judge (when --use-llm and an LLM is configured) or a deterministic fallback.

Outputs:
  - {out-dir}/scores.json: normalized data with scores per (query, provider)
  - {out-dir}/index.html: simple static report to browse the results
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root importable when running from tools/
import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

# Load .env so config.get_llm can work if requested
load_dotenv()

from config import get_llm, MODEL_PROVIDER  # type: ignore


def _read_all_entries(input_dir: Path) -> List[Dict[str, Any]]:
    """Read all JSON files in directory and concatenate entries.

    Supports two formats:
    1. Files where root is a list of entries: [...]
    2. Files with summary structure: {"summary": {...}, "results": [...]}
    """
    entries: List[Dict[str, Any]] = []
    for p in sorted(input_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        
        # Handle {"summary": {...}, "results": [...]} format (from evaluator.py)
        if isinstance(data, dict) and "results" in data:
            results = data.get("results", [])
            if isinstance(results, list):
                for item in results:
                    if isinstance(item, dict):
                        item["_source_file"] = str(p)
                        entries.append(item)
        # Handle direct list format: [...]
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    item["_source_file"] = str(p)
                    entries.append(item)
    return entries


def _normalize_query(q: str) -> str:
    """Lightweight Cypher normalization to reduce superficial differences."""
    if not q:
        return ""
    s = q.strip()
    # Normalize whitespace and case, remove trailing semicolons
    s = s.replace("\r", "\n")
    s = "\n".join(line.strip() for line in s.splitlines())
    s = " ".join(s.split())
    s = s.removeprefix(";").removesuffix(";")
    return s.lower()


def _jaccard_similarity(a: str, b: str) -> float:
    """Deterministic fallback similarity on token sets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    # Keep alphanumerics and common cypher punctuation as separate tokens
    def toks(s: str) -> List[str]:
        out: List[str] = []
        cur = []
        for ch in s:
            if ch.isalnum() or ch in "_":
                cur.append(ch)
            else:
                if cur:
                    out.append("".join(cur))
                    cur = []
                if ch in ",()[]{}:" :
                    out.append(ch)
        if cur:
            out.append("".join(cur))
        return out
    A = set(toks(a))
    B = set(toks(b))
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _llm_judge(llm: Any, expected: str, candidate: str, prompt: str, category: str) -> Tuple[float, str]:
    """Ask the LLM to grade whether the candidate query is a valid interpretation.

    Returns (score_float_0_1, reasoning, tokens).
    """
    system = (
        "You are a Cypher query validator. Your task is to determine if the candidate query "
        "is a VALID interpretation of the user's question, NOT whether it matches the expected query exactly. "
        "Natural language questions can have multiple valid Cypher interpretations. "
        "\n\n"
        "**IMPORTANT: Efficiency is NOT being evaluated.**\n"
        "Only assess:\n"
        "1. Is the Cypher query syntactically valid?\n"
        "2. Does it correctly answer the user's prompt?\n"
        "\n"
        "Score 1.0 if the candidate query:\n"
        "- Is syntactically valid Cypher (no syntax errors)\n"
        "- Would execute successfully on the schema\n"
        "- Correctly answers the user's question (even with different approaches)\n"
        "- Uses correct graph patterns and relationships\n"
        "\n"
        "Score 0.0 if the candidate query:\n"
        "- Has syntax errors\n"
        "- References non-existent nodes, relationships, or properties\n"
        "- Does NOT answer the user's question correctly\n"
        "- Returns completely wrong data for the question\n"
        "- Is empty or malformed\n"
        "\n"
        "The expected query is provided only as a reference - the candidate does NOT need to match it. "
        "Different approaches (e.g., different aggregations, filtering strategies, or return formats) "
        "can all be valid if they answer the question correctly.\n"
        "\n"
        "**Do NOT penalize for:**\n"
        "- Query performance or efficiency\n"
        "- Optimal execution plans\n"
        "- Index usage\n"
        "- Different but valid approaches to the same problem"
    )
    user = f"""
Task category: {category}
User question: {prompt}

Reference query (for context only - candidate doesn't need to match this):
{expected}

Candidate query to evaluate:
{candidate}

Evaluate whether the candidate query is a VALID interpretation that answers the user's question.

**IMPORTANT: Multiple valid interpretations may exist.** The candidate query does NOT need to match the reference query. Any valid Cypher query that correctly answers the user's question should receive a score of 1.0.

**Classification:**
- 1.0: Answers the prompt (valid Cypher + correctly answers the question under any reasonable interpretation)
- 0.0: Does not answer the prompt (invalid Cypher OR wrong interpretation)

**Remember: Efficiency is NOT evaluated.** Focus only on validity and correctness.

Respond with a single JSON object with keys:
- score: 1.0 if it answers the prompt, 0.0 if it does not
- reasoning: brief explanation of why it answers or doesn't answer the prompt
"""
    try:
        resp = llm.invoke([("system", system), ("user", user)])
        raw = getattr(resp, "content", resp)

        def _coerce_text(value: Any) -> str:
            if isinstance(value, str):
                return value
            if isinstance(value, list):
                parts: List[str] = []
                for v in value:
                    if isinstance(v, str):
                        parts.append(v)
                    elif isinstance(v, dict):
                        t = v.get("text") or v.get("content")
                        if isinstance(t, str):
                            parts.append(t)
                    else:
                        t = getattr(v, "text", None)
                        if isinstance(t, str):
                            parts.append(t)
                return "".join(parts) if parts else str(value)
            return str(value)

        content = _coerce_text(raw).strip()
        # Attempt to extract JSON object
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end != -1 and end > start:
            content = content[start : end + 1]
        data = json.loads(content)
        score = float(data.get("score", 0))
        score = max(0.0, min(1.0, score))
        reasoning = str(data.get("reasoning", "")).strip()
        # Try to attach token usage if present (LangChain-style)
        tok_used = 0
        try:
            # Try usage_metadata first (newer LangChain)
            if hasattr(resp, "usage_metadata"):
                um = resp.usage_metadata
                if isinstance(um, dict):
                    tok_used = int(um.get("total_tokens", 0))
            # Fallback to response_metadata
            elif hasattr(resp, "response_metadata"):
                meta = resp.response_metadata
                if isinstance(meta, dict):
                    tu = meta.get("token_usage") or meta.get("usage") or meta.get("tokenUsage") or {}
                    if isinstance(tu, dict):
                        tok_used = int(tu.get("total") or tu.get("input") or tu.get("prompt_tokens") or 0) or (
                            int(tu.get("prompt_tokens", 0)) + int(tu.get("completion_tokens", 0))
                        )
        except Exception:
            tok_used = 0
        return score, reasoning, tok_used
    except Exception as e:
        # On failure, return a conservative fallback using token overlap
        exp_n = _normalize_query(expected)
        cand_n = _normalize_query(candidate)
        return round(_jaccard_similarity(exp_n, cand_n), 2), f"fallback: {e}", 0


class _TokenBudgetExceeded(Exception):
    pass


class _TokenBudgetLLM:
    """Simple wrapper around an LLM that tracks token usage and enforces a token budget.

    It expects the wrapped LLM to expose .invoke(messages) and for the returned
    object to provide token usage in resp.response_metadata['token_usage'] or
    resp.response_metadata.token_usage; otherwise token usage is considered 0.
    """
    def __init__(self, base_llm: Any, budget_tokens: int | None, min_tokens_per_call: int = 10):
        self.base_llm = base_llm
        self.budget = int(budget_tokens) if budget_tokens and budget_tokens > 0 else None
        self.min_per_call = int(min_tokens_per_call)

    def invoke(self, messages: Any):
        if self.budget is not None and self.budget < self.min_per_call:
            raise _TokenBudgetExceeded(f"LLM token budget exhausted: remaining={self.budget}")
        resp = self.base_llm.invoke(messages)
        # Deduct tokens from the budget if available
        try:
            meta = getattr(resp, "response_metadata", None) or getattr(resp, "response", None) or {}
            if isinstance(meta, dict):
                tu = meta.get("token_usage") or meta.get("usage") or meta.get("tokenUsage") or {}
            else:
                tu = getattr(meta, "token_usage", {}) if meta is not None else {}
            if isinstance(tu, dict):
                tok_used = int(tu.get("total") or tu.get("input") or 0) or (
                    int(tu.get("prompt_tokens", 0)) + int(tu.get("completion_tokens", 0))
                )
            else:
                tok_used = 0
        except Exception:
            tok_used = 0
        if self.budget is not None:
            self.budget -= int(tok_used or 0)
        return resp


def _grade_entries(entries: List[Dict[str, Any]], use_llm: bool, llm_token_budget: int | None = None, llm_min_tokens_per_call: int = 10) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compute validity score for each entry, with special focus on re-evaluating failed queries.

    The LLM judge evaluates whether queries are valid interpretations, not exact matches.
    Failed queries (with errors) are automatically sent to the LLM judge to determine if they
    are actually valid despite execution errors.

    Returns: (graded_entries, skipped_no_query, impossible_questions)
    """
    llm = get_llm() if use_llm else None
    if use_llm and llm is None:
        print("Warning: --use-llm was requested but no LLM was configured/found; using deterministic fallback. Check your MODEL_PROVIDER and API keys.")
    if llm and llm_token_budget:
        llm = _TokenBudgetLLM(llm, llm_token_budget, min_tokens_per_call=llm_min_tokens_per_call)

    graded: List[Dict[str, Any]] = []
    skipped_no_query: List[Dict[str, Any]] = []
    impossible_questions: List[Dict[str, Any]] = []
    total_entries = len(entries)
    failed_count = 0
    reevaluated_count = 0

    for idx, item in enumerate(entries, 1):
        expected = _normalize_query(str(item.get("expected_cypher", "")))
        candidate_raw = str(item.get("agent_cypher", ""))
        candidate = _normalize_query(candidate_raw)
        prompt = str(item.get("prompt", ""))
        category = str(item.get("category", ""))

        # Skip if no query was generated
        if not candidate or candidate.strip().lower() in ("none", "", "null"):
            print(f"  Skipping [{idx}/{total_entries}]: {prompt[:60]}... (no query generated)", file=sys.stderr)
            enriched = dict(item)
            enriched["similarity_score"] = None
            enriched["judge_reasoning"] = "Skipped: No query generated"
            enriched["judge_tokens"] = 0
            enriched["was_failed_query"] = True
            skipped_no_query.append(enriched)
            failed_count += 1
            continue

        # Check for impossible questions (category: impossible or expected says "not enough information")
        expected_cypher = item.get("expected_cypher", "")
        is_impossible = (
            category == "impossible" or
            ("not enough information" in expected_cypher.lower()) or
            ("no" in expected_cypher.lower() and "malware family" in expected_cypher.lower())
        )

        if is_impossible:
            print(f"  Marking as impossible [{idx}/{total_entries}]: {prompt[:60]}...", file=sys.stderr)
            enriched = dict(item)
            enriched["similarity_score"] = None
            enriched["judge_reasoning"] = "Skipped: Impossible question (schema limitation)"
            enriched["judge_tokens"] = 0
            enriched["was_failed_query"] = False
            impossible_questions.append(enriched)
            continue
        error = item.get("error")
        
        # Track if this was a failed query
        was_failed = bool(error) or not candidate
        if was_failed:
            failed_count += 1

        if not expected and not candidate:
            score, reasoning, judge_tokens = 0.0, "no expected or candidate", 0
        elif not candidate:
            score, reasoning, judge_tokens = 0.0, "empty candidate - no query generated", 0
        elif not expected:
            # Rare, but avoid giving free credit
            score, reasoning, judge_tokens = 0.0, "missing expected", 0
        else:
            # For failed queries OR when LLM is enabled, always use LLM judge
            use_llm_for_this = llm is not None and (use_llm or was_failed)
            
            if use_llm_for_this:
                if was_failed:
                    reevaluated_count += 1
                    print(f"  Re-evaluating failed query [{idx}/{total_entries}]: {prompt[:60]}...")
                try:
                    out = _llm_judge(llm, expected, candidate, prompt, category)
                    # _llm_judge returns (score, reasoning, tokens)
                    if isinstance(out, tuple) and len(out) == 3:
                        score, reasoning, judge_tokens = out
                    else:
                        # Backwards compat: if older llm returned 2 values
                        score, reasoning = out
                        judge_tokens = 0
                except _TokenBudgetExceeded as e:
                    # Budget exhausted; fallback to deterministic
                    score = round(_jaccard_similarity(expected, candidate), 2)
                    reasoning = f"fallback: token budget exhausted ({e})"
                    judge_tokens = 0
            else:
                score = round(_jaccard_similarity(expected, candidate), 2)
                reasoning = "deterministic token overlap"
                judge_tokens = 0

        enriched = dict(item)
        enriched["similarity_score"] = float(round(score, 2))
        enriched["judge_reasoning"] = reasoning
        enriched["judge_tokens"] = int(judge_tokens or 0)
        enriched["was_failed_query"] = was_failed
        # Extract provider if present
        provider = None
        md = item.get("metadata")
        if isinstance(md, dict):
            provider = md.get("provider") or md.get("model_provider")
        enriched["provider"] = provider or "unknown"
        # Total tokens and duration convenience fields
        usage = item.get("token_usage") or {}
        enriched["total_tokens"] = int(usage.get("total", 0) or 0)
        enriched["duration_seconds"] = float(item.get("duration_seconds", 0.0) or 0.0)
        graded.append(enriched)

    print(f"\nProcessed {total_entries} entries:")
    print(f"  - Evaluated with LLM: {len(graded)}")
    print(f"  - Skipped (no query generated): {len(skipped_no_query)}")
    print(f"  - Impossible questions: {len(impossible_questions)}")
    print(f"  - Re-evaluated failed queries: {reevaluated_count}")

    return graded, skipped_no_query, impossible_questions


def _aggregate_for_report(graded: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a structure convenient for HTML rendering: queries x providers matrix.

    Supports multiple repetitions per provider per query by grouping runs and
    computing per-cell averages.
    """
    # Identify queries by index if present; fallback to prompt text
    queries: Dict[str, Dict[str, Any]] = {}
    providers_set = set()

    for item in graded:
        # Build a stable key: prefer explicit index
        if "index" in item:
            qkey = f"{int(item['index'])}: {item.get('prompt','')[:60]}".strip()
        else:
            qkey = item.get("prompt") or f"{item.get('_source_file','file')}:{item.get('category','') }"
        provider = item.get("provider", "unknown")
        providers_set.add(provider)

        # Initialize structure for provider runs
        qrow = queries.setdefault(qkey, {})
        cell = qrow.setdefault(provider, {
            "runs": [],
            "expected_cypher": item.get("expected_cypher"),
            "category": item.get("category"),
        })

        cell["runs"].append({
            "score": item.get("similarity_score", 0.0),
            "tokens": item.get("total_tokens", 0),
            "duration": item.get("duration_seconds", 0.0),
            "error": item.get("error"),
            "agent_cypher": item.get("agent_cypher"),
            "reason": item.get("judge_reasoning"),
            "started_at": item.get("started_at"),
            "ended_at": item.get("ended_at"),
            "source": item.get("_source_file"),
            "model": (item.get("metadata") or {}).get("model"),
        })

    # Compute averages per cell and provider-level aggregates
    provider_stats: Dict[str, Dict[str, Any]] = {}
    for _, row in queries.items():
        for provider, cell in row.items():
            runs = cell.get("runs", [])
            if not runs:
                continue
            avg_score = sum(float(r.get("score") or 0.0) for r in runs) / max(1, len(runs))
            avg_tokens = sum(int(r.get("tokens") or 0) for r in runs) / max(1, len(runs))
            avg_duration = sum(float(r.get("duration") or 0.0) for r in runs) / max(1, len(runs))
            cell["avg_score"] = round(avg_score, 3)
            cell["avg_tokens"] = round(avg_tokens, 2)
            cell["avg_duration"] = round(avg_duration, 3)
            cell["count"] = len(runs)

    # Provider aggregates: average of per-query avg scores; totals over all runs for tokens/time
    for provider in providers_set:
        # Collect the per-query cells for this provider
        cells = [row[provider] for row in queries.values() if provider in row]
        if not cells:
            continue
        avg_over_queries = sum(float(c.get("avg_score") or 0.0) for c in cells) / max(1, len(cells))
        total_tokens = sum(int(r.get("tokens") or 0) for c in cells for r in c.get("runs", []))
        total_duration = sum(float(r.get("duration") or 0.0) for c in cells for r in c.get("runs", []))
        provider_stats[provider] = {
            "avg_score": round(avg_over_queries, 3),
            "total_tokens": int(total_tokens),
            "total_duration": round(float(total_duration), 2),
            "queries": len(cells),
            "runs": sum(int(c.get("count") or 0) for c in cells),
        }

    return {
        "providers": sorted(providers_set),
        "queries": queries,
        "provider_stats": provider_stats,
    }


def _write_scores_json(out_dir: Path, graded: List[Dict[str, Any]], skipped_no_query: List[Dict[str, Any]], impossible_questions: List[Dict[str, Any]], matrix: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "entries": graded,
        "skipped_no_query": skipped_no_query,
        "impossible_questions": impossible_questions,
        "matrix": matrix,
    }
    (out_dir / "scores.json").write_text(json.dumps(payload, indent=2))


def _render_html(out_dir: Path, matrix: Dict[str, Any]) -> None:
    providers: List[str] = matrix.get("providers", [])
    queries: Dict[str, Dict[str, Any]] = matrix.get("queries", {})
    provider_stats: Dict[str, Any] = matrix.get("provider_stats", {})

    # Simple HTML with inline CSS for portability
    css = """
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 16px; }
    h1 { font-size: 20px; margin: 0 0 8px; }
    .meta { margin-bottom: 12px; color: #555; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 8px; font-size: 13px; vertical-align: top; }
    th { background: #f5f5f5; position: sticky; top: 0; }
    .score { font-weight: 600; }
    .score.good { color: #0a7a2f; }
    .score.mid { color: #b07c00; }
    .score.bad { color: #b00020; }
    .mono { font-family: ui-monospace, Menlo, Monaco, Consolas, monospace; white-space: pre-wrap; }
    details { margin-top: 6px; }
    .stats { margin: 12px 0; }
    .stat { display: inline-block; margin-right: 20px; }
    """

    def score_class(v: float) -> str:
        if v >= 0.8:
            return "good"
        if v >= 0.5:
            return "mid"
        return "bad"

    # Header row: Provider columns
    header_cells = "<th>Query</th>" + "".join(f"<th>{p}</th>" for p in providers)

    # Rows: one per query; each cell shows avg score + avg tokens + avg duration and details of runs
    rows_html: List[str] = []
    for qkey, row in queries.items():
        cells: List[str] = [f"<td class=mono>{qkey}</td>"]
        for p in providers:
            cell = row.get(p)
            if not cell:
                cells.append("<td>-</td>")
                continue
            s = float(cell.get("avg_score") or 0.0)
            cls = score_class(s)
            tokens = int(cell.get("avg_tokens") or 0)
            dur = float(cell.get("avg_duration") or 0.0)
            cat = cell.get("category") or ""
            exp = cell.get("expected_cypher") or ""
            runs = cell.get("runs") or []
            run_rows: List[str] = []
            for i, r in enumerate(runs, start=1):
                rscore = float(r.get("score") or 0.0)
                rcls = score_class(rscore)
                rtok = int(r.get("tokens") or 0)
                rdur = float(r.get("duration") or 0.0)
                rerr = r.get("error") or ""
                rstart = r.get("started_at") or ""
                rmodel = r.get("model") or ""
                rcand = r.get("agent_cypher") or ""
                rreason = r.get("reason") or ""
                sub = (
                    f"<details><summary>Show query</summary>"
                    f"<div><b>Reasoning:</b> {rreason}</div>"
                    f"<div><b>Candidate:</b><pre class=mono>{rcand}</pre></div>"
                    f"</details>"
                )
                run_rows.append(
                    f"<tr>"
                    f"<td>{i}</td>"
                    f"<td><span class='score {rcls}'>{rscore:.2f}</span></td>"
                    f"<td>{rtok:,}</td>"
                    f"<td>{rdur:.2f}s</td>"
                    f"<td>{rmodel}</td>"
                    f"<td class=mono>{rstart}</td>"
                    f"<td>{'—' if not rerr else rerr}</td>"
                    f"<td>{sub}</td>"
                    f"</tr>"
                )
            runs_table = (
                "<table><thead><tr>"
                "<th>#</th><th>Score</th><th>Tokens</th><th>Time</th><th>Model</th><th>Started</th><th>Error</th><th>Details</th>"
                "</tr></thead><tbody>" + "".join(run_rows) + "</tbody></table>"
            )
            details = (
                f"<details><summary>Details</summary>"
                f"<div><b>Category:</b> {cat}</div>"
                f"<div><b>Expected:</b><pre class=mono>{exp}</pre></div>"
                f"<div><b>Runs:</b> {len(runs)}</div>"
                f"{runs_table}"
                f"</details>"
            )
            cells.append(
                f"<td>"
                f"<div><span class='score {cls}'>Score: {s:.2f}</span></div>"
                f"<div>Runs: {int(cell.get('count', 0))}</div>"
                f"<div>Avg tokens: {int(round(tokens)):,}</div>"
                f"<div>Avg time: {dur:.2f}s</div>"
                f"{details}"
                f"</td>"
            )
        rows_html.append("<tr>" + "".join(cells) + "</tr>")

    # Provider stats block
    stats_html = []
    for p in providers:
        st = provider_stats.get(p) or {}
        stats_html.append(
            f"<div class=stat><b>{p}</b> — Avg score: {st.get('avg_score', 0):.3f} · "
            f"Tokens: {int(st.get('total_tokens',0)):,} · Time: {float(st.get('total_duration',0.0)):.1f}s · "
            f"Queries: {int(st.get('queries',0))} · Runs: {int(st.get('runs',0))}</div>"
        )

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>text2Cypher Evaluation Report</title>
  <style>{css}</style>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta http-equiv="X-UA-Compatible" content="IE=edge" />
  <meta name="robots" content="noindex,nofollow" />
  <script>
    // Simple client-side filter by provider min score
    function filterByMinScore() {{
      const min = parseFloat(document.getElementById('minScore').value || '0');
      const rows = document.querySelectorAll('tbody tr');
      rows.forEach(tr => {{
        let ok = false;
        tr.querySelectorAll('td').forEach((td, i) => {{
          if (i === 0) return; // skip query column
          const m = td.textContent.match(/Score: ([0-9.]+)/);
          if (m && parseFloat(m[1]) >= min) ok = true;
        }});
        tr.style.display = ok ? '' : 'none';
      }});
    }}
  </script>
  </head>
<body>
  <h1>text2Cypher Evaluation Report</h1>
  <div class=meta>Generated by tools/evaluate_results.py</div>
  <div class=stats>{''.join(stats_html)}</div>
  <div style="margin: 8px 0;">
    <label>Min score filter: <input type="number" id="minScore" value="0" min="0" max="1" step="0.05" oninput="filterByMinScore()"/></label>
  </div>
  <table>
    <thead><tr>{header_cells}</tr></thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</body>
</html>
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(html)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=".", help="Directory of evaluation JSONs")
    parser.add_argument("--out-dir", default="report", help="Directory to write scores.json and index.html")
    parser.add_argument("--use-llm", action="store_true", help="Use configured LLM for judging (fallback applies)")
    parser.add_argument("--llm-token-budget", type=int, default=0, help="Optional total token budget for LLM judging (0 = unlimited)")
    parser.add_argument("--llm-min-tokens-per-call", type=int, default=10, help="Minimum estimated tokens per LLM judge invocation; used to stop invoking when remaining budget is lower")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    if not in_dir.exists() or not in_dir.is_dir():
        print(f"Input directory not found: {in_dir}")
        return 2

    entries = _read_all_entries(in_dir)
    if not entries:
        print("No evaluation entries found in directory.")
        return 0

    llm_token_budget = int(args.llm_token_budget) if int(args.llm_token_budget or 0) > 0 else None
    # Print selected judge model info (best-effort) when using LLM
    if args.use_llm:
        trial_llm = get_llm()
        model_desc = None
        if trial_llm is not None:
            model_desc = getattr(trial_llm, "model", getattr(trial_llm, "model_name", None))
        print(f"Using LLM judge: provider={MODEL_PROVIDER}, model={model_desc or 'unknown'}, token_budget={llm_token_budget or 'unlimited'}")
    graded, skipped_no_query, impossible_questions = _grade_entries(entries, use_llm=args.use_llm, llm_token_budget=llm_token_budget, llm_min_tokens_per_call=args.llm_min_tokens_per_call)
    matrix = _aggregate_for_report(graded)
    _write_scores_json(out_dir, graded, skipped_no_query, impossible_questions, matrix)
    _render_html(out_dir, matrix)

    # Calculate final percentage of valid Cypher queries (only counting evaluated queries)
    total_queries = len(graded)
    valid_queries = sum(1 for entry in graded if entry.get("similarity_score", 0) >= 0.5)
    total_evaluated = total_queries + len(skipped_no_query) + len(impossible_questions)
    valid_percentage = (valid_queries / total_queries * 100) if total_queries > 0 else 0

    print(f"\n{'='*80}")
    print(f"FINAL EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Total queries in dataset: {total_evaluated}")
    print(f"  - Evaluated by LLM judge: {total_queries}")
    print(f"  - Skipped (no query generated): {len(skipped_no_query)}")
    print(f"  - Impossible questions (schema limits): {len(impossible_questions)}")
    print(f"")
    print(f"Of evaluated queries ({total_queries}):")
    print(f"  - Valid Cypher queries (score >= 0.5): {valid_queries}")
    print(f"  - Invalid Cypher queries (score < 0.5): {total_queries - valid_queries}")
    print(f"  - PERCENTAGE OF CORRECT CYPHER: {valid_percentage:.2f}%")
    print(f"{'='*80}")
    print(f"\nWrote: {out_dir / 'scores.json'} and {out_dir / 'index.html'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
