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

from config import get_llm  # type: ignore


def _read_all_entries(input_dir: Path) -> List[Dict[str, Any]]:
    """Read all JSON arrays in directory and concatenate entries.

    Supports files where root is a list of entries. Ignores files whose
    content cannot be parsed as JSON or is not a list.
    """
    entries: List[Dict[str, Any]] = []
    for p in sorted(input_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # Attach the source file path for possible disambiguation
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
    """Ask the LLM to grade semantic and structural similarity.

    Returns (score_float_0_1, reasoning).
    """
    system = (
        "You are a strict Cypher evaluation judge. Given an expected query and a candidate,"
        " assign a similarity score in [0,1] that reflects whether the candidate would answer"
        " the same question on the same schema. Focus on: MATCH patterns, WHERE filters,"
        " grouping/aggregations, ORDER BY, LIMIT, and returned fields. Ignore aliases, minor"
        " formatting, whitespace, capitalization, and synonymous property names when harmless."
        " Heavily penalize missing key filters, wrong joins, wrong aggregations, or omitted LIMITs"
        " when the expected has them."
    )
    user = f"""
Task category: {category}
Prompt/question: {prompt}

Expected Cypher:
{expected}

Candidate Cypher:
{candidate}

Respond with a single JSON object with keys:
- score: number in [0,1], rounded to 2 decimals
- reasoning: brief explanation
"""
    try:
        resp = llm.invoke([("system", system), ("user", user)])
        content = getattr(resp, "content", str(resp)).strip()
        # Attempt to extract JSON object
        start, end = content.find("{"), content.rfind("}")
        if start != -1 and end != -1 and end > start:
            content = content[start : end + 1]
        data = json.loads(content)
        score = float(data.get("score", 0))
        score = max(0.0, min(1.0, score))
        reasoning = str(data.get("reasoning", "")).strip()
        return score, reasoning
    except Exception as e:
        # On failure, return a conservative fallback using token overlap
        exp_n = _normalize_query(expected)
        cand_n = _normalize_query(candidate)
        return round(_jaccard_similarity(exp_n, cand_n), 2), f"fallback: {e}"


def _grade_entries(entries: List[Dict[str, Any]], use_llm: bool) -> List[Dict[str, Any]]:
    """Compute similarity for each entry, returning new list with score and reasoning."""
    llm = get_llm() if use_llm else None
    graded: List[Dict[str, Any]] = []
    for item in entries:
        expected = _normalize_query(str(item.get("expected_cypher", "")))
        candidate = _normalize_query(str(item.get("agent_cypher", "")))
        prompt = str(item.get("prompt", ""))
        category = str(item.get("category", ""))

        if not expected and not candidate:
            score, reasoning = 0.0, "no expected or candidate"
        elif not candidate:
            score, reasoning = 0.0, "empty candidate"
        elif not expected:
            # Rare, but avoid giving free credit
            score, reasoning = 0.0, "missing expected"
        else:
            if llm is not None:
                score, reasoning = _llm_judge(llm, expected, candidate, prompt, category)
            else:
                score = round(_jaccard_similarity(expected, candidate), 2)
                reasoning = "deterministic token overlap"

        enriched = dict(item)
        enriched["similarity_score"] = float(round(score, 2))
        enriched["judge_reasoning"] = reasoning
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
    return graded


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


def _write_scores_json(out_dir: Path, graded: List[Dict[str, Any]], matrix: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "entries": graded,
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
                f"<div class=score { 'title="score"' }><span class='score {cls}'>Score: {s:.2f}</span></div>"
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

    graded = _grade_entries(entries, use_llm=args.use_llm)
    matrix = _aggregate_for_report(graded)
    _write_scores_json(out_dir, graded, matrix)
    _render_html(out_dir, matrix)
    print(f"Wrote: {out_dir / 'scores.json'} and {out_dir / 'index.html'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
