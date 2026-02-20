#!/usr/bin/env python3
"""
Validate Cypher queries using EXPLAIN to test syntax without running queries.

This script reads JSON files from the gpt-5.2_results_950k directory,
validates each agent_cypher query using Cypher's EXPLAIN command,
and outputs the validation results.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


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
        except Exception as e:
            print(f"Warning: Could not parse {p}: {e}", file=sys.stderr)
            continue

        # Handle {"summary": {...}, "results": [...]} format
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


def _cypher_explain(cypher_query: str, neo4j_uri: str = "bolt://localhost:7688",
                    username: str = "neo4j", password: str = "password") -> Tuple[bool, str]:
    """
    Validate a Cypher query using EXPLAIN.

    Returns (is_valid, error_message_or_plan).
    """
    if not cypher_query or not cypher_query.strip():
        return False, "Empty query"

    # Escape single quotes in the query
    escaped_query = cypher_query.replace("\\", "\\\\").replace("'", "\\'")

    cypher_command = f"EXPLAIN {escaped_query}"

    # Use cypher-shell to validate the query
    cmd = [
        "docker", "exec", "neo4j-b",
        "cypher-shell",
        "-u", username,
        "-p", password,
        "-a", neo4j_uri,
        "--format", "plain",
        cypher_command
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )

        # EXPLAIN should succeed even if the query would return no results
        # If it fails, it's likely a syntax error or invalid schema reference
        if result.returncode == 0:
            return True, result.stdout
        else:
            error_output = result.stderr or result.stdout or "Unknown error"
            # Check for common error patterns
            if "Invalid input" in error_output or "syntax error" in error_output.lower():
                return False, f"Syntax error: {error_output}"
            elif "Type mismatch" in error_output or "already declared" in error_output:
                return False, f"Semantic error: {error_output}"
            else:
                return False, f"Execution error: {error_output}"
    except subprocess.TimeoutExpired:
        return False, "Timeout during EXPLAIN"
    except Exception as e:
        return False, f"Exception: {str(e)}"


def _validate_entries(entries: List[Dict[str, Any]],
                      neo4j_uri: str = "bolt://localhost:7688",
                      username: str = "neo4j",
                      password: str = "password") -> List[Dict[str, Any]]:
    """Validate each entry's agent_cypher using EXPLAIN."""
    validated = []
    total = len(entries)

    for idx, item in enumerate(entries, 1):
        agent_cypher = item.get("agent_cypher", "")
        prompt = item.get("prompt", "")
        category = item.get("category", "")
        source_file = item.get("_source_file", "")

        print(f"[{idx}/{total}] Validating: {prompt[:60]}...", file=sys.stderr)

        enriched = dict(item)

        if not agent_cypher:
            enriched["cypher_valid"] = False
            enriched["cypher_validation_error"] = "No query generated"
            enriched["cypher_explain_plan"] = None
        else:
            is_valid, result = _cypher_explain(agent_cypher, neo4j_uri, username, password)
            enriched["cypher_valid"] = is_valid
            if is_valid:
                enriched["cypher_validation_error"] = None
                enriched["cypher_explain_plan"] = result[:500] if result else ""  # Truncate plan
            else:
                enriched["cypher_validation_error"] = result
                enriched["cypher_explain_plan"] = None

        validated.append(enriched)

    return validated


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Cypher queries using EXPLAIN without running them"
    )
    parser.add_argument(
        "--input-dir",
        default="gpt-5.2_results_950k",
        help="Directory containing result JSON files"
    )
    parser.add_argument(
        "--output",
        default="cypher_validation_results.json",
        help="Output file for validation results"
    )
    parser.add_argument(
        "--neo4j-uri",
        default="bolt://localhost:7688",
        help="Neo4j Bolt URI"
    )
    parser.add_argument(
        "--username",
        default="neo4j",
        help="Neo4j username"
    )
    parser.add_argument(
        "--password",
        default="password",
        help="Neo4j password"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    print(f"Reading entries from {input_dir}...", file=sys.stderr)
    entries = _read_all_entries(input_dir)

    if not entries:
        print("No entries found in input directory", file=sys.stderr)
        return 0

    print(f"Validating {len(entries)} Cypher queries using EXPLAIN...", file=sys.stderr)
    print(f"Neo4j URI: {args.neo4j_uri}", file=sys.stderr)

    validated = _validate_entries(
        entries,
        neo4j_uri=args.neo4j_uri,
        username=args.username,
        password=args.password
    )

    # Calculate statistics
    valid_count = sum(1 for v in validated if v.get("cypher_valid", False))
    invalid_count = len(validated) - valid_count

    print(f"\n{'='*80}", file=sys.stderr)
    print(f"VALIDATION SUMMARY", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    print(f"Total queries: {len(validated)}", file=sys.stderr)
    print(f"Valid Cypher (EXPLAIN succeeded): {valid_count}", file=sys.stderr)
    print(f"Invalid Cypher (EXPLAIN failed): {invalid_count}", file=sys.stderr)
    print(f"Validity rate: {valid_count / len(validated) * 100:.2f}%", file=sys.stderr)
    print(f"{'='*80}\n", file=sys.stderr)

    # Write output
    output_path = Path(args.output)
    output_path.write_text(json.dumps(validated, indent=2))
    print(f"Results written to: {output_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
