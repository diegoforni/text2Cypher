"""
Analyze evaluation results and provide detailed insights about failures and issues.

Usage:
  python tools/analyze_evaluation.py <evaluation_results.json>
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def analyze_query_status(result: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a single query result and determine its status."""
    index = result.get("index")
    category = result.get("category")
    prompt = result.get("prompt", "")[:80]
    
    status = {
        "index": index,
        "category": category,
        "prompt": prompt,
        "agent_cypher": result.get("agent_cypher"),
        "error": result.get("error"),
        "response_type": type(result.get("response")).__name__,
        "response_count": len(result.get("response")) if isinstance(result.get("response"), list) else None,
    }
    
    # Determine query execution status
    if result.get("error"):
        status["status"] = "ERROR"
        status["error_type"] = classify_error(result.get("error"))
    elif not result.get("agent_cypher") or result.get("agent_cypher") == "":
        status["status"] = "NO_QUERY_GENERATED"
        # Check generation trace for why
        gen_trace = result.get("agents", {}).get("generation", {}).get("trace", [])
        if gen_trace:
            attempts = []
            for subproblem in gen_trace:
                for attempt in subproblem.get("attempts", []):
                    attempts.append({
                        "fragment": attempt.get("fragment", "")[:100],
                        "ok": attempt.get("ok"),
                        "error": attempt.get("error"),
                    })
            status["generation_attempts"] = attempts
    elif isinstance(result.get("response"), list):
        if len(result.get("response")) == 0:
            status["status"] = "SUCCESS_ZERO_ROWS"
        else:
            status["status"] = "SUCCESS"
    elif result.get("response") is None:
        status["status"] = "EXECUTION_FAILED"
    else:
        status["status"] = "UNKNOWN"
    
    return status


def classify_error(error_msg: str) -> str:
    """Classify the type of error."""
    if not error_msg:
        return "UNKNOWN"
    
    error_lower = error_msg.lower()
    
    if "json serializable" in error_lower:
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


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument("input", help="Path to evaluation results JSON file")
    parser.add_argument("--output", help="Optional output file for detailed analysis")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed information")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return 1
    
    # Load results
    print(f"Loading evaluation results from {input_path}...")
    results = json.loads(input_path.read_text())
    
    print(f"\n{'='*80}")
    print(f"EVALUATION ANALYSIS REPORT")
    print(f"{'='*80}\n")
    
    print(f"Total queries: {len(results)}")
    
    # Analyze each result
    analyzed = [analyze_query_status(r) for r in results]
    
    # Group by status
    by_status = {}
    for a in analyzed:
        status = a["status"]
        if status not in by_status:
            by_status[status] = []
        by_status[status].append(a)
    
    # Print summary
    print("\nStatus Summary:")
    print("-" * 80)
    for status, items in sorted(by_status.items()):
        print(f"  {status:30s}: {len(items):3d} queries")
    
    # Detailed breakdown
    print("\n" + "="*80)
    print("DETAILED BREAKDOWN")
    print("="*80)
    
    # Successes with zero rows
    if "SUCCESS_ZERO_ROWS" in by_status:
        print(f"\n🟡 SUCCESS WITH ZERO ROWS ({len(by_status['SUCCESS_ZERO_ROWS'])} queries)")
        print("-" * 80)
        print("These queries executed successfully but returned no data.")
        print("This could mean:")
        print("  - The data doesn't exist in the database")
        print("  - The query logic is incorrect")
        print("  - Filters are too restrictive\n")
        
        for item in by_status["SUCCESS_ZERO_ROWS"]:
            print(f"\nIndex {item['index']} ({item['category']}): {item['prompt']}")
            if args.verbose and item.get("agent_cypher"):
                print(f"Query:\n{item['agent_cypher']}\n")
    
    # Errors
    if "ERROR" in by_status:
        print(f"\n🔴 ERRORS ({len(by_status['ERROR'])} queries)")
        print("-" * 80)
        
        # Group errors by type
        errors_by_type = {}
        for item in by_status["ERROR"]:
            error_type = item.get("error_type", "UNKNOWN")
            if error_type not in errors_by_type:
                errors_by_type[error_type] = []
            errors_by_type[error_type].append(item)
        
        for error_type, items in sorted(errors_by_type.items()):
            print(f"\n  {error_type} ({len(items)} queries):")
            for item in items:
                print(f"    Index {item['index']} ({item['category']}): {item['prompt']}")
                print(f"    Error: {item['error']}")
                if args.verbose and item.get("agent_cypher"):
                    print(f"    Query: {item['agent_cypher'][:200]}...")
                print()
    
    # No query generated
    if "NO_QUERY_GENERATED" in by_status:
        print(f"\n🟠 NO QUERY GENERATED ({len(by_status['NO_QUERY_GENERATED'])} queries)")
        print("-" * 80)
        print("These queries failed during generation phase.\n")
        
        for item in by_status["NO_QUERY_GENERATED"]:
            print(f"\nIndex {item['index']} ({item['category']}): {item['prompt']}")
            
            if item.get("generation_attempts"):
                print(f"  Generation attempts: {len(item['generation_attempts'])}")
                for i, attempt in enumerate(item["generation_attempts"], 1):
                    print(f"\n  Attempt {i}:")
                    print(f"    OK: {attempt['ok']}")
                    if attempt.get("error"):
                        print(f"    Error: {attempt['error']}")
                    if args.verbose and attempt.get("fragment"):
                        print(f"    Fragment: {attempt['fragment']}...")
    
    # Successes
    if "SUCCESS" in by_status:
        print(f"\n✅ SUCCESS ({len(by_status['SUCCESS'])} queries)")
        print("-" * 80)
        if not args.verbose:
            print("Use --verbose to see successful queries")
        else:
            for item in by_status["SUCCESS"]:
                print(f"\nIndex {item['index']} ({item['category']}): {item['prompt']}")
                print(f"  Returned {item['response_count']} rows")
    
    # Output detailed analysis if requested
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(analyzed, indent=2))
        print(f"\nDetailed analysis written to: {output_path}")
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # Identify key issues
    print("\n1. JSON Serialization Errors:")
    print("   - Caused by Neo4j Date objects that can't be directly serialized to JSON")
    print("   - Solution: Convert dates to strings in the query (e.g., toString(date))")
    
    print("\n2. Empty Query Errors:")
    print("   - Agent failed to generate a valid query after multiple attempts")
    print("   - Common causes: missing parameters, memory errors, complex logic")
    
    print("\n3. Zero Row Results:")
    print("   - Query executed but returned no data")
    print("   - May indicate incorrect query logic or missing data")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
