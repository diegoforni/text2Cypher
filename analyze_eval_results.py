"""
Analyze evaluation results to understand patterns and suggest improvements.
Randomly samples 3 cycles for each query and analyzes the generated Cypher.
"""

import json
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

def load_all_results(eval_dir: Path) -> List[Dict[str, Any]]:
    """Load all evaluation results from the directory."""
    all_results = []
    for json_file in sorted(eval_dir.glob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
                for entry in data:
                    entry['_source_file'] = json_file.name
                    all_results.append(entry)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    return all_results

def group_by_query(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group results by query prompt."""
    grouped = defaultdict(list)
    for entry in results:
        prompt = entry.get('prompt', '')
        if prompt:
            grouped[prompt].append(entry)
    return grouped

def sample_queries(grouped: Dict[str, List[Dict[str, Any]]], sample_size: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """Sample N entries for each query."""
    sampled = {}
    for prompt, entries in grouped.items():
        if len(entries) <= sample_size:
            sampled[prompt] = entries
        else:
            sampled[prompt] = random.sample(entries, sample_size)
    return sampled

def analyze_query_patterns(sampled: Dict[str, List[Dict[str, Any]]]) -> None:
    """Analyze patterns in generated queries vs expected queries."""
    
    print("="*100)
    print("ANALYSIS OF EVALUATION RESULTS")
    print("="*100)
    print()
    
    issues = defaultdict(list)
    
    for idx, (prompt, entries) in enumerate(sampled.items(), 1):
        print(f"\n{'='*100}")
        print(f"QUERY #{idx}: {prompt}")
        print(f"{'='*100}")
        
        # Get the expected query (should be same across all entries)
        expected = entries[0].get('expected_cypher', '')
        category = entries[0].get('category', 'unknown')
        
        print(f"\nCategory: {category}")
        print(f"\nExpected Cypher:")
        print(expected)
        print()
        
        for i, entry in enumerate(entries, 1):
            print(f"\n--- Sample {i} (from {entry.get('_source_file', 'unknown')}) ---")
            agent_cypher = entry.get('agent_cypher', '')
            print(f"Generated Cypher:")
            print(agent_cypher)
            
            # Check for common issues
            issues_found = []
            
            # Issue 1: Missing LIMIT when expected has it
            if 'LIMIT' in expected.upper() and 'LIMIT' not in agent_cypher.upper():
                issues_found.append("Missing LIMIT clause")
                issues['missing_limit'].append({
                    'prompt': prompt,
                    'expected': expected,
                    'generated': agent_cypher
                })
            
            # Issue 2: Different aggregation approach
            if 'count(' in expected.lower() and 'count(' not in agent_cypher.lower():
                issues_found.append("Missing COUNT aggregation")
                issues['missing_aggregation'].append({
                    'prompt': prompt,
                    'expected': expected,
                    'generated': agent_cypher
                })
            
            # Issue 3: Extra WHERE clauses not in expected
            expected_where_count = expected.upper().count('WHERE')
            agent_where_count = agent_cypher.upper().count('WHERE')
            if agent_where_count > expected_where_count:
                issues_found.append(f"Extra WHERE clause(s) ({agent_where_count} vs {expected_where_count})")
                issues['extra_where'].append({
                    'prompt': prompt,
                    'expected': expected,
                    'generated': agent_cypher
                })
            
            # Issue 4: Different MATCH pattern
            if expected.split('MATCH')[1].split('\n')[0] if 'MATCH' in expected else '' != \
               agent_cypher.split('MATCH')[1].split('\n')[0] if 'MATCH' in agent_cypher else '':
                issues_found.append("Different MATCH pattern")
            
            # Issue 5: Using WITH vs direct aggregation
            expected_has_with = 'WITH' in expected.upper()
            agent_has_with = 'WITH' in agent_cypher.upper()
            if expected_has_with != agent_has_with:
                issues_found.append(f"WITH clause mismatch (expected: {expected_has_with}, got: {agent_has_with})")
                issues['with_mismatch'].append({
                    'prompt': prompt,
                    'expected': expected,
                    'generated': agent_cypher
                })
            
            # Issue 6: ORDER BY differences
            if 'ORDER BY' in expected.upper() and 'ORDER BY' not in agent_cypher.upper():
                issues_found.append("Missing ORDER BY")
                issues['missing_order_by'].append({
                    'prompt': prompt,
                    'expected': expected,
                    'generated': agent_cypher
                })
            
            # Issue 7: Filtering unknown values when not needed
            if "'unknown'" in agent_cypher.lower() and "'unknown'" not in expected.lower():
                issues_found.append("Unnecessary 'unknown' filter added")
                issues['unnecessary_unknown_filter'].append({
                    'prompt': prompt,
                    'expected': expected,
                    'generated': agent_cypher
                })
            
            if issues_found:
                print(f"\n⚠️  Issues detected:")
                for issue in issues_found:
                    print(f"   - {issue}")
            else:
                print(f"\n✓ No major issues detected")
            
            # Check if query executed successfully
            error = entry.get('error')
            if error:
                print(f"\n❌ Error: {error}")
                issues['execution_errors'].append({
                    'prompt': prompt,
                    'error': error,
                    'generated': agent_cypher
                })
            else:
                response = entry.get('response', [])
                print(f"\n✓ Query executed successfully, returned {len(response)} row(s)")
    
    # Print summary of issues
    print(f"\n\n{'='*100}")
    print("SUMMARY OF COMMON ISSUES")
    print(f"{'='*100}\n")
    
    for issue_type, occurrences in sorted(issues.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\n{issue_type.upper().replace('_', ' ')}: {len(occurrences)} occurrences")
        print("-" * 80)
        if len(occurrences) > 0:
            print(f"Example from: '{occurrences[0]['prompt']}'")
            if 'expected' in occurrences[0]:
                print(f"\nExpected:\n{occurrences[0]['expected']}")
            if 'generated' in occurrences[0]:
                print(f"\nGenerated:\n{occurrences[0]['generated']}")
            if 'error' in occurrences[0]:
                print(f"\nError: {occurrences[0]['error']}")

def generate_recommendations(issues: Dict[str, List]) -> None:
    """Generate recommendations for improving the system prompt."""
    print(f"\n\n{'='*100}")
    print("RECOMMENDATIONS FOR SYSTEM PROMPT IMPROVEMENTS")
    print(f"{'='*100}\n")
    
    recommendations = []
    
    if 'unnecessary_unknown_filter' in issues and len(issues['unnecessary_unknown_filter']) > 3:
        recommendations.append({
            'priority': 'HIGH',
            'issue': 'Unnecessary filtering of "unknown" values',
            'recommendation': 'Modify the system prompt to NOT automatically filter out "unknown" values unless explicitly requested by the user. The prompt should clarify: "Only add WHERE clauses to filter unknown/null values if the user\'s question specifically requires it."'
        })
    
    if 'missing_limit' in issues and len(issues['missing_limit']) > 2:
        recommendations.append({
            'priority': 'MEDIUM',
            'issue': 'Missing LIMIT clauses',
            'recommendation': 'Emphasize in the system prompt: "When the question asks for \'the most\', \'the top\', or \'the highest/lowest\', always include a LIMIT clause with the appropriate number (default to LIMIT 1 for superlatives)."'
        })
    
    if 'with_mismatch' in issues and len(issues['with_mismatch']) > 2:
        recommendations.append({
            'priority': 'MEDIUM',
            'issue': 'Inconsistent use of WITH clauses',
            'recommendation': 'Add guidance: "Use WITH clauses for intermediate aggregations and transformations. When you need to aggregate first and then sort/limit, use: MATCH ... WITH aggregation ORDER BY ... LIMIT ... RETURN ..."'
        })
    
    if 'extra_where' in issues and len(issues['extra_where']) > 2:
        recommendations.append({
            'priority': 'HIGH',
            'issue': 'Adding extra WHERE clauses not required by the query',
            'recommendation': 'Strengthen the prompt: "Only add WHERE clauses that are explicitly required by the user\'s question. Do not add defensive filters unless the question specifically asks to exclude certain values."'
        })
    
    if 'missing_aggregation' in issues:
        recommendations.append({
            'priority': 'LOW',
            'issue': 'Missing or different aggregation functions',
            'recommendation': 'Clarify: "Pay close attention to aggregation requirements. Use count(*) for counting relationships/nodes, sum() for totals, and appropriate aggregation functions as needed."'
        })
    
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. [{rec['priority']}] {rec['issue']}")
        print(f"   Recommendation: {rec['recommendation']}")
        print()

def main():
    eval_dir = Path("/srv/neo4j/text2Cypher/eval_results")
    
    print("Loading evaluation results...")
    all_results = load_all_results(eval_dir)
    print(f"Loaded {len(all_results)} total entries")
    
    print("Grouping by query...")
    grouped = group_by_query(all_results)
    print(f"Found {len(grouped)} unique queries")
    
    print("Sampling 3 cycles per query...")
    random.seed(42)  # For reproducibility
    sampled = sample_queries(grouped, sample_size=3)
    
    analyze_query_patterns(sampled)

if __name__ == "__main__":
    main()
