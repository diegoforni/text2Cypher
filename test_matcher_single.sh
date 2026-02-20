#!/bin/bash
# Test matcher with a single problematic query

MODEL="gpt-5.2"
OUTPUT_FILE="test_matcher_run_1.json"
PROMPTS_FILE="test_prompts_single.json"

echo "=========================================="
echo "Single Query Matcher Test"
echo "=========================================="
echo "Model: $MODEL"
echo "Prompts: $PROMPTS_FILE"
echo "Output: $OUTPUT_FILE"
echo "Started at: $(date)"
echo "=========================================="
echo ""

# Run the evaluator with our test prompts
MODEL_PROVIDER=openai OPENAI_MODEL="$MODEL" .venv/bin/python3.12 tools/evaluator.py \
    --output "$OUTPUT_FILE" \
    --runs 1 \
    --prompts "$PROMPTS_FILE" 2>&1 | tee test_matcher_run_1.log

echo ""
echo "=========================================="
echo "Test Complete"
echo "=========================================="
echo "Output: $OUTPUT_FILE"
echo "Log: test_matcher_run_1.log"
echo "Ended at: $(date)"
echo "=========================================="

# Show the result
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "Agent Cypher Generated:"
    cat "$OUTPUT_FILE" | jq -r '.results[0].agent_cypher'
    echo ""
    echo "Expected Cypher:"
    cat "$OUTPUT_FILE" | jq -r '.results[0].expected_cypher'
fi
