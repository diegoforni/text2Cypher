#!/bin/bash
# Run evaluator and stop after reaching token limit

TOKEN_LIMIT=1000000
OUTPUT_FILE="gpt-5.2_results.json"
LOG_FILE="gpt-5.2_run.log"

echo "Starting evaluation with ${TOKEN_LIMIT} token limit..."
echo "Output: ${OUTPUT_FILE}"
echo "Log: ${LOG_FILE}"
echo "PID: $$"
echo ""

MODEL_PROVIDER=openai OPENAI_MODEL=gpt-5.2 .venv/bin/python3.12 tools/evaluator.py --output "$OUTPUT_FILE" --runs 1 2>&1 | tee "$LOG_FILE"

# Check final token count
TOKENS_USED=$(cat "$OUTPUT_FILE" 2>/dev/null | jq '.summary.total_tokens // 0')
echo ""
echo "=========================================="
echo "EVALUATION COMPLETE"
echo "=========================================="
echo "Tokens used: ${TOKENS_USED}"
echo "Token limit: ${TOKEN_LIMIT}"
if [ "$TOKENS_USED" -gt "$TOKEN_LIMIT" ]; then
    echo "⚠️  WARNING: Exceeded token limit!"
else
    echo "✅ Within token limit"
fi
echo "Results: ${OUTPUT_FILE}"
echo "=========================================="
