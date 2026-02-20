#!/bin/bash
# Run 2 parallel evaluations until 950K tokens total

TOKEN_LIMIT=470000
BUFFER=30000
STOP_LIMIT=$((TOKEN_LIMIT - BUFFER))
MODEL="gpt-5.2"
OUTPUT_BASE="gpt-5.2_results_950k"
LOG_FILE="${OUTPUT_BASE}_main.log"

# Create output directory
mkdir -p "${OUTPUT_BASE}"

# Get starting token count (from existing files)
INITIAL_TOKENS=0
if ls "${OUTPUT_BASE}"/run_*.json 2>/dev/null | head -1 | grep -q .; then
    INITIAL_TOKENS=$(cat "${OUTPUT_BASE}"/run_*.json 2>/dev/null | jq -s '[.[].summary.total_tokens] | add' || echo "0")
fi
echo "Initial tokens in directory: ${INITIAL_TOKENS}"

echo "=========================================="
echo "2-Parallel Evaluation until Token Limit"
echo "=========================================="
echo "Model: $MODEL"
echo "Token limit: ${TOKEN_LIMIT} (will stop at: ${STOP_LIMIT})"
echo "Output base: $OUTPUT_BASE"
echo "Log file: $LOG_FILE"
echo "Started at: $(date)"
echo "=========================================="
echo ""

# Start both workers in background using the external worker script
echo "Starting Worker 1..."
/srv/neo4j/text2Cypher/worker.sh 1 $TOKEN_LIMIT $STOP_LIMIT "$MODEL" "/srv/neo4j/text2Cypher/${OUTPUT_BASE}" > "/srv/neo4j/text2Cypher/${OUTPUT_BASE}_worker1_status.log" 2>&1 &
worker1_pid=$!
echo "Worker 1 PID: $worker1_pid"

echo "Starting Worker 2..."
/srv/neo4j/text2Cypher/worker.sh 2 $TOKEN_LIMIT $STOP_LIMIT "$MODEL" "/srv/neo4j/text2Cypher/${OUTPUT_BASE}" > "/srv/neo4j/text2Cypher/${OUTPUT_BASE}_worker2_status.log" 2>&1 &
worker2_pid=$!
echo "Worker 2 PID: $worker2_pid"

echo ""
echo "Both workers started. PIDs: $worker1_pid, $worker2_pid"
echo "Output directory: ${OUTPUT_BASE}"
echo "Monitor logs:"
echo "  tail -f ${OUTPUT_BASE}_worker1.log"
echo "  tail -f ${OUTPUT_BASE}_worker2.log"
echo ""

# Wait for both workers and capture their token counts from token files
wait $worker1_pid
tokens1=$(cat "/srv/neo4j/text2Cypher/${OUTPUT_BASE}_worker1_tokens.txt" 2>/dev/null || echo "0")

wait $worker2_pid
tokens2=$(cat "/srv/neo4j/text2Cypher/${OUTPUT_BASE}_worker2_tokens.txt" 2>/dev/null || echo "0")

total_all=$((tokens1 + tokens2))

echo ""
echo "==========================================" | tee -a "$LOG_FILE"
echo "FINAL SUMMARY" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "Worker 1 tokens: ${tokens1}" | tee -a "$LOG_FILE"
echo "Worker 2 tokens: ${tokens2}" | tee -a "$LOG_FILE"
echo "Total tokens: ${total_all}" | tee -a "$LOG_FILE"
echo "Token limit: ${TOKEN_LIMIT}" | tee -a "$LOG_FILE"
echo "Percentage: $((total_all * 100 / TOKEN_LIMIT))%" | tee -a "$LOG_FILE"
echo "Output directory: ${OUTPUT_BASE}" | tee -a "$LOG_FILE"
echo "Ended at: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
