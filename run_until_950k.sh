#!/bin/bash
# Run evaluator repeatedly until reaching 950K tokens

TOKEN_LIMIT=950000
BUFFER=50000
STOP_LIMIT=$((TOKEN_LIMIT - BUFFER))  # Stop at 900K to be safe
MODEL="gpt-5.2"
OUTPUT_DIR="gpt-5.2_results_950k"
LOG_FILE="gpt-5.2_950k_run.log"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Multi-Run Evaluation until Token Limit"
echo "=========================================="
echo "Model: $MODEL"
echo "Token limit: ${TOKEN_LIMIT} (will stop at: ${STOP_LIMIT})"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Started at: $(date)"
echo "PID: $$"
echo "=========================================="
echo ""

run_number=1
total_tokens=0

while [ $total_tokens -lt $STOP_LIMIT ]; do
    output_file="${OUTPUT_DIR}/run_${run_number}.json"

    echo "==========================================" | tee -a "$LOG_FILE"
    echo "Starting Run #${run_number}" | tee -a "$LOG_FILE"
    echo "Output: ${output_file}" | tee -a "$LOG_FILE"
    echo "Tokens used so far: ${total_tokens}" | tee -a "$LOG_FILE"
    echo "Tokens remaining: $((STOP_LIMIT - total_tokens))" | tee -a "$LOG_FILE"
    echo "Started at: $(date)" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"

    # Run the evaluator
    MODEL_PROVIDER=openai OPENAI_MODEL="$MODEL" .venv/bin/python3.12 tools/evaluator.py \
        --output "$output_file" \
        --runs 1 2>&1 | tee -a "$LOG_FILE"

    # Get tokens used in this run
    if [ -f "$output_file" ]; then
        run_tokens=$(cat "$output_file" | jq '.summary.total_tokens // 0')
        total_tokens=$((total_tokens + run_tokens))

        echo "" | tee -a "$LOG_FILE"
        echo "Run #${run_number} completed" | tee -a "$LOG_FILE"
        echo "Tokens in this run: ${run_tokens}" | tee -a "$LOG_FILE"
        echo "Total tokens: ${total_tokens} / ${TOKEN_LIMIT}" | tee -a "$LOG_FILE"
        echo "Progress: $((total_tokens * 100 / TOKEN_LIMIT))%" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"

        run_number=$((run_number + 1))

        # Check if we've reached the limit
        if [ $total_tokens -ge $STOP_LIMIT ]; then
            echo "==========================================" | tee -a "$LOG_FILE"
            echo "TOKEN LIMIT REACHED!" | tee -a "$LOG_FILE"
            echo "Total tokens: ${total_tokens}" | tee -a "$LOG_FILE"
            echo "Stopped at: $(date)" | tee -a "$LOG_FILE"
            echo "Total runs: $((run_number - 1))" | tee -a "$LOG_FILE"
            echo "==========================================" | tee -a "$LOG_FILE"
            break
        fi

        # Small delay between runs
        sleep 5
    else
        echo "ERROR: Output file not created. Stopping." | tee -a "$LOG_FILE"
        break
    fi
done

# Final summary
echo ""
echo "==========================================" | tee -a "$LOG_FILE"
echo "FINAL SUMMARY" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "Total runs completed: $((run_number - 1))" | tee -a "$LOG_FILE"
echo "Total tokens used: ${total_tokens}" | tee -a "$LOG_FILE"
echo "Token limit: ${TOKEN_LIMIT}" | tee -a "$LOG_FILE"
echo "Percentage: $((total_tokens * 100 / TOKEN_LIMIT))%" | tee -a "$LOG_FILE"
echo "Output directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "Results: $(ls -1 "$OUTPUT_DIR" | wc -l) files" | tee -a "$LOG_FILE"
echo "Ended at: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
