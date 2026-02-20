#!/bin/bash
# Worker script - runs evaluator until token limit

WORKER_ID=$1
TOKEN_LIMIT=$2
STOP_LIMIT=$3
MODEL=$4
OUTPUT_BASE=$5

log_file="${OUTPUT_BASE}_worker${WORKER_ID}.log"
token_file="${OUTPUT_BASE}_worker${WORKER_ID}_tokens.txt"

echo "[DEBUG] Worker ${WORKER_ID} started" >> "${log_file}"

# Find next available run number in the shared directory
run_number=$(ls -1 "${OUTPUT_BASE}"/run_*.json 2>/dev/null | grep -o 'run_[0-9]*' | grep -o '[0-9]*' | sort -n | tail -1)
if [ -z "$run_number" ]; then
    run_number=1
else
    run_number=$((run_number + 1))
fi

# Track tokens added in THIS execution only
execution_tokens=0

while [ $execution_tokens -lt $STOP_LIMIT ]; do
    # Both workers check for the next available run number to avoid conflicts
    run_number=$(ls -1 "${OUTPUT_BASE}"/run_*.json 2>/dev/null | grep -o 'run_[0-9]*' | grep -o '[0-9]*' | sort -n | tail -1)
    if [ -z "$run_number" ]; then
        run_number=1
    else
        run_number=$((run_number + 1))
    fi

    output_file="${OUTPUT_BASE}/run_${run_number}.json"

    echo "[$(date)] Worker ${WORKER_ID}: Starting Run #${run_number}" | tee -a "$log_file"
    echo "[$(date)] Worker ${WORKER_ID}: Tokens in this execution: ${execution_tokens}" | tee -a "$log_file"

    # Run the evaluator (must be in text2Cypher directory for queries.json)
    cd /srv/neo4j/text2Cypher
    MODEL_PROVIDER=openai OPENAI_MODEL="$MODEL" .venv/bin/python3.12 tools/evaluator.py \
        --output "$output_file" \
        --runs 1 >> "$log_file" 2>&1

    # Get tokens used in this run
    if [ -f "$output_file" ]; then
        run_tokens=$(cat "$output_file" | jq '.summary.total_tokens // 0')
        execution_tokens=$((execution_tokens + run_tokens))

        echo "[$(date)] Worker ${WORKER_ID}: Run #${run_number} completed" | tee -a "$log_file"
        echo "[$(date)] Worker ${WORKER_ID}: Tokens in this run: ${run_tokens}" | tee -a "$log_file"
        echo "[$(date)] Worker ${WORKER_ID}: Execution tokens: ${execution_tokens}" | tee -a "$log_file"
        echo "[$(date)] Worker ${WORKER_ID}: Progress: $((execution_tokens * 100 / STOP_LIMIT))%" | tee -a "$log_file"

        # Check if we've reached the limit
        if [ $execution_tokens -ge $STOP_LIMIT ]; then
            echo "[$(date)] Worker ${WORKER_ID}: Limit reached!" | tee -a "$log_file"
            break
        fi

        sleep 2
    else
        echo "[$(date)] Worker ${WORKER_ID}: ERROR - Output file not created" | tee -a "$log_file"
        break
    fi
done

# Write the final token count to the token file
echo "$execution_tokens" > "$token_file"
echo "[DEBUG] Worker ${WORKER_ID} finished with ${execution_tokens} tokens" >> "${log_file}"
