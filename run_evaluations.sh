#!/bin/bash

# Activate virtual environment
source .venv/bin/activate

# Set environment variables
export MODEL_PROVIDER=openai
export OPENAI_MODEL=gpt-5.1

# Create logs directory if it doesn't exist
mkdir -p new_openai_evals/logs

# Run first 2 parallel evaluator executions
echo "Starting first 2 parallel evaluator executions with OpenAI GPT-5.1..."

nohup python3.12 tools/evaluator.py --output new_openai_evals/gpt5_1_execution_5.json > new_openai_evals/logs/gpt5_1_execution_5.log 2>&1 &
PID1=$!
echo "Execution 1 started with PID: $PID1"

nohup python3.12 tools/evaluator.py --output new_openai_evals/gpt5_1_execution_2.json > new_openai_evals/logs/gpt5_1_execution_2.log 2>&1 &
PID2=$!
echo "Execution 2 started with PID: $PID2"

# Wait for first 2 to complete
echo "Waiting for first 2 executions to complete..."
wait $PID1 $PID2

echo "First 2 executions completed. Starting next 2..."

# Run second 2 parallel evaluator executions
nohup python3.12 tools/evaluator.py --output new_openai_evals/gpt5_1_execution_3.json > new_openai_evals/logs/gpt5_1_execution_3.log 2>&1 &
PID3=$!
echo "Execution 3 started with PID: $PID3"

nohup python3.12 tools/evaluator.py --output new_openai_evals/gpt5_1_execution_4.json > new_openai_evals/logs/gpt5_1_execution_4.log 2>&1 &
PID4=$!
echo "Execution 4 started with PID: $PID4"

# Wait for second 2 to complete
echo "Waiting for last 2 executions to complete..."
wait $PID3 $PID4

echo "All 4 evaluator executions completed!"
echo "Results saved to: new_openai_evals/gpt5_1_execution_*.json"
echo "Logs saved to: new_openai_evals/logs/gpt5_1_execution_*.log"