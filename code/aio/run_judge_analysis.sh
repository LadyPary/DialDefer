#!/bin/bash
# Run analysis for judge experiments
# Usage: ./run_judge_analysis.sh <model> <input_stem> [baseline_file]
# Example: ./run_judge_analysis.sh qwen-2.5-7b-instruct finalAIOdata_experiment_ready results/baseline_speaker_c_results.jsonl

MODEL_NAME=${1:-"qwen-2.5-7b-instruct"}
INPUT_STEM=${2:-"finalAIOdata_experiment_ready"}
BASELINE_FILE=${3:-""}  # Optional baseline for C1 when analyzing A judges B

echo "======================================"
echo "Running analysis for model: $MODEL_NAME"
echo "======================================"

# File paths
A_JUDGES_B_FILE="results/${INPUT_STEM}_${MODEL_NAME}_speaker_a_judges_speaker_b_c2only_results.jsonl"
B_JUDGES_A_FILE="results/${INPUT_STEM}_${MODEL_NAME}_speaker_b_judges_speaker_a_results.jsonl"

echo ""
echo ">>> Analysis 1: Speaker B judges Speaker A (same file - has both C1 and C2)"
if [ -f "$B_JUDGES_A_FILE" ]; then
    python aio_analyzer.py \
        --input "$B_JUDGES_A_FILE" \
        --output "analysis/${INPUT_STEM}_${MODEL_NAME}_speaker_b_judges_speaker_a_analysis.csv"
else
    echo "File not found: $B_JUDGES_A_FILE"
fi

echo ""
echo ">>> Analysis 2: Speaker A judges Speaker B (C2 only - needs baseline for C1)"
if [ -f "$A_JUDGES_B_FILE" ]; then
    if [ -n "$BASELINE_FILE" ] && [ -f "$BASELINE_FILE" ]; then
        echo "Using baseline file for C1: $BASELINE_FILE"
        python aio_analyzer.py \
            --c1-file "$BASELINE_FILE" \
            --c2-file "$A_JUDGES_B_FILE" \
            --output "analysis/${INPUT_STEM}_${MODEL_NAME}_speaker_a_judges_speaker_b_cross_analysis.csv"
    else
        echo "No baseline file provided or file not found."
        echo "Analyzing C2 only (no DDS calculation possible):"
        python aio_analyzer.py \
            --input "$A_JUDGES_B_FILE" \
            --output "analysis/${INPUT_STEM}_${MODEL_NAME}_speaker_a_judges_speaker_b_c2only_analysis.csv"
    fi
else
    echo "File not found: $A_JUDGES_B_FILE"
fi

echo ""
echo "======================================"
echo "Analysis complete!"
echo "======================================"

