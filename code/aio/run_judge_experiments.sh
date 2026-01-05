#!/bin/bash
# Run judge experiments for different models
# Usage: ./run_judge_experiments.sh <model> <input_file>
# Example: ./run_judge_experiments.sh qwen/qwen-2.5-7b-instruct finalAIOdata_experiment_ready.jsonl

MODEL=${1:-"qwen/qwen-2.5-7b-instruct"}
INPUT=${2:-"finalAIOdata_experiment_ready.jsonl"}

echo "======================================"
echo "Running experiments with model: $MODEL"
echo "Input file: $INPUT"
echo "======================================"

# Extract model name for filenames
MODEL_NAME=$(echo $MODEL | sed 's/.*\///')

echo ""
echo ">>> Experiment 1: Speaker A judges Speaker B (C2 only)"
echo "    This tests attribution to Speaker A as judger"
python aio_run_experiment_speaker_c.py \
    --input "$INPUT" \
    --model "$MODEL" \
    --judger "Speaker A" \
    --judgee "Speaker B" \
    --c2-only

echo ""
echo ">>> Experiment 2: Speaker B judges Speaker A (C1 + C2)"
echo "    Ground truth flips because Speaker A is being judged"
python aio_run_experiment_speaker_c.py \
    --input "$INPUT" \
    --model "$MODEL" \
    --judger "Speaker B" \
    --judgee "Speaker A"

echo ""
echo "======================================"
echo "Experiments complete!"
echo "======================================"
echo "Output files:"
echo "  - results/$(basename $INPUT .jsonl)_${MODEL_NAME}_speaker_a_judges_speaker_b_c2only_results.jsonl"
echo "  - results/$(basename $INPUT .jsonl)_${MODEL_NAME}_speaker_b_judges_speaker_a_results.jsonl"

