#!/bin/bash
# Run benchmark experiments with amazon/nova-lite-v1 on all sampled_formatted datasets

# Configuration
MODEL="amazon/nova-lite-v1"
MODEL_SHORT="nova-lite-v1"
INPUT_DIR="../../dataset/benchmark_data/sampled_formatted"
OUTPUT_DIR="../../results/benchmark/${MODEL_SHORT}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# List of datasets
DATASETS=(
    "advisorqa_formatted"
    "amqa_formatted"
    "bbq_formatted"
    "gpqa_formatted"
    "halueval_qa_formatted"
    "harp_mcq_formatted"
    "plausibleqa_formatted"
    "socialiqa_formatted"
    "truthfulqa_formatted"
)

echo "=============================================="
echo "Running Benchmark Experiments with ${MODEL}"
echo "=============================================="
echo ""

# Run experiment for each dataset
for dataset in "${DATASETS[@]}"; do
    echo "Processing: ${dataset}"
    
    # Extract base name (remove _formatted suffix)
    base_name="${dataset%_formatted}"
    
    # Output file
    output_file="${OUTPUT_DIR}/${base_name}_${MODEL_SHORT}_Speaker1_vs_Speaker2_all.jsonl"
    
    # Skip if output already exists
    if [ -f "$output_file" ]; then
        echo "  → Skipping (output exists): ${output_file}"
        continue
    fi
    
    # Run experiment
    python bench_run_experiment.py \
        --input "${INPUT_DIR}/${dataset}.jsonl" \
        --output "$output_file" \
        --model "$MODEL"
    
    echo "  → Saved to: ${output_file}"
    echo ""
done

echo "=============================================="
echo "All experiments complete!"
echo "=============================================="
