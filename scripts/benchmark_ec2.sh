#!/bin/bash
# Benchmark script for SPANN on EC2
# Datasets: GIST, GloVe, Cohere, SIFT
# Usage: ./benchmark_ec2.sh [dataset_name]

set -e

# Configuration
DATA_DIR="${DATA_DIR:-/data}"
BASE_INDEX_DIR="${BASE_INDEX_DIR:-/nvme/indexes}"
BASE_RESULTS_DIR="${BASE_RESULTS_DIR:-/data/results}"
BINARY="./target/release/bench_cohere_ondemand"

# Build if needed
if [ ! -f "$BINARY" ]; then
    echo "Building benchmark binary..."
    cargo build --release --bin bench_cohere_ondemand
fi

# Dataset configurations (using .bin format)
declare -A DATASETS
DATASETS[gist]="gist 960 l2"
DATASETS[glove]="glove 100 cosine"
DATASETS[cohere]="cohere 768 ip"
DATASETS[sift]="sift 128 l2"
DATASETS[mpnet]="mpnet-msmarco 768 l2"
DATASETS[snowflake]="snowflake-msmarco 768 l2"
DATASETS[tasb]="tasb-msmarco 768 ip"

# SPANN parameters
NUM_HEADS=64
MAX_CHECK=8192

run_benchmark() {
    local name=$1
    local config=${DATASETS[$name]}
    read -r dataset_name dim metric <<< "$config"
    
    local data_path="$DATA_DIR/$dataset_name"
    local index_dir="$BASE_INDEX_DIR/$dataset_name"
    local results_dir="$BASE_RESULTS_DIR/$dataset_name"
    local index_path="$index_dir/${name}_spann.idx"
    local log_file="$results_dir/${name}_benchmark.log"
    
    # Create directories
    mkdir -p "$index_dir" "$results_dir"
    
    echo "========================================="
    echo "Benchmarking: $name"
    echo "Dataset: $dataset_name"
    echo "Dimension: $dim"
    echo "Metric: $metric"
    echo "========================================="
    echo "Data path: $data_path"
    echo "Index dir: $index_dir"
    echo "Results dir: $results_dir"
    echo ""
    
    # Check if data exists (look for .bin files)
    if [ ! -f "$data_path/base.bin" ]; then
        echo "ERROR: Data not found at $data_path/base.bin"
        echo "Please ensure base.bin, query.bin, and groundtruth.bin exist"
        return 1
    fi
    
    # Run benchmark
    echo "Starting benchmark at $(date)"
    echo "Index path: $index_path"
    echo "Log file: $log_file"
    echo ""
    
    $BINARY \
        --data-path "$data_path" \
        --index-path "$index_path" \
        --zstd \
        --delta \
        --num-heads $NUM_HEADS \
        --max-check $MAX_CHECK \
        --metric $metric \
        2>&1 | tee "$log_file"
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "✓ Benchmark completed successfully"
        echo "Results saved to: $log_file"
        
        # Extract key metrics
        echo ""
        echo "=== Summary ==="
        grep -E "Built BK-Tree|Building TP-Trees|Building RNG|Recall@|QPS|Build time" "$log_file" | tail -20
    else
        echo ""
        echo "✗ Benchmark failed with exit code $exit_code"
        return $exit_code
    fi
    
    echo ""
}

# Main
if [ $# -eq 0 ]; then
    echo "Usage: $0 <dataset>"
    echo ""
    echo "Available datasets:"
    for dataset in "${!DATASETS[@]}"; do
        echo "  - $dataset"
    done
    echo ""
    echo "Configuration:"
    echo "  DATA_DIR=$DATA_DIR"
    echo "  BASE_INDEX_DIR=$BASE_INDEX_DIR"
    echo "  BASE_RESULTS_DIR=$BASE_RESULTS_DIR"
    echo ""
    echo "Examples:"
    echo "  $0 gist"
    echo "  $0 glove"
    echo "  $0 cohere"
    echo "  $0 sift"
    echo ""
    echo "Custom paths:"
    echo "  BASE_INDEX_DIR=/nvme/indexes BASE_RESULTS_DIR=/data/results $0 cohere"
    echo ""
    echo "Run all:"
    echo "  for ds in gist glove cohere sift; do $0 \$ds; done"
    exit 1
fi

DATASET=$1

if [ -z "${DATASETS[$DATASET]}" ]; then
    echo "ERROR: Unknown dataset '$DATASET'"
    echo "Available: ${!DATASETS[@]}"
    exit 1
fi

run_benchmark "$DATASET"
