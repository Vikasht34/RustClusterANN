#!/bin/bash

# Test all datasets on EC2 with on-demand mode, zstd, delta, and default parameters
# Usage: ./test_all_datasets_ec2.sh /path/to/hdfc/data /path/to/output

set -e

DATA_ROOT=${1:-"/data"}
OUTPUT_ROOT=${2:-"/tmp/results"}
BINARY="./target/release/bench_cohere_ondemand"

# Create output directory
mkdir -p "$OUTPUT_ROOT"

# Dataset configurations: name, data_path, index_path, k, metric
declare -a DATASETS=(
    "gist:${DATA_ROOT}/gist:${OUTPUT_ROOT}/gist.idx:100:l2"
    "glove:${DATA_ROOT}/glove:${OUTPUT_ROOT}/glove.idx:100:cosine"
    "mpnet-msmarco:${DATA_ROOT}/mpnet-msmarco:${OUTPUT_ROOT}/mpnet.idx:10:l2"
    "tasb-msmarco:${DATA_ROOT}/tasb-msmarco:${OUTPUT_ROOT}/tasb.idx:10:inner-product"
    "snowflake-msmarco:${DATA_ROOT}/snowflake-msmarco:${OUTPUT_ROOT}/snowflake.idx:10:l2"
)

# Function to run benchmark for a dataset
run_benchmark() {
    local config=$1
    IFS=':' read -r name data_path index_path k metric <<< "$config"
    
    local log_file="${OUTPUT_ROOT}/${name}_ondemand.log"
    
    echo "========================================" | tee -a "$log_file"
    echo "Testing: $name" | tee -a "$log_file"
    echo "Data: $data_path" | tee -a "$log_file"
    echo "Index: $index_path" | tee -a "$log_file"
    echo "K: $k" | tee -a "$log_file"
    echo "Metric: $metric" | tee -a "$log_file"
    echo "Started: $(date)" | tee -a "$log_file"
    echo "========================================" | tee -a "$log_file"
    
    # Run benchmark with default parameters (128 heads, 4096 max_check)
    $BINARY \
        --data-path "$data_path" \
        --index-path "$index_path" \
        --zstd \
        --delta \
        --metric "$metric" \
        2>&1 | tee -a "$log_file"
    
    echo "" | tee -a "$log_file"
    echo "Completed: $(date)" | tee -a "$log_file"
    echo "========================================" | tee -a "$log_file"
    echo "" | tee -a "$log_file"
}

# Export function for parallel execution
export -f run_benchmark
export BINARY OUTPUT_ROOT

echo "Starting parallel benchmarks for all datasets..."
echo "Data root: $DATA_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Timestamp: $(date)"
echo ""

# Run all benchmarks in parallel
printf '%s\n' "${DATASETS[@]}" | xargs -P 5 -I {} bash -c 'run_benchmark "$@"' _ {}

echo ""
echo "All benchmarks completed!"
echo "Results saved to: $OUTPUT_ROOT"
echo ""

# Generate comprehensive report
echo "Generating comprehensive report..."
./generate_report.sh "$OUTPUT_ROOT"

