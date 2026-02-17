#!/bin/bash
# Run GloVe benchmark on EC2
set -e

DATA_PATH=${1:-"/data/glove"}
INDEX_PATH=${2:-"/tmp/glove.idx"}
OUTPUT_LOG=${3:-"/tmp/glove_benchmark.log"}

echo "=== GloVe Benchmark ===" | tee "$OUTPUT_LOG"
echo "Data: $DATA_PATH" | tee -a "$OUTPUT_LOG"
echo "Index: $INDEX_PATH" | tee -a "$OUTPUT_LOG"
echo "Started: $(date)" | tee -a "$OUTPUT_LOG"
echo "" | tee -a "$OUTPUT_LOG"

./target/release/bench_cohere_ondemand \
    --data-path "$DATA_PATH" \
    --index-path "$INDEX_PATH" \
    --zstd \
    --delta \
    --num-heads 128 \
    --max-check 8192 \
    --metric cosine \
    2>&1 | tee -a "$OUTPUT_LOG"

echo "" | tee -a "$OUTPUT_LOG"
echo "Completed: $(date)" | tee -a "$OUTPUT_LOG"
