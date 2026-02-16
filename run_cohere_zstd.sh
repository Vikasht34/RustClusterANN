#!/bin/bash

# Cohere 1M benchmark with zstd compression on EC2
# Measures latency breakdown and data transfer per query

set -e

DATA_PATH="/home/ubuntu/data"
INDEX_PATH="/home/ubuntu/cohere_zstd.idx"
BINARY="./target/release/bench_cohere_1bit"

echo "=== Cohere 1M with zstd + delta compression ==="
echo "Data path: $DATA_PATH"
echo "Index path: $INDEX_PATH"
echo ""

# Build if needed
if [ ! -f "$BINARY" ]; then
    echo "Building benchmark..."
    cargo build --release --bin bench_cohere_1bit
fi

# Run benchmark
echo "Running benchmark..."
$BINARY \
    --data-path "$DATA_PATH" \
    --index-path "$INDEX_PATH" \
    --zstd \
    --delta

echo ""
echo "=== Benchmark complete ==="
