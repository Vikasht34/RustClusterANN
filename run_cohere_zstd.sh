#!/bin/bash

# Cohere 1M benchmark with zstd compression on EC2
# Measures latency breakdown and data transfer per query

set -e

DATA_DIR="/data"
OUTPUT_DIR="/data/results/cohere"
INDEX_DIR="/nvme/indexes/cohere"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$INDEX_DIR"

echo "=== Cohere 1M with zstd + delta compression ==="
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Index directory: $INDEX_DIR"
echo "Started at: $(date)"

cd /RustClusterANN

# Build release binary
cargo build --release --bin bench_cohere_1bit

# Run benchmark with zstd + delta (no quantization)
./target/release/bench_cohere_1bit \
  --data-path "$DATA_DIR" \
  --index-path "$INDEX_DIR/cohere_1m_no_quant.idx" \
  --zstd \
  --delta \
  2>&1 | tee "$OUTPUT_DIR/cohere_no_quant_benchmark_$(date +%Y%m%d_%H%M%S).log"

echo "Completed at: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "Index saved to: $INDEX_DIR"
