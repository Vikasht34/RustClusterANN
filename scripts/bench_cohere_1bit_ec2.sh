#!/bin/bash
set -e

DATA_DIR="/data"
OUTPUT_DIR="/data/results/cohere_1bit"
INDEX_DIR="/nvme/indexes/cohere_1bit"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$INDEX_DIR"

echo "=== Cohere 1M 1-bit Quantization Benchmark on EC2 ==="
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Index directory: $INDEX_DIR"
echo "Started at: $(date)"

cd /RustClusterANN

# Build release binary
cargo build --release --bin bench_cohere_1bit

# Run benchmark with 1-bit quantization
./target/release/bench_cohere_1bit \
  --data-path "$DATA_DIR" \
  --index-path "$INDEX_DIR/cohere_1m_1bit.idx" \
  --enable-compression \
  2>&1 | tee "$OUTPUT_DIR/cohere_1bit_benchmark_$(date +%Y%m%d_%H%M%S).log"

echo "Completed at: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "Index saved to: $INDEX_DIR"
