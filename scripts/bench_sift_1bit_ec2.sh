#!/bin/bash
set -e

SIFT_DIR="/data/sift"
OUTPUT_DIR="/data/results/sift_1bit"
INDEX_DIR="/nvme/indexes/sift_1bit"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$INDEX_DIR"

echo "=== SIFT 1M 1-bit Quantization Benchmark on EC2 ==="
echo "Data directory: $SIFT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Index directory: $INDEX_DIR"
echo "Started at: $(date)"

cd /RustClusterANN

# Build release binary
cargo build --release --bin bench_sift_1bit

# Run benchmark with 1-bit quantization
./target/release/bench_sift_1bit \
  --sift-dir "$SIFT_DIR" \
  --index-path "$INDEX_DIR/sift_1m_1bit.idx" \
  --enable-compression \
  2>&1 | tee "$OUTPUT_DIR/sift_1bit_benchmark_$(date +%Y%m%d_%H%M%S).log"

echo "Completed at: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "Index saved to: $INDEX_DIR"
