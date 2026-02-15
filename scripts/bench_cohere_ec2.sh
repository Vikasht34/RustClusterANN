#!/bin/bash
set -e

COHERE_FILE="/data/documents-1m.hdf5"
OUTPUT_DIR="/data/results/cohere"
INDEX_DIR="/nvme/indexes/cohere"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$INDEX_DIR"

echo "=== Cohere 1M Benchmark on EC2 ==="
echo "Data file: $COHERE_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Index directory: $INDEX_DIR"
echo "Started at: $(date)"

cd /home/ec2-user/rustsptag

# Build release binary
cargo build --release --bin bench_cohere1m

# Run benchmark with storage
./target/release/bench_cohere1m \
  --data-path "$COHERE_FILE" \
  --index-path "$INDEX_DIR/cohere_1m.idx" \
  --enable-compression \
  2>&1 | tee "$OUTPUT_DIR/cohere_benchmark_$(date +%Y%m%d_%H%M%S).log"

echo "Completed at: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "Index saved to: $INDEX_DIR"
