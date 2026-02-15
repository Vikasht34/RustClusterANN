#!/bin/bash
set -e

SIFT_DIR="/data/sift"
OUTPUT_DIR="/nvme/results/sift"
INDEX_DIR="/nvme/indexes/sift"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$INDEX_DIR"

echo "=== SIFT 1M Benchmark on EC2 ==="
echo "Data directory: $SIFT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Index directory: $INDEX_DIR"
echo "Started at: $(date)"

cd /home/ec2-user/rustsptag

# Build release binary
cargo build --release --bin bench_sift_storage

# Run benchmark with storage
./target/release/bench_sift_storage \
  --sift-dir "$SIFT_DIR" \
  --index-path "$INDEX_DIR/sift_1m.idx" \
  --enable-compression \
  2>&1 | tee "$OUTPUT_DIR/sift_benchmark_$(date +%Y%m%d_%H%M%S).log"

echo "Completed at: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "Index saved to: $INDEX_DIR"
