#!/bin/bash
set -e

SIFT_DIR="/data/sift"
OUTPUT_DIR="/data/results/sift_ondemand"
INDEX_DIR="/nvme/indexes/sift_ondemand"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$INDEX_DIR"

echo "=== SIFT 1M On-Demand Loading Benchmark on EC2 ==="
echo "Data directory: $SIFT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Index directory: $INDEX_DIR"
echo "Started at: $(date)"

cd /RustClusterANN

# Build release binary
cargo build --release --bin bench_sift_ondemand

# Run benchmark with on-demand loading
./target/release/bench_sift_ondemand \
  --sift-dir "$SIFT_DIR" \
  --index-path "$INDEX_DIR/sift_1m_ondemand.idx" \
  --enable-compression \
  2>&1 | tee "$OUTPUT_DIR/sift_ondemand_benchmark_$(date +%Y%m%d_%H%M%S).log"

echo "Completed at: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "Index saved to: $INDEX_DIR"
