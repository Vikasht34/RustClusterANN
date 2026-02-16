#!/bin/bash
# Test Direct I/O on existing Cohere index on EC2

set -e

DATA_DIR="/data"
INDEX_DIR="/nvme/indexes/cohere"
OUTPUT_DIR="/data/results/cohere"

echo "=== Testing Direct I/O on Cohere 1M (EC2) ==="
echo "Started at: $(date)"
echo ""

cd /RustClusterANN

# Pull latest code with Direct I/O fix
echo "Pulling latest code..."
git pull origin 2.x

# Build release binary
echo "Building release binary..."
cargo build --release --bin bench_cohere_ondemand

# Run search on existing index (no rebuild)
echo ""
echo "=== Running search benchmark with Direct I/O ==="
echo "Using existing index: $INDEX_DIR/cohere_1m_no_quant.idx"
echo ""

./target/release/bench_cohere_ondemand \
  --data-path "$DATA_DIR" \
  --index-path "$INDEX_DIR/cohere_1m_no_quant.idx" \
  2>&1 | tee "$OUTPUT_DIR/cohere_direct_io_test_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Completed at: $(date)"
echo "Results saved to: $OUTPUT_DIR"
