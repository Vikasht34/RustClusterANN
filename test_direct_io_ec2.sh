#!/bin/bash
# Rebuild Cohere index with proper Direct I/O alignment

set -e

DATA_DIR="/data"
INDEX_DIR="/nvme/indexes/cohere"
OUTPUT_DIR="/data/results/cohere"

echo "=== Rebuilding Cohere 1M with Direct I/O alignment ==="
echo "Started at: $(date)"
echo ""

cd /RustClusterANN

# Pull latest code
echo "Pulling latest code..."
git pull origin 2.x

# Build release binary
echo "Building release binary..."
cargo build --release --bin bench_cohere_ondemand

# Delete old index (not properly aligned)
echo ""
echo "Removing old index (not aligned for Direct I/O)..."
rm -f "$INDEX_DIR/cohere_1m_no_quant.idx"*

# Rebuild index with proper alignment
echo ""
echo "=== Building new index with Direct I/O alignment ==="
./target/release/bench_cohere_ondemand \
  --data-path "$DATA_DIR" \
  --index-path "$INDEX_DIR/cohere_1m_no_quant.idx" \
  --zstd \
  --delta \
  2>&1 | tee "$OUTPUT_DIR/cohere_rebuild_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Completed at: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Index files:"
ls -lh "$INDEX_DIR/cohere_1m_no_quant.idx"*
