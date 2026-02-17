#!/bin/bash

DATA_DIR="/data/gist"
INDEX_DIR="/nvme/indexes/gist"
OUTPUT_DIR="/data/results/gist"

echo "=== Building GIST 1M with Direct I/O alignment ==="
echo "Started at: $(date)"
echo ""

cd /RustClusterANN

# Pull latest code
echo "Pulling latest code..."
git pull origin 2.x

# Build release binary
echo "Building release binary..."
cargo build --release --bin bench_cohere_ondemand

# Create directories
echo "Creating directories..."
sudo mkdir -p "$INDEX_DIR"
sudo mkdir -p "$OUTPUT_DIR"
sudo chown -R $USER:$USER "$INDEX_DIR"
sudo chown -R $USER:$USER "$OUTPUT_DIR"

# Delete old index if exists
echo ""
echo "Removing old index..."
rm -f "$INDEX_DIR/gist_1m.idx"*

# Build index with proper alignment
echo ""
echo "=== Building GIST index with Direct I/O alignment ==="
echo "Note: Using 64 heads (reduced from 128) for 960D vectors to reduce memory"
./target/release/bench_cohere_ondemand \
  --data-path "$DATA_DIR" \
  --index-path "$INDEX_DIR/gist_1m.idx" \
  --zstd \
  --delta \
  --num-heads 64 \
  --max-check 8192 \
  --metric l2 \
  2>&1 | tee "$OUTPUT_DIR/gist_build_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Completed at: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Index files:"
ls -lh "$INDEX_DIR/gist_1m.idx"*
