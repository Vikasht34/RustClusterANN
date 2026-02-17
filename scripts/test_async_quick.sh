#!/bin/bash
# Quick test: build index and test search_async
set -e

INDEX_PATH="/tmp/cohere_quick.idx"
DATA_PATH="/Users/viktari/rustsptag/data"

echo "=== Quick On-Demand Search Test ==="
echo ""

# Build or use existing index
if [ -f "$INDEX_PATH" ]; then
    echo "Using existing index: $INDEX_PATH"
else
    echo "Building index (this will take ~15 minutes)..."
    ./target/release/bench_cohere_ondemand \
        --data-path "$DATA_PATH" \
        --index-path "$INDEX_PATH" \
        --num-heads 128 \
        --max-check 8192 \
        --metric inner-product
fi

echo ""
echo "Test complete! Check output above for recall results."
