#!/bin/bash
# Test all optimizations on SIFT 1M

SIFT_DIR="/Users/viktari/rustsptag/data/sift"

echo "=== Testing SIFT 1M with All Optimizations ==="
echo ""

# Test 1: Baseline (no optimizations)
echo "Test 1: Baseline (no optimizations)"
rm -f /tmp/sift_baseline.idx*
./target/release/bench_sift_ondemand \
  --sift-dir "$SIFT_DIR" \
  --index-path /tmp/sift_baseline.idx \
  2>&1 | grep -E "(File size|Recall|p50|QPS)"

echo ""
echo "---"
echo ""

# Test 2: Delta encoding only
echo "Test 2: Delta encoding (SIMD)"
rm -f /tmp/sift_delta.idx*
./target/release/bench_sift_ondemand \
  --sift-dir "$SIFT_DIR" \
  --index-path /tmp/sift_delta.idx \
  --delta \
  2>&1 | grep -E "(File size|Recall|p50|QPS)"

echo ""
echo "---"
echo ""

# Test 3: Zstd compression only
echo "Test 3: Zstd compression"
rm -f /tmp/sift_zstd.idx*
./target/release/bench_sift_ondemand \
  --sift-dir "$SIFT_DIR" \
  --index-path /tmp/sift_zstd.idx \
  --zstd \
  2>&1 | grep -E "(File size|Recall|p50|QPS)"

echo ""
echo "---"
echo ""

# Test 4: All optimizations
echo "Test 4: All optimizations (zstd + delta + rearrange)"
rm -f /tmp/sift_all.idx*
./target/release/bench_sift_ondemand \
  --sift-dir "$SIFT_DIR" \
  --index-path /tmp/sift_all.idx \
  --zstd \
  --delta \
  --rearrange \
  2>&1 | grep -E "(File size|Recall|p50|QPS)"

echo ""
echo "=== Summary ==="
ls -lh /tmp/sift_*.idx | awk '{print $9, $5}'
