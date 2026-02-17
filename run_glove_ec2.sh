#!/bin/bash
set -e

echo "=== GloVe Benchmark on EC2 ==="
echo "Started at: $(date)"

# Configuration
DATA_DIR="/data"
INDEX_PATH="/tmp/glove.idx"
DATASET="glove-100-angular"

# Download and convert GloVe dataset
echo ""
echo "=== Downloading GloVe Dataset ==="
cd /tmp
if [ ! -f "${DATA_DIR}/glove_base.bin" ]; then
    echo "Downloading GloVe-100 dataset..."
    wget -q http://ann-benchmarks.com/glove-100-angular.hdf5 -O glove.hdf5
    
    echo "Converting to binary format..."
    python3 << 'EOF'
import h5py
import struct
import numpy as np

with h5py.File('glove.hdf5', 'r') as f:
    train = np.array(f['train'])
    test = np.array(f['test'])
    neighbors = np.array(f['neighbors'])
    distances = np.array(f['distances'])

# Write base vectors
with open('glove_base.bin', 'wb') as f:
    n, d = train.shape
    f.write(struct.pack('II', n, d))
    for vec in train:
        f.write(vec.astype('float32').tobytes())

# Write query vectors
with open('glove_query.bin', 'wb') as f:
    n, d = test.shape
    f.write(struct.pack('II', n, d))
    for vec in test:
        f.write(vec.astype('float32').tobytes())

# Write ground truth (neighbors are 0-indexed already for GloVe)
with open('glove_groundtruth.bin', 'wb') as f:
    n, k = neighbors.shape
    f.write(struct.pack('II', n, k))
    for row in neighbors:
        f.write(row.astype('int32').tobytes())

print(f"Converted: {n} base vectors, {test.shape[0]} queries, {d}D")
EOF

    # Move to data directory
    mkdir -p ${DATA_DIR}
    mv glove_base.bin glove_query.bin glove_groundtruth.bin ${DATA_DIR}/
    rm glove.hdf5
    
    echo "Dataset ready!"
else
    echo "Dataset already exists, skipping download"
fi

# Build benchmark binary
echo ""
echo "=== Building Benchmark ==="
cd /workspace
cargo build --release --bin bench_glove

# Run benchmark
echo ""
echo "=== Running Benchmark ==="
echo "Dataset: GloVe-100 (1.18M vectors, 100D, Angular/Cosine)"
echo "Index: ${INDEX_PATH}"
echo ""

./target/release/bench_glove \
    --data-path ${DATA_DIR} \
    --index-path ${INDEX_PATH} \
    --num-heads 128 \
    --max-check 8192

echo ""
echo "=== Benchmark Complete ==="
echo "Finished at: $(date)"
