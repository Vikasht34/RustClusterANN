#!/usr/bin/env python3
import h5py
import numpy as np
import struct

# Read HDF5
print("Loading Cohere 1M from HDF5...")
with h5py.File('/Users/viktari/pysptag/data/cohere/documents-1m.hdf5', 'r') as f:
    train = np.array(f['train'])
    test = np.array(f['test'])
    neighbors = np.array(f['neighbors'])

print(f"Train: {train.shape}")
print(f"Test: {test.shape}")
print(f"Neighbors: {neighbors.shape}")

# Write to binary format
print("\nWriting to binary format...")

# Write train vectors
with open('/Users/viktari/rustsptag/data/cohere_base.bin', 'wb') as f:
    n, d = train.shape
    f.write(struct.pack('II', n, d))
    train.astype(np.float32).tofile(f)
print(f"  Wrote {n} base vectors ({d}D)")

# Write test vectors
with open('/Users/viktari/rustsptag/data/cohere_query.bin', 'wb') as f:
    n, d = test.shape
    f.write(struct.pack('II', n, d))
    test.astype(np.float32).tofile(f)
print(f"  Wrote {n} query vectors ({d}D)")

# Write ground truth
with open('/Users/viktari/rustsptag/data/cohere_groundtruth.bin', 'wb') as f:
    n, k = neighbors.shape
    f.write(struct.pack('II', n, k))
    neighbors.astype(np.int32).tofile(f)
print(f"  Wrote {n} ground truth vectors (top-{k})")

print("\nDone!")
