#!/usr/bin/env python3
import h5py
import struct
import numpy as np
import os

hdf5_path = '/Users/viktari/Downloads/glove-200-angular.hdf5'
output_dir = '/Users/viktari/pysptag/data/glove/'

os.makedirs(output_dir, exist_ok=True)

with h5py.File(hdf5_path, 'r') as f:
    print(f"Keys: {list(f.keys())}")
    
    train = np.array(f['train'])
    test = np.array(f['test'])
    neighbors = np.array(f['neighbors'])
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Neighbors shape: {neighbors.shape}")
    
    # Check if vectors are normalized
    norms = np.linalg.norm(train[:100], axis=1)
    print(f"\nFirst 10 vector norms: {norms[:10]}")
    print(f"Mean norm: {norms.mean():.6f}")
    print(f"Std norm: {norms.std():.6f}")
    print(f"Are vectors normalized? {np.allclose(norms, 1.0, atol=0.01)}")
    
    # Write base vectors
    n, d = train.shape
    with open(os.path.join(output_dir, 'base.bin'), 'wb') as out:
        out.write(struct.pack('II', n, d))
        train.astype('float32').tofile(out)
    print(f"\nWrote base.bin: {n} vectors, {d} dims")
    
    # Write query vectors
    n, d = test.shape
    with open(os.path.join(output_dir, 'query.bin'), 'wb') as out:
        out.write(struct.pack('II', n, d))
        test.astype('float32').tofile(out)
    print(f"Wrote query.bin: {n} vectors, {d} dims")
    
    # Write ground truth
    n, k = neighbors.shape
    with open(os.path.join(output_dir, 'groundtruth.bin'), 'wb') as out:
        out.write(struct.pack('II', n, k))
        neighbors.astype('int32').tofile(out)
    print(f"Wrote groundtruth.bin: {n} queries, top-{k}")

print(f"\nConverted to {output_dir}")
