#!/usr/bin/env python3
import h5py
import struct
import numpy as np
import os

hdf5_path = '/Users/viktari/Downloads/gist-960-euclidean.hdf5'
output_dir = '/Users/viktari/pysptag/data/gist/'

os.makedirs(output_dir, exist_ok=True)

with h5py.File(hdf5_path, 'r') as f:
    train = np.array(f['train'])
    test = np.array(f['test'])
    neighbors = np.array(f['neighbors'])
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Neighbors shape: {neighbors.shape}")
    
    # Write base vectors (train)
    with open(os.path.join(output_dir, 'gist_base.fvecs'), 'wb') as out:
        for vec in train:
            out.write(struct.pack('i', len(vec)))
            out.write(struct.pack('f' * len(vec), *vec))
    
    # Write query vectors (test)
    with open(os.path.join(output_dir, 'gist_query.fvecs'), 'wb') as out:
        for vec in test:
            out.write(struct.pack('i', len(vec)))
            out.write(struct.pack('f' * len(vec), *vec))
    
    # Write ground truth
    with open(os.path.join(output_dir, 'gist_groundtruth.ivecs'), 'wb') as out:
        for neighbor_list in neighbors:
            out.write(struct.pack('i', len(neighbor_list)))
            out.write(struct.pack('i' * len(neighbor_list), *neighbor_list))

print(f"\nConverted to {output_dir}")
