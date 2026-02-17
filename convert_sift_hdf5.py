#!/usr/bin/env python3
import h5py
import struct
import numpy as np

# Convert SIFT HDF5 to fvecs format
hdf5_path = '/Users/viktari/Downloads/sift-128-euclidean.hdf5'
output_dir = '/Users/viktari/pysptag/data/sift'

with h5py.File(hdf5_path, 'r') as f:
    print("Keys in HDF5:", list(f.keys()))
    
    # Convert train/base vectors
    if 'train' in f:
        train = np.array(f['train'])
        print(f"Train: {train.shape} {train.dtype}")
        with open(f'{output_dir}/sift_base.fvecs', 'wb') as out:
            for vec in train:
                out.write(struct.pack('i', len(vec)))
                out.write(vec.astype('float32').tobytes())
        print(f"Wrote {len(train)} vectors to sift_base.fvecs")
    
    # Convert test/query vectors
    if 'test' in f:
        test = np.array(f['test'])
        print(f"Test: {test.shape} {test.dtype}")
        with open(f'{output_dir}/sift_query.fvecs', 'wb') as out:
            for vec in test:
                out.write(struct.pack('i', len(vec)))
                out.write(vec.astype('float32').tobytes())
        print(f"Wrote {len(test)} vectors to sift_query.fvecs")
    
    # Convert ground truth
    if 'neighbors' in f:
        neighbors = np.array(f['neighbors'])
        print(f"Neighbors: {neighbors.shape} {neighbors.dtype}")
        with open(f'{output_dir}/sift_groundtruth.ivecs', 'wb') as out:
            for vec in neighbors:
                out.write(struct.pack('i', len(vec)))
                out.write(vec.astype('int32').tobytes())
        print(f"Wrote {len(neighbors)} ground truth vectors to sift_groundtruth.ivecs")

print("\nConversion complete!")
