#!/usr/bin/env python3
import struct
import numpy as np

def fvecs_to_bin(fvecs_path, bin_path):
    """Convert .fvecs to .bin format"""
    vectors = []
    with open(fvecs_path, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack('f' * dim, f.read(4 * dim))
            vectors.append(vec)
    
    data = np.array(vectors, dtype='float32')
    n, d = data.shape
    
    with open(bin_path, 'wb') as f:
        f.write(struct.pack('II', n, d))
        data.tofile(f)
    
    print(f"Converted {fvecs_path} -> {bin_path}: {n} vectors, {d} dims")

def ivecs_to_bin(ivecs_path, bin_path):
    """Convert .ivecs to groundtruth .bin format"""
    vectors = []
    with open(ivecs_path, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack('i' * dim, f.read(4 * dim))
            vectors.append(vec)
    
    data = np.array(vectors, dtype='int32')
    n, k = data.shape
    
    with open(bin_path, 'wb') as f:
        f.write(struct.pack('II', n, k))
        data.tofile(f)
    
    print(f"Converted {ivecs_path} -> {bin_path}: {n} queries, top-{k}")

# Convert GIST
fvecs_to_bin('/Users/viktari/pysptag/data/gist/gist_base.fvecs', 
             '/Users/viktari/pysptag/data/gist/base.bin')
fvecs_to_bin('/Users/viktari/pysptag/data/gist/gist_query.fvecs', 
             '/Users/viktari/pysptag/data/gist/query.bin')
ivecs_to_bin('/Users/viktari/pysptag/data/gist/gist_groundtruth.ivecs', 
             '/Users/viktari/pysptag/data/gist/groundtruth.bin')
