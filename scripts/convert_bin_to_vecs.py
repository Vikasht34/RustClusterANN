#!/usr/bin/env python3
"""Convert .bin format to .fvecs/.ivecs format"""
import struct
import sys
import numpy as np

def bin_to_fvecs(bin_file, fvecs_file, dtype=np.float32):
    """Convert binary file to fvecs format"""
    # Read binary file
    data = np.fromfile(bin_file, dtype=dtype)
    
    # Assume first value is dimension
    dim = int(data[0])
    data = data[1:]
    
    # Reshape to (n_vectors, dim)
    n_vectors = len(data) // dim
    vectors = data[:n_vectors * dim].reshape(n_vectors, dim)
    
    print(f"Converting {bin_file} -> {fvecs_file}")
    print(f"  Vectors: {n_vectors}, Dimension: {dim}")
    
    # Write fvecs format
    with open(fvecs_file, 'wb') as f:
        for vec in vectors:
            f.write(struct.pack('i', dim))
            f.write(vec.astype(np.float32).tobytes())
    
    print(f"  ✓ Done")

def bin_to_ivecs(bin_file, ivecs_file):
    """Convert binary groundtruth to ivecs format"""
    # Read binary file
    data = np.fromfile(bin_file, dtype=np.int32)
    
    # Assume first value is k (number of neighbors)
    k = int(data[0])
    data = data[1:]
    
    # Reshape to (n_queries, k)
    n_queries = len(data) // k
    neighbors = data[:n_queries * k].reshape(n_queries, k)
    
    print(f"Converting {bin_file} -> {ivecs_file}")
    print(f"  Queries: {n_queries}, K: {k}")
    
    # Write ivecs format
    with open(ivecs_file, 'wb') as f:
        for row in neighbors:
            f.write(struct.pack('i', k))
            f.write(row.astype(np.int32).tobytes())
    
    print(f"  ✓ Done")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage:")
        print("  Convert vectors: python3 convert_bin_to_vecs.py base.bin base.fvecs float")
        print("  Convert queries: python3 convert_bin_to_vecs.py query.bin query.fvecs float")
        print("  Convert groundtruth: python3 convert_bin_to_vecs.py groundtruth.bin groundtruth.ivecs int")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    data_type = sys.argv[3]
    
    if data_type == "int":
        bin_to_ivecs(input_file, output_file)
    else:
        bin_to_fvecs(input_file, output_file)
