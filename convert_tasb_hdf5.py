#!/usr/bin/env python3
import h5py
import numpy as np
import struct
import sys

def write_binary(filename, data):
    """Write vectors in binary format: [n, d, vector_data]"""
    n, d = data.shape
    with open(filename, 'wb') as f:
        f.write(struct.pack('II', n, d))
        data.astype('float32').tofile(f)
    print(f"Wrote {filename}: {n} vectors, {d} dimensions")

def main():
    hdf5_file = sys.argv[1] if len(sys.argv) > 1 else '/Users/viktari/Downloads/marco_tasb.hdf5'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '/Users/viktari/rustsptag/data/tasb'
    
    print(f"Reading {hdf5_file}...")
    with h5py.File(hdf5_file, 'r') as f:
        print(f"Keys: {list(f.keys())}")
        
        # Read datasets
        train = np.array(f['train'])
        test = np.array(f['test'])
        neighbors = np.array(f['neighbors'])
        
        print(f"Train shape: {train.shape}")
        print(f"Test shape: {test.shape}")
        print(f"Neighbors shape: {neighbors.shape}")
    
    # Create output directory
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Write binary files
    write_binary(f'{output_dir}/base.bin', train)
    write_binary(f'{output_dir}/query.bin', test)
    
    # Write ground truth (neighbors are already indices)
    n, k = neighbors.shape
    with open(f'{output_dir}/groundtruth.bin', 'wb') as f:
        f.write(struct.pack('II', n, k))
        neighbors.astype('int32').tofile(f)
    print(f"Wrote {output_dir}/groundtruth.bin: {n} queries, top-{k}")
    
    print(f"\nDone! Files written to {output_dir}/")

if __name__ == '__main__':
    main()
