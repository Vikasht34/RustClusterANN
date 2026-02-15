#!/usr/bin/env python3
import h5py
import struct
import sys

def write_binary(filename, data):
    n, d = data.shape
    with open(filename, 'wb') as f:
        f.write(struct.pack('I', n))
        f.write(struct.pack('I', d))
        data.astype('float32').tofile(f)
    print(f"Wrote {filename}: {n} vectors, {d} dimensions")

def write_groundtruth(filename, data):
    n, k = data.shape
    with open(filename, 'wb') as f:
        f.write(struct.pack('I', n))
        f.write(struct.pack('I', k))
        data.astype('int32').tofile(f)
    print(f"Wrote {filename}: {n} queries, top-{k}")

hdf5_file = sys.argv[1] if len(sys.argv) > 1 else '/data/documents-1m.hdf5'
output_dir = sys.argv[2] if len(sys.argv) > 2 else '/data'

print(f"Reading {hdf5_file}...")
with h5py.File(hdf5_file, 'r') as f:
    write_binary(f'{output_dir}/base.bin', f['train'][:])
    write_binary(f'{output_dir}/query.bin', f['test'][:])
    write_groundtruth(f'{output_dir}/groundtruth.bin', f['neighbors'][:])

print("Done!")
