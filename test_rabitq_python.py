#!/usr/bin/env python3
"""
Test RaBitQ quantization to understand correct behavior
"""
import numpy as np
import struct

def read_fvecs(filename):
    """Read .fvecs file"""
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack('f' * dim, f.read(4 * dim))
            vectors.append(vec)
    return np.array(vectors, dtype='float32')

# Load a few SIFT vectors
base = read_fvecs('/Users/viktari/pysptag/data/sift/sift_base.fvecs')[:100]
query = read_fvecs('/Users/viktari/pysptag/data/sift/sift_query.fvecs')[:10]

print(f"Base: {base.shape}")
print(f"Query: {query.shape}")

# Simple test: compute exact L2 distances
print("\n=== Exact L2 Distances ===")
for i in range(3):
    for j in range(3):
        dist = np.sum((query[i] - base[j]) ** 2)
        print(f"Query {i} -> Base {j}: {dist:.6f}")

# Test with centroid
centroid = np.mean(base, axis=0)
print(f"\nCentroid norm: {np.linalg.norm(centroid):.6f}")

# Compute residuals
residuals = base - centroid
print(f"Residual norms (first 5): {[np.linalg.norm(r) for r in residuals[:5]]}")

# Normalize residuals
residual_norms = np.linalg.norm(residuals, axis=1, keepdims=True)
normalized_residuals = residuals / (residual_norms + 1e-10)
print(f"Normalized residual norms (first 5): {[np.linalg.norm(r) for r in normalized_residuals[:5]]}")

# Simple 1-bit quantization
binary_codes = (residuals >= 0).astype(np.int8)
print(f"\nBinary codes shape: {binary_codes.shape}")
print(f"Binary codes[0][:10]: {binary_codes[0][:10]}")

# Test query normalization
query_residual = query[0] - centroid
query_norm = np.linalg.norm(query_residual)
query_normalized = query_residual / query_norm
print(f"\nQuery residual norm: {query_norm:.6f}")
print(f"Query normalized norm: {np.linalg.norm(query_normalized):.6f}")

# Compute IP with binary codes
ip_with_normalized = np.dot(query_normalized, (binary_codes[0] * 2 - 1))
ip_with_unnormalized = np.dot(query[0], (binary_codes[0] * 2 - 1))
print(f"\nIP (normalized query): {ip_with_normalized:.6f}")
print(f"IP (unnormalized query): {ip_with_unnormalized:.6f}")
