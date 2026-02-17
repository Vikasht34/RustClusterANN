# SPANN Benchmark on EC2

## Quick Start

### 1. Setup EC2 Instance
```bash
# Recommended: c7i.8xlarge or similar (32 vCPUs, 64GB RAM)
# OS: Amazon Linux 2023 or Ubuntu 22.04

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install dependencies
sudo yum install -y git gcc  # Amazon Linux
# OR
sudo apt install -y git build-essential  # Ubuntu
```

### 2. Clone and Build
```bash
git clone <your-repo-url> RustClusterANN
cd RustClusterANN
cargo build --release --bin bench_cohere_ondemand
```

### 3. Download Datasets
```bash
# Create data directory
mkdir -p /data

# Download datasets (example for GIST)
cd /data
wget http://corpus-texmex.irisa.fr/gist.tar.gz
tar -xzf gist.tar.gz
mv gist gist-data
# Convert to required format if needed
```

### 4. Run Benchmarks
```bash
cd RustClusterANN

# Single dataset
./benchmark_ec2.sh gist

# All datasets
for ds in gist glove cohere sift; do
    ./benchmark_ec2.sh $ds
done
```

## Dataset Locations

Expected directory structure:
```
/data/
├── gist/
│   ├── gist_base.fvecs
│   ├── gist_query.fvecs
│   └── gist_groundtruth.ivecs
├── glove-100/
│   ├── glove_base.fvecs
│   ├── glove_query.fvecs
│   └── glove_groundtruth.ivecs
├── cohere/
│   ├── cohere_base.fvecs
│   ├── cohere_query.fvecs
│   └── cohere_groundtruth.ivecs
└── sift/
    ├── sift_base.fvecs
    ├── sift_query.fvecs
    └── sift_groundtruth.ivecs
```

## Configuration

Edit `benchmark_ec2.sh` to customize:
- `DATA_DIR`: Where datasets are stored (default: `/data`)
- `INDEX_DIR`: Where to save indices (default: `/tmp/indices`)
- `RESULTS_DIR`: Where to save logs (default: `/tmp/results`)
- `NUM_HEADS`: Number of cluster heads (default: 64)
- `MAX_CHECK`: Max candidates to check (default: 8192)

## Parameters

### SPANN Index Parameters
- **BK-Tree**: K=32, leaf_size=8, lambdaFactor=100
- **TP-Trees**: 32 trees, leaf_size=2000
- **RNG**: max_degree=32, 2 refinement iterations

### Compression
- `--zstd`: Enable zstd compression for vectors
- `--delta`: Enable delta encoding for posting lists

## Expected Performance

### GIST (1M vectors, 960D, L2)
- Build time: ~5-10 minutes
- Memory: ~7GB
- Recall@10: >0.90 at 1000+ QPS

### SIFT (1M vectors, 128D, L2)
- Build time: ~2-5 minutes
- Memory: ~2GB
- Recall@10: >0.95 at 5000+ QPS

## Troubleshooting

### Out of Memory
- Reduce `NUM_HEADS` to 32 or 16
- Use smaller `MAX_CHECK` (4096 or 2048)
- Use instance with more RAM

### Slow Build
- Check CPU usage (should be near 100%)
- Ensure using `--release` build
- Check disk I/O if using slow storage

### Poor Recall
- Increase `MAX_CHECK` to 16384 or 32768
- Increase `NUM_HEADS` to 128 or 256
- Check metric matches dataset (L2 vs cosine)

## Results

Results are saved to `/tmp/results/` with format:
- `{dataset}_benchmark.log`: Full benchmark output
- Key metrics: Build time, index size, recall@K, QPS

Extract summary:
```bash
grep -E "Built BK-Tree|Recall@|QPS" /tmp/results/*.log
```
