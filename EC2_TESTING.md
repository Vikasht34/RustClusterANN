# EC2 Dataset Testing Guide

## Datasets to Test
1. **gist** - 1M vectors, 960D
2. **glove** - 1.2M vectors, 100D
3. **mpnet-msmarco** - 1M vectors, 768D
4. **tasb-msmarco** - 1M vectors, 768D
5. **snowflake-msmarco** - 1M vectors, 768D

## Prerequisites on EC2

### 1. Install Rust
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### 2. Clone Repository
```bash
git clone https://github.com/Vikasht34/RustClusterANN.git
cd RustClusterANN
git checkout 2.x
```

### 3. Build Binary
```bash
cargo build --release --bin bench_cohere_ondemand
```

### 4. Verify Data Location
Ensure all datasets are in HDFC at `/data` with structure:
```
/data/
├── gist/
│   ├── base.bin
│   ├── query.bin
│   └── groundtruth.bin
├── glove/
│   ├── base.bin
│   ├── query.bin
│   └── groundtruth.bin
├── mpnet-msmarco/
│   ├── base.bin
│   ├── query.bin
│   └── groundtruth.bin
├── tasb-msmarco/
│   ├── base.bin
│   ├── query.bin
│   └── groundtruth.bin
└── snowflake-msmarco/
    ├── base.bin
    ├── query.bin
    └── groundtruth.bin
```

## Running Tests

### Run All Datasets in Parallel
```bash
./test_all_datasets_ec2.sh /data /tmp/results
```

This will:
- Test all 5 datasets in parallel
- Use default parameters: 128 heads, 4096 max_check
- Enable zstd compression and delta encoding
- Save results to `/tmp/results/`

### Run Single Dataset
```bash
./target/release/bench_cohere_ondemand \
  --data-path /data/gist \
  --index-path /tmp/gist.idx \
  --zstd --delta
```

### Custom Parameters
```bash
# Test with higher max_check for better recall
./target/release/bench_cohere_ondemand \
  --data-path /data/gist \
  --index-path /tmp/gist.idx \
  --zstd --delta \
  --max-check 8192

# Test with different num_heads
./target/release/bench_cohere_ondemand \
  --data-path /data/gist \
  --index-path /tmp/gist.idx \
  --zstd --delta \
  --num-heads 64 \
  --max-check 4096
```

## Output Files

Results will be saved to:
```
/tmp/results/
├── gist_ondemand.log
├── glove_ondemand.log
├── mpnet_ondemand.log
├── tasb_ondemand.log
└── snowflake_ondemand.log
```

Each log contains:
- Build time and index size
- Search latency (p50, p90, p99)
- Recall@K
- QPS (queries per second)
- Memory usage

## Expected Performance (EC2 with O_DIRECT)

| Dataset | Recall Target | Expected p50 | Expected QPS |
|---------|---------------|--------------|--------------|
| gist | ~90% | 20-30ms | 30-50 |
| glove | ~90% | 15-25ms | 40-60 |
| mpnet | ~90% | 20-30ms | 30-50 |
| tasb | ~90% | 20-30ms | 30-50 |
| snowflake | ~90% | 20-30ms | 30-50 |

## Monitoring Progress

```bash
# Watch all logs in real-time
tail -f /tmp/results/*.log

# Check specific dataset
tail -f /tmp/results/gist_ondemand.log

# Check completion status
ls -lh /tmp/results/*.log
```

## Troubleshooting

### Out of Memory
If EC2 runs out of memory, run datasets sequentially:
```bash
# Edit script and change -P 5 to -P 1
sed -i 's/-P 5/-P 1/g' test_all_datasets_ec2.sh
./test_all_datasets_ec2.sh /data /tmp/results
```

### Disk Space
Check available space:
```bash
df -h /tmp
```

Indices will be ~6-8GB each, so ensure at least 50GB free.

### Build Errors
```bash
# Update Rust
rustup update

# Clean and rebuild
cargo clean
cargo build --release --bin bench_cohere_ondemand
```
