#!/bin/bash
set -e

echo "=== SPANN EC2 Setup ==="

# Install Rust if not present
if ! command -v cargo &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
fi

# Clone repository
if [ ! -d "rustsptag" ]; then
    echo "Cloning repository..."
    git clone https://github.com/Vikasht34/rustsptag.git
fi

cd rustsptag

# Build release binaries
echo "Building release binaries..."
cargo build --release --bin bench_sift_storage --bin bench_cohere1m

# Create output directories
mkdir -p /data/results/sift
mkdir -p /data/results/cohere
mkdir -p /nvme/indexes/sift
mkdir -p /nvme/indexes/cohere

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run benchmarks in parallel:"
echo "  ./scripts/bench_sift_ec2.sh &"
echo "  ./scripts/bench_cohere_ec2.sh &"
echo ""
echo "Monitor progress:"
echo "  tail -f /data/results/sift/*.log"
echo "  tail -f /data/results/cohere/*.log"
