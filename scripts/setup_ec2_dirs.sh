#!/bin/bash
# Setup script for EC2 benchmarking
# Creates necessary directories and mounts NVMe if needed

set -e

echo "========================================="
echo "EC2 Benchmark Setup"
echo "========================================="

# Create base directories
echo "Creating directories..."
sudo mkdir -p /data/results
sudo mkdir -p /nvme/indexes

# Set permissions
echo "Setting permissions..."
sudo chown -R $USER:$USER /data/results
sudo chown -R $USER:$USER /nvme/indexes

# Check if NVMe is mounted
if mountpoint -q /nvme; then
    echo "✓ /nvme is already mounted"
else
    echo "⚠ /nvme is not mounted"
    echo ""
    echo "To mount NVMe drive:"
    echo "  1. Find device: lsblk"
    echo "  2. Format (if needed): sudo mkfs.ext4 /dev/nvme1n1"
    echo "  3. Mount: sudo mount /dev/nvme1n1 /nvme"
    echo "  4. Set permissions: sudo chown -R \$USER:\$USER /nvme"
fi

# Create dataset directories
for dataset in gist glove cohere sift; do
    mkdir -p /nvme/indexes/$dataset
    mkdir -p /data/results/$dataset
    echo "  ✓ Created directories for $dataset"
done

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Directory structure:"
echo "  /data/results/     - Benchmark results (persistent)"
echo "  /nvme/indexes/     - Index files (fast storage)"
echo ""
echo "Next steps:"
echo "  1. Place datasets in /data/{gist,glove,cohere,sift}/"
echo "  2. Run: ./scripts/benchmark_ec2.sh <dataset>"
echo ""
