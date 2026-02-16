#!/bin/bash

# Complete EC2 setup: download datasets, convert to binary, and run benchmarks
# Usage: ./setup_and_test_ec2.sh

set -e

DATA_ROOT="/data"
OUTPUT_ROOT="/tmp/results"

# Dataset configurations: name, hf_dataset_id, split, k
declare -a DATASETS=(
    "gist:Vikasht34/gist-960-euclidean:test:100"
    "glove:Vikasht34/glove-100-angular:test:100"
    "mpnet-msmarco:Vikasht34/msmarco-mpnet-base-dot-v5:test:10"
    "tasb-msmarco:Vikasht34/msmarco-tasb-dot:test:10"
    "snowflake-msmarco:Vikasht34/msmarco-snowflake-arctic-embed-m-v1.5:test:10"
)

echo "=========================================="
echo "EC2 Dataset Setup and Testing"
echo "=========================================="
echo "Data root: $DATA_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Timestamp: $(date)"
echo ""

# Create directories
mkdir -p "$DATA_ROOT"
mkdir -p "$OUTPUT_ROOT"

# Function to download and convert a dataset
setup_dataset() {
    local config=$1
    IFS=':' read -r name hf_id split k <<< "$config"
    
    local dataset_dir="${DATA_ROOT}/${name}"
    
    echo "=========================================="
    echo "Setting up: $name"
    echo "HuggingFace: $hf_id"
    echo "Directory: $dataset_dir"
    echo "=========================================="
    
    # Check if already exists
    if [ -f "${dataset_dir}/base.bin" ] && [ -f "${dataset_dir}/query.bin" ] && [ -f "${dataset_dir}/groundtruth.bin" ]; then
        echo "✓ Dataset already exists, skipping download"
        return 0
    fi
    
    mkdir -p "$dataset_dir"
    
    # Create Python conversion script
    cat > /tmp/convert_${name}.py << 'PYTHON_SCRIPT'
import sys
import struct
from datasets import load_dataset

def write_fvecs(filename, data):
    """Write vectors in fvecs format (used by SPTAG)"""
    with open(filename, 'wb') as f:
        n = len(data)
        d = len(data[0]) if n > 0 else 0
        # Write header: n (uint32), d (uint32)
        f.write(struct.pack('I', n))
        f.write(struct.pack('I', d))
        # Write vectors
        for vec in data:
            f.write(struct.pack(f'{len(vec)}f', *vec))

def write_ivecs(filename, data):
    """Write integer vectors (groundtruth)"""
    with open(filename, 'wb') as f:
        n = len(data)
        k = len(data[0]) if n > 0 else 0
        # Write header: n (uint32), k (uint32)
        f.write(struct.pack('I', n))
        f.write(struct.pack('I', k))
        # Write vectors
        for vec in data:
            f.write(struct.pack(f'{len(vec)}i', *vec))

if __name__ == "__main__":
    dataset_id = sys.argv[1]
    split = sys.argv[2]
    output_dir = sys.argv[3]
    
    print(f"Loading dataset: {dataset_id} (split: {split})")
    dataset = load_dataset(dataset_id, split=split)
    
    print(f"Converting base vectors...")
    base = [item['emb'] for item in dataset]
    write_fvecs(f"{output_dir}/base.bin", base)
    print(f"  Wrote {len(base)} base vectors")
    
    print(f"Converting query vectors...")
    queries = [item['query_emb'] for item in dataset]
    write_fvecs(f"{output_dir}/query.bin", queries)
    print(f"  Wrote {len(queries)} query vectors")
    
    print(f"Converting groundtruth...")
    groundtruth = [item['neighbors'] for item in dataset]
    write_ivecs(f"{output_dir}/groundtruth.bin", groundtruth)
    print(f"  Wrote {len(groundtruth)} groundtruth entries")
    
    print(f"✓ Conversion complete!")
PYTHON_SCRIPT
    
    # Run conversion
    echo "Downloading and converting dataset..."
    python3 /tmp/convert_${name}.py "$hf_id" "$split" "$dataset_dir"
    
    # Cleanup
    rm /tmp/convert_${name}.py
    
    echo "✓ Setup complete for $name"
    echo ""
}

# Install dependencies if needed
echo "Checking dependencies..."
if ! command -v python3 &> /dev/null; then
    echo "Installing Python3..."
    sudo yum install -y python3 || sudo apt-get install -y python3
fi

if ! python3 -c "import datasets" 2>/dev/null; then
    echo "Installing HuggingFace datasets..."
    pip3 install datasets --user
fi

echo "✓ Dependencies ready"
echo ""

# Setup all datasets sequentially (to avoid memory issues)
for config in "${DATASETS[@]}"; do
    setup_dataset "$config"
done

echo ""
echo "=========================================="
echo "All datasets ready!"
echo "=========================================="
echo ""
echo "Dataset locations:"
for config in "${DATASETS[@]}"; do
    IFS=':' read -r name _ _ _ <<< "$config"
    echo "  ${DATA_ROOT}/${name}/"
done

echo ""
echo "Starting benchmarks..."
echo ""

# Run benchmarks
./test_all_datasets_ec2.sh "$DATA_ROOT" "$OUTPUT_ROOT"

echo ""
echo "=========================================="
echo "Complete! Results in: $OUTPUT_ROOT"
echo "=========================================="
