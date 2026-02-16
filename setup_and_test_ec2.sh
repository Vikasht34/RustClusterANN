#!/bin/bash

# Complete EC2 setup: download datasets from S3, convert to binary, and run benchmarks
# Usage: ./setup_and_test_ec2.sh

set -e

DATA_ROOT="/data"
OUTPUT_ROOT="/tmp/results"
S3_BUCKET="s3://one-click-tests/datasets"

# Dataset configurations: name, s3_file, k, metric
declare -a DATASETS=(
    "gist:gist-960-euclidean.hdf5:100:l2"
    "glove:glove-200-angular.hdf5:100:cosine"
    "mpnet-msmarco:mpnet_marco.hdf5:100:l2"
    "tasb-msmarco:marco_tasb.hdf5:100:inner-product"
    "snowflake-msmarco:snowflake_embeddings.hdf5:100:l2"
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
    IFS=':' read -r name s3_file k metric <<< "$config"
    
    local dataset_dir="${DATA_ROOT}/${name}"
    local hdf5_file="${dataset_dir}/${s3_file}"
    
    echo "=========================================="
    echo "Setting up: $name"
    echo "S3 file: ${S3_BUCKET}/${s3_file}"
    echo "Directory: $dataset_dir"
    echo "=========================================="
    
    # Check if already exists
    if [ -f "${dataset_dir}/base.bin" ] && [ -f "${dataset_dir}/query.bin" ] && [ -f "${dataset_dir}/groundtruth.bin" ]; then
        echo "✓ Dataset already exists, skipping download"
        return 0
    fi
    
    mkdir -p "$dataset_dir"
    
    # Download from S3 if not exists
    if [ ! -f "$hdf5_file" ]; then
        echo "Downloading from S3..."
        aws s3 cp "${S3_BUCKET}/${s3_file}" "$hdf5_file"
    else
        echo "✓ HDF5 file already downloaded"
    fi
    
    # Create Python conversion script
    cat > /tmp/convert_${name}.py << 'PYTHON_SCRIPT'
import sys
import struct
import h5py

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
    hdf5_file = sys.argv[1]
    output_dir = sys.argv[2]
    
    print(f"Loading HDF5 file: {hdf5_file}")
    with h5py.File(hdf5_file, 'r') as f:
        print(f"Available keys: {list(f.keys())}")
        
        # Read base vectors (train dataset)
        print(f"Converting base vectors...")
        base = f['train'][:]
        write_fvecs(f"{output_dir}/base.bin", base)
        print(f"  Wrote {len(base)} base vectors ({base.shape[1]}D)")
        
        # Read query vectors (test dataset)
        print(f"Converting query vectors...")
        queries = f['test'][:]
        write_fvecs(f"{output_dir}/query.bin", queries)
        print(f"  Wrote {len(queries)} query vectors")
        
        # Read groundtruth
        print(f"Converting groundtruth...")
        groundtruth = f['neighbors'][:]
        write_ivecs(f"{output_dir}/groundtruth.bin", groundtruth)
        print(f"  Wrote {len(groundtruth)} groundtruth entries (k={groundtruth.shape[1]})")
    
    print(f"✓ Conversion complete!")
PYTHON_SCRIPT
    
    # Run conversion
    echo "Converting HDF5 to binary format..."
    python3 /tmp/convert_${name}.py "$hdf5_file" "$dataset_dir"
    
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

if ! command -v aws &> /dev/null; then
    echo "Installing AWS CLI..."
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
    rm -rf aws awscliv2.zip
fi

if ! python3 -c "import h5py" 2>/dev/null; then
    echo "Installing h5py..."
    pip3 install h5py --user
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
echo ""
echo "View reports:"
echo "  CSV:      ${OUTPUT_ROOT}/benchmark_summary.csv"
echo "  Markdown: ${OUTPUT_ROOT}/benchmark_report.md"
echo "  Logs:     ${OUTPUT_ROOT}/*_ondemand.log"

