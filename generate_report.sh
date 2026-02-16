#!/bin/bash

# Generate comprehensive benchmark report for all datasets
# Usage: ./generate_report.sh /path/to/results

RESULTS_DIR=${1:-"/tmp/results"}

echo "=========================================="
echo "SPANN On-Demand Benchmark Report"
echo "=========================================="
echo "Generated: $(date)"
echo "Results directory: $RESULTS_DIR"
echo ""

# Create CSV header
CSV_FILE="${RESULTS_DIR}/benchmark_summary.csv"
echo "Dataset,Vectors,Dimension,Metric,Recall@10,p50(ms),p90(ms),p99(ms),QPS,Avg_IO_per_Query(MB),Index_Size(MB),Build_Time(s)" > "$CSV_FILE"

# Create detailed markdown report
MD_FILE="${RESULTS_DIR}/benchmark_report.md"
cat > "$MD_FILE" << 'EOF'
# SPANN On-Demand Benchmark Report

## Test Configuration
- **Mode**: On-Demand (only heads in RAM)
- **Compression**: Zstd
- **Delta Encoding**: Enabled
- **num_heads**: 128
- **max_check**: 4096

## Results Summary

| Dataset | Vectors | Dim | Metric | Recall@10 | p50 (ms) | p90 (ms) | p99 (ms) | QPS | Avg IO/Query (MB) | Index Size (MB) |
|---------|---------|-----|--------|-----------|----------|----------|----------|-----|-------------------|-----------------|
EOF

# Function to extract metrics from log file
extract_metrics() {
    local log_file=$1
    local dataset_name=$2
    
    if [ ! -f "$log_file" ]; then
        echo "⚠️  Log file not found: $log_file"
        return
    fi
    
    # Extract metrics using grep and awk
    local vectors=$(grep "Loaded.*vectors" "$log_file" | head -1 | awk '{print $3}')
    local dimension=$(grep "Loaded.*vectors" "$log_file" | head -1 | grep -oP '\(\K[0-9]+(?=D\))')
    local metric=$(grep "Metric:" "$log_file" | awk '{print $2}')
    local recall=$(grep "Recall@10:" "$log_file" | tail -1 | awk '{print $2}' | tr -d '%')
    local p50=$(grep "p50:" "$log_file" | tail -1 | awk '{print $2}')
    local p90=$(grep "p90:" "$log_file" | tail -1 | awk '{print $2}')
    local p99=$(grep "p99:" "$log_file" | tail -1 | awk '{print $2}')
    local qps=$(grep "QPS:" "$log_file" | tail -1 | awk '{print $2}')
    local io_per_query=$(grep "Avg bytes read:" "$log_file" | tail -1 | awk '{print $6}' | tr -d '()')
    local index_size=$(grep "Index size:" "$log_file" | tail -1 | awk '{print $3}')
    local build_time=$(grep "Build time:" "$log_file" | head -1 | awk '{print $3}' | tr -d 's')
    
    # Add to CSV
    echo "${dataset_name},${vectors},${dimension},${metric},${recall},${p50},${p90},${p99},${qps},${io_per_query},${index_size},${build_time}" >> "$CSV_FILE"
    
    # Add to markdown table
    echo "| ${dataset_name} | ${vectors} | ${dimension} | ${metric} | ${recall}% | ${p50} | ${p90} | ${p99} | ${qps} | ${io_per_query} | ${index_size} |" >> "$MD_FILE"
    
    # Print to console
    echo ""
    echo "=== ${dataset_name} ==="
    echo "  Vectors:        ${vectors}"
    echo "  Dimension:      ${dimension}"
    echo "  Metric:         ${metric}"
    echo "  Recall@10:      ${recall}%"
    echo "  Latency p50:    ${p50} ms"
    echo "  Latency p90:    ${p90} ms"
    echo "  Latency p99:    ${p99} ms"
    echo "  QPS:            ${qps}"
    echo "  IO per query:   ${io_per_query} MB"
    echo "  Index size:     ${index_size} MB"
    echo "  Build time:     ${build_time} s"
}

# Process all log files
echo ""
echo "Processing results..."
echo ""

for log_file in "$RESULTS_DIR"/*_ondemand.log; do
    if [ -f "$log_file" ]; then
        dataset_name=$(basename "$log_file" _ondemand.log)
        extract_metrics "$log_file" "$dataset_name"
    fi
done

# Add detailed sections to markdown
cat >> "$MD_FILE" << 'EOF'

## Detailed Analysis

### Latency Distribution

EOF

# Add latency charts for each dataset
for log_file in "$RESULTS_DIR"/*_ondemand.log; do
    if [ -f "$log_file" ]; then
        dataset_name=$(basename "$log_file" _ondemand.log)
        
        cat >> "$MD_FILE" << EOF

#### ${dataset_name}
\`\`\`
$(grep -A 6 "=== Latency Statistics" "$log_file" | tail -6)
\`\`\`

EOF
    fi
done

# Add IO analysis
cat >> "$MD_FILE" << 'EOF'

### I/O Analysis

EOF

for log_file in "$RESULTS_DIR"/*_ondemand.log; do
    if [ -f "$log_file" ]; then
        dataset_name=$(basename "$log_file" _ondemand.log)
        
        cat >> "$MD_FILE" << EOF

#### ${dataset_name}
\`\`\`
$(grep -A 2 "=== Data Transfer per Query ===" "$log_file" | tail -2)
\`\`\`

EOF
    fi
done

# Add build times
cat >> "$MD_FILE" << 'EOF'

### Build Performance

| Dataset | Build Time | Index Size | Compression Ratio |
|---------|------------|------------|-------------------|
EOF

for log_file in "$RESULTS_DIR"/*_ondemand.log; do
    if [ -f "$log_file" ]; then
        dataset_name=$(basename "$log_file" _ondemand.log)
        build_time=$(grep "Build time:" "$log_file" | head -1 | awk '{print $3}')
        index_size=$(grep "Index size:" "$log_file" | tail -1 | awk '{print $3, $4}')
        
        echo "| ${dataset_name} | ${build_time} | ${index_size} | - |" >> "$MD_FILE"
    fi
done

echo ""
echo "=========================================="
echo "Report Generation Complete!"
echo "=========================================="
echo ""
echo "Files generated:"
echo "  CSV:      $CSV_FILE"
echo "  Markdown: $MD_FILE"
echo ""
echo "Summary:"
cat "$CSV_FILE"
echo ""
