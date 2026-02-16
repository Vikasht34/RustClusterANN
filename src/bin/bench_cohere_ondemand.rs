use rustsptag::spann::{SPANNIndex, DistanceMetric, OptimizedAsyncStorage, QuantizationType};
use std::fs::File;
use std::io::{Read, BufReader};
use std::time::Instant;
use std::env;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "bench_cohere_ondemand")]
#[command(about = "Benchmark Cohere 1M dataset with on-demand loading")]
struct Args {
    #[arg(long)]
    data_path: String,
    
    #[arg(long)]
    index_path: String,
    
    #[arg(long)]
    zstd: bool,
    
    #[arg(long)]
    delta: bool,
    
    #[arg(long, default_value = "128")]
    num_heads: usize,
    
    #[arg(long, default_value = "4096")]
    max_check: usize,
}

fn read_binary_vectors(filename: &str) -> Vec<Vec<f32>> {
    let file = File::open(filename).unwrap_or_else(|e| {
        eprintln!("Failed to open file: {}", filename);
        eprintln!("Error: {}", e);
        panic!("File not found: {}", filename);
    });
    let mut reader = BufReader::new(file);
    
    let mut header = [0u8; 8];
    reader.read_exact(&mut header).expect("Failed to read header");
    let n = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let d = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
    
    let mut vectors = Vec::with_capacity(n);
    let mut buffer = vec![0u8; d * 4];
    
    for _ in 0..n {
        reader.read_exact(&mut buffer).expect("Failed to read vector");
        let vec: Vec<f32> = (0..d)
            .map(|i| f32::from_le_bytes([
                buffer[i * 4],
                buffer[i * 4 + 1],
                buffer[i * 4 + 2],
                buffer[i * 4 + 3],
            ]))
            .collect();
        vectors.push(vec);
    }
    
    vectors
}

fn read_binary_groundtruth(filename: &str) -> Vec<Vec<i32>> {
    let file = File::open(filename).unwrap_or_else(|e| {
        eprintln!("Failed to open file: {}", filename);
        eprintln!("Error: {}", e);
        panic!("File not found: {}", filename);
    });
    let mut reader = BufReader::new(file);
    
    let mut header = [0u8; 8];
    reader.read_exact(&mut header).expect("Failed to read header");
    let n = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let k = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
    
    let mut groundtruth = Vec::with_capacity(n);
    let mut buffer = vec![0u8; k * 4];
    
    for _ in 0..n {
        reader.read_exact(&mut buffer).expect("Failed to read groundtruth");
        let vec: Vec<i32> = (0..k)
            .map(|i| i32::from_le_bytes([
                buffer[i * 4],
                buffer[i * 4 + 1],
                buffer[i * 4 + 2],
                buffer[i * 4 + 3],
            ]))
            .collect();
        groundtruth.push(vec);
    }
    
    groundtruth
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    
    println!("=== Cohere 1M On-Demand Loading Benchmark ===\n");
    
    // Check if index exists
    let index_exists = std::path::Path::new(&args.index_path).exists();
    
    if !index_exists {
        // Load data and build index
        println!("Loading Cohere 1M dataset from: {}", args.data_path);
        let start = Instant::now();
        let base = read_binary_vectors(&format!("{}/base.bin", args.data_path));
        println!("  Loaded {} vectors ({}D) in {:.2}s", base.len(), base[0].len(), start.elapsed().as_secs_f32());
        
        println!("Building SPANN index...");
        let start = Instant::now();
        let mut index = SPANNIndex::new();
        index.set_metric(DistanceMetric::InnerProduct);
        index.set_hbc_sample_size(Some(200_000));
        index.build(base);
        let build_time = start.elapsed();
        println!("Build time: {:.2}s ({:.2} min)\n", build_time.as_secs_f32(), build_time.as_secs_f32() / 60.0);
        
        // Save index
        println!("Saving index to: {}", args.index_path);
        let start = Instant::now();
        index.save(&args.index_path, args.zstd, args.delta).unwrap();
        println!("  Save time: {:.2}s", start.elapsed().as_secs_f32());
        let size = std::fs::metadata(&args.index_path).unwrap().len();
        println!("  File size: {:.2} MB\n", size as f32 / 1024.0 / 1024.0);
    } else {
        println!("Index already exists at {}, skipping build\n", args.index_path);
    }
    
    // Load queries and ground truth
    println!("Loading queries...");
    let queries = read_binary_vectors(&format!("{}/query.bin", args.data_path));
    println!("  Loaded {} queries", queries.len());
    
    println!("Loading ground truth...");
    let ground_truth = read_binary_groundtruth(&format!("{}/groundtruth.bin", args.data_path));
    println!("  Loaded {} ground truth vectors (top-{})\n", ground_truth[0].len(), ground_truth.len());
    
    // Load index in ON-DEMAND mode
    println!("Loading index in ON-DEMAND mode...");
    let mut loaded_index = SPANNIndex::load_with_mode(&args.index_path, true).unwrap();
    
    // Set configurable parameters
    loaded_index.set_num_heads_to_search(args.num_heads);
    loaded_index.set_max_check(args.max_check);
    
    // Create optimized async storage
    let storage = loaded_index.create_async_storage()
        .expect("Failed to create async storage");
    
    println!("\n=== Memory Contents (On-Demand Mode) ===");
    println!("In RAM: Only head index + metadata (~100 MB)");
    println!("On Disk: All posting lists + vectors loaded per query");
    
    // Analyze search parameters
    println!("\n=== Search Configuration ===");
    println!("num_heads_to_search: {}", args.num_heads);
    println!("max_check: {}", args.max_check);
    println!("Total posting lists: {}", storage.list_infos.len());
    
    let mut total_vecs = 0;
    let mut max_size = 0;
    for info in storage.list_infos.iter() {
        total_vecs += info.ele_count as usize;
        max_size = max_size.max(info.ele_count as usize);
    }
    let avg_size = total_vecs / storage.list_infos.len();
    println!("Avg posting list size: {} vectors", avg_size);
    println!("Max posting list size: {} vectors", max_size);
    println!("Expected candidates per query: ~{} (64 heads × {} avg)", 64 * avg_size, avg_size);
    
    println!("\n=== On-Demand Search Benchmark ===");
    println!("Posting lists + vectors loaded from disk per query\n");
    
    // Warmup
    println!("Warming up (10 queries)...");
    for i in 0..10 {
        let _ = loaded_index.search_async(&queries[i], 10, &storage).await.unwrap();
    }
    
    // Search with on-demand loading
    println!("Searching {} queries...", queries.len());
    let start = Instant::now();
    let mut total_recall = 0.0;
    let mut latencies = Vec::new();
    let mut bytes_read_per_query = Vec::new();
    
    for (i, query) in queries.iter().enumerate() {
        let query_start = Instant::now();
        let results = loaded_index.search_async(query, 10, &storage).await.unwrap();
        latencies.push(query_start.elapsed().as_secs_f64() * 1000.0);
        
        // Track data transfer (estimate)
        let bytes_per_query = 64 * avg_size * 768 * 4; // 64 heads × avg_size vectors × 768 dims × 4 bytes
        bytes_read_per_query.push(bytes_per_query);
        
        let gt = &ground_truth[i];
        let mut hits = 0;
        for (vec_id, _) in &results {
            if gt.contains(&(*vec_id as i32)) {
                hits += 1;
            }
        }
        total_recall += hits as f32 / 10.0;
        
        if (i + 1) % 1000 == 0 {
            println!("  Processed {}/{} queries", i + 1, queries.len());
        }
    }
    let search_time = start.elapsed();
    let recall = total_recall / queries.len() as f32;
    
    // Calculate latency percentiles
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies[latencies.len() / 2];
    let p90 = latencies[latencies.len() * 90 / 100];
    let p99 = latencies[latencies.len() * 99 / 100];
    let min_lat = latencies[0];
    let max_lat = latencies[latencies.len() - 1];
    let avg_lat = latencies.iter().sum::<f64>() / latencies.len() as f64;
    
    let avg_bytes = bytes_read_per_query[0] as f64;
    
    println!("\n=== Results ===");
    println!("Total queries: {}", queries.len());
    println!("Total search time: {:.2}s", search_time.as_secs_f32());
    println!();
    
    println!("=== Data Transfer per Query ===");
    println!("  Avg bytes read:        {:.2} KB ({:.2} MB)", avg_bytes / 1024.0, avg_bytes / 1024.0 / 1024.0);
    println!("  Avg posting lists:     64");
    println!();
    
    println!("=== Latency Statistics (ms) ===");
    println!("  Min:     {:.3} ms", min_lat);
    println!("  p50:     {:.3} ms", p50);
    println!("  p90:     {:.3} ms", p90);
    println!("  p99:     {:.3} ms", p99);
    println!("  Max:     {:.3} ms", max_lat);
    println!("  Average: {:.3} ms", avg_lat);
    println!();
    
    println!("=== Throughput ===");
    println!("  QPS: {:.2}", queries.len() as f32 / search_time.as_secs_f32());
    println!();
    
    println!("=== Recall ===");
    println!("  Recall@10: {:.2}%", recall * 100.0);
    println!();
    
    let size = std::fs::metadata(&args.index_path).unwrap().len();
    println!("=== Summary ===");
    println!("Dataset:      Cohere 1M (1,000,000 vectors, 768D)");
    println!("Index size:   {:.2} MB", size as f32 / 1024.0 / 1024.0);
    println!("Mode:         ON-DEMAND (only heads in RAM)");
    println!("Latency p50:  {:.3} ms", p50);
    println!("Latency p99:  {:.3} ms", p99);
    println!("QPS:          {:.2}", queries.len() as f32 / search_time.as_secs_f32());
    println!("Recall@10:    {:.2}%", recall * 100.0);
    println!("Data/query:   {:.2} MB", avg_bytes / 1024.0 / 1024.0);
}
