use rustsptag::spann::SPANNIndex;
use rustsptag::simd;
use std::fs::File;
use std::io::{Read, BufReader};
use std::time::Instant;
use std::env;

fn read_binary_vectors(filename: &str) -> Vec<Vec<f32>> {
    let file = File::open(filename).expect("Failed to open file");
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
    let file = File::open(filename).expect("Failed to open file");
    let mut reader = BufReader::new(file);
    
    let mut header = [0u8; 8];
    reader.read_exact(&mut header).expect("Failed to read header");
    let n = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let k = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
    
    let mut vectors = Vec::with_capacity(n);
    let mut buffer = vec![0u8; k * 4];
    
    for _ in 0..n {
        reader.read_exact(&mut buffer).expect("Failed to read vector");
        let vec: Vec<i32> = (0..k)
            .map(|i| i32::from_le_bytes([
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

fn main() {
    let args: Vec<String> = env::args().collect();
    
    let data_path = if args.len() > 1 && args[1] == "--data-path" {
        args[2].clone()
    } else {
        "data/documents-1m.hdf5".to_string()
    };
    
    let index_path = if args.len() > 3 && args[3] == "--index-path" {
        args[4].clone()
    } else {
        "cohere_1m.idx".to_string()
    };
    
    let enable_compression = args.contains(&"--enable-compression".to_string());
    
    println!("=== SPANN on Cohere 1M ===\n");

    // Load Cohere 1M dataset
    println!("Loading Cohere 1M from: {}", data_path);
    let start = Instant::now();
    let base = read_binary_vectors(&format!("{}/base.bin", data_path));
    println!("  Loaded {} vectors ({}D) in {:.2}s", base.len(), base[0].len(), start.elapsed().as_secs_f32());
    
    println!("Loading queries...");
    let queries = read_binary_vectors(&format!("{}/query.bin", data_path));
    println!("  Loaded {} queries", queries.len());
    
    println!("Loading ground truth...");
    let ground_truth = read_binary_groundtruth(&format!("{}/groundtruth.bin", data_path));
    println!("  Loaded {} ground truth vectors (top-{})\n", ground_truth.len(), ground_truth[0].len());

    // Build SPANN index
    println!("Building SPANN index...");
    let start = Instant::now();
    let mut index = SPANNIndex::new();
    index.set_metric(rustsptag::spann::DistanceMetric::InnerProduct);
    index.set_hbc_sample_size(Some(200_000));
    index.build(base);
    let build_time = start.elapsed();
    println!("\nBuild time: {:.2}s ({:.2} min)", build_time.as_secs_f32(), build_time.as_secs_f32() / 60.0);
    
    // Save index
    println!("\nSaving index to: {}", index_path);
    let start = Instant::now();
    index.save(&index_path, enable_compression, false).unwrap();
    println!("  Save time: {:.2}s", start.elapsed().as_secs_f32());
    let size = std::fs::metadata(&index_path).unwrap().len();
    println!("  File size: {:.2} MB\n", size as f32 / 1024.0 / 1024.0);
    
    // Load index
    println!("Loading index from disk...");
    let start = Instant::now();
    let loaded_index = SPANNIndex::load(&index_path).unwrap();
    println!("  Load time: {:.2}s\n", start.elapsed().as_secs_f32());

    // Warmup
    println!("Warming up (10 queries)...");
    for i in 0..10 {
        let _ = loaded_index.search(&queries[i], 10);
    }

    // Search all queries
    println!("\nSearching {} queries...", queries.len());
    let mut latencies = Vec::new();
    let mut recalls = Vec::new();
    
    let overall_start = Instant::now();
    for (i, query) in queries.iter().enumerate() {
        let start = Instant::now();
        let results = loaded_index.search(query, 10);
        let latency = start.elapsed();
        latencies.push(latency.as_secs_f64() * 1000.0); // Convert to ms
        
        // Compute recall
        let result_ids: Vec<i32> = results.iter().map(|(id, _)| *id as i32).collect();
        let gt_ids = &ground_truth[i];
        let matches = result_ids.iter()
            .filter(|id| gt_ids.contains(id))
            .count();
        let recall = matches as f32 / 10.0;
        recalls.push(recall);
        
        if (i + 1) % 1000 == 0 {
            println!("  Processed {}/{} queries", i + 1, queries.len());
        }
    }
    let total_search_time = overall_start.elapsed();
    
    // Compute statistics
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies[latencies.len() / 2];
    let p90 = latencies[latencies.len() * 9 / 10];
    let p99 = latencies[latencies.len() * 99 / 100];
    let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let min_latency = latencies[0];
    let max_latency = latencies[latencies.len() - 1];
    
    let avg_recall = recalls.iter().sum::<f32>() / recalls.len() as f32;
    let qps = queries.len() as f64 / total_search_time.as_secs_f64();

    // Print results
    println!("\n=== Search Results ===");
    println!("Total queries: {}", queries.len());
    println!("Total search time: {:.2}s", total_search_time.as_secs_f32());
    println!();
    
    println!("=== Latency Statistics (ms) ===");
    println!("  Min:     {:.3} ms", min_latency);
    println!("  p50:     {:.3} ms", p50);
    println!("  p90:     {:.3} ms", p90);
    println!("  p99:     {:.3} ms", p99);
    println!("  Max:     {:.3} ms", max_latency);
    println!("  Average: {:.3} ms", avg_latency);
    println!();
    
    println!("=== Throughput ===");
    println!("  QPS: {:.2}", qps);
    println!();
    
    println!("=== Recall ===");
    println!("  Recall@10: {:.2}%", avg_recall * 100.0);
    println!();
    
    println!("=== Summary ===");
    println!("Dataset:      Cohere 1M (1,000,000 vectors, 768D)");
    println!("Index size:   {} vectors", index.len());
    println!("Build time:   {:.2}s ({:.2} min)", build_time.as_secs_f32(), build_time.as_secs_f32() / 60.0);
    println!("Latency p50:  {:.3} ms", p50);
    println!("Latency p99:  {:.3} ms", p99);
    println!("QPS:          {:.2}", qps);
    println!("Recall@10:    {:.2}%", avg_recall * 100.0);
}
