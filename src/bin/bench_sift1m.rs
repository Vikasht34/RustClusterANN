use rustsptag::spann::SPANNIndex;
use rustsptag::simd;
use std::fs::File;
use std::io::{Read, BufReader};
use std::time::Instant;

fn read_fvecs(filename: &str) -> Vec<Vec<f32>> {
    let file = File::open(filename).expect("Failed to open file");
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();
    
    loop {
        let mut dim_bytes = [0u8; 4];
        if reader.read_exact(&mut dim_bytes).is_err() {
            break;
        }
        let dim = i32::from_le_bytes(dim_bytes) as usize;
        
        let mut vec = vec![0f32; dim];
        let mut bytes = vec![0u8; dim * 4];
        reader.read_exact(&mut bytes).expect("Failed to read vector");
        
        for i in 0..dim {
            vec[i] = f32::from_le_bytes([
                bytes[i * 4],
                bytes[i * 4 + 1],
                bytes[i * 4 + 2],
                bytes[i * 4 + 3],
            ]);
        }
        vectors.push(vec);
    }
    vectors
}

fn read_ivecs(filename: &str) -> Vec<Vec<i32>> {
    let file = File::open(filename).expect("Failed to open file");
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();
    
    loop {
        let mut dim_bytes = [0u8; 4];
        if reader.read_exact(&mut dim_bytes).is_err() {
            break;
        }
        let dim = i32::from_le_bytes(dim_bytes) as usize;
        
        let mut vec = vec![0i32; dim];
        let mut bytes = vec![0u8; dim * 4];
        reader.read_exact(&mut bytes).expect("Failed to read vector");
        
        for i in 0..dim {
            vec[i] = i32::from_le_bytes([
                bytes[i * 4],
                bytes[i * 4 + 1],
                bytes[i * 4 + 2],
                bytes[i * 4 + 3],
            ]);
        }
        vectors.push(vec);
    }
    vectors
}

fn main() {
    println!("=== SPANN on SIFT 1M (SPTAG defaults) ===\n");

    // Load SIFT 1M base vectors
    println!("Loading SIFT 1M base vectors...");
    let start = Instant::now();
    let base = read_fvecs("data/sift/sift_base.fvecs");
    println!("  Loaded {} vectors in {:.2}s", base.len(), start.elapsed().as_secs_f32());
    
    // Load queries
    println!("Loading SIFT queries...");
    let queries = read_fvecs("data/sift/sift_query.fvecs");
    println!("  Loaded {} queries", queries.len());
    
    // Load ground truth
    println!("Loading ground truth...");
    let ground_truth = read_ivecs("data/sift/sift_groundtruth.ivecs");
    println!("  Loaded {} ground truth vectors\n", ground_truth.len());

    // Build SPANN index
    println!("Building SPANN index...");
    let start = Instant::now();
    let mut index = SPANNIndex::new();
    index.build(base);
    let build_time = start.elapsed();
    println!("\n=== Build Complete ===");
    println!("Build time: {:.2}s ({:.2} min)", build_time.as_secs_f32(), build_time.as_secs_f32() / 60.0);

    // Warmup
    println!("\nWarming up (10 queries)...");
    for i in 0..10 {
        let _ = index.search(&queries[i], 10);
    }

    // Search all queries
    println!("\nSearching {} queries...", queries.len());
    let mut latencies = Vec::new();
    let mut recalls = Vec::new();
    
    let overall_start = Instant::now();
    for (i, query) in queries.iter().enumerate() {
        let start = Instant::now();
        let results = index.search(query, 10);
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
        
        if (i + 1) % 100 == 0 {
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
    println!("Dataset:      SIFT 1M (1,000,000 vectors, 128D)");
    println!("Index size:   {} vectors", index.len());
    println!("Build time:   {:.2}s ({:.2} min)", build_time.as_secs_f32(), build_time.as_secs_f32() / 60.0);
    println!("Latency p50:  {:.3} ms", p50);
    println!("Latency p99:  {:.3} ms", p99);
    println!("QPS:          {:.2}", qps);
    println!("Recall@10:    {:.2}%", avg_recall * 100.0);
}
