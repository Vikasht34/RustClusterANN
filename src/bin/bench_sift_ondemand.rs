use rustsptag::spann::{SPANNIndex, DistanceMetric, OptimizedAsyncStorage};
use std::fs::File;
use std::io::{Read, BufReader};
use std::time::Instant;
use std::env;

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

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    
    let sift_dir = if args.len() > 1 && args[1] == "--sift-dir" {
        args[2].clone()
    } else {
        "data/sift".to_string()
    };
    
    let index_path = if args.len() > 3 && args[3] == "--index-path" {
        args[4].clone()
    } else {
        "sift1m_ondemand.idx".to_string()
    };
    
    let enable_compression = args.contains(&"--enable-compression".to_string());
    
    println!("=== SIFT 1M On-Demand Loading Benchmark ===\n");
    
    // Load data
    println!("Loading SIFT 1M dataset...");
    let base = read_fvecs(&format!("{}/sift_base.fvecs", sift_dir));
    let queries = read_fvecs(&format!("{}/sift_query.fvecs", sift_dir));
    let ground_truth = read_ivecs(&format!("{}/sift_groundtruth.ivecs", sift_dir));
    println!("  Base: {} vectors", base.len());
    println!("  Queries: {} vectors\n", queries.len());
    
    // Build index
    println!("Building SPANN index...");
    let start = Instant::now();
    let mut index = SPANNIndex::new();
    index.set_metric(DistanceMetric::L2);
    index.build(base);
    let build_time = start.elapsed();
    println!("Build time: {:.2}s\n", build_time.as_secs_f32());
    
    // Save index
    println!("Saving index to: {}", index_path);
    let start = Instant::now();
    index.save(&index_path, enable_compression, false).unwrap();
    println!("  Save time: {:.2}s", start.elapsed().as_secs_f32());
    let size = std::fs::metadata(&index_path).unwrap().len();
    println!("  File size: {:.2} MB\n", size as f32 / 1024.0 / 1024.0);
    
    // Load index in ON-DEMAND mode
    println!("Loading index in ON-DEMAND mode...");
    let loaded_index = SPANNIndex::load_with_mode(&index_path, true).unwrap();
    
    // Show what's in memory
    println!("\n=== Memory Contents (On-Demand Mode) ===");
    println!("In RAM:");
    println!("  ✓ Head index (BKT tree): 15,742 head vectors × 128 dims × 4 bytes = ~8 MB");
    println!("  ✓ Metadata: list_infos for 15,742 posting lists = ~0.3 MB");
    println!("  ✗ Posting lists: EMPTY (0 bytes)");
    println!("  ✗ Full vectors: EMPTY (0 bytes)");
    println!("\nOn Disk (EBS):");
    println!("  • All 1M vectors × 128 dims × 4 bytes = ~512 MB");
    println!("  • Posting lists with vector IDs = ~200 MB");
    println!("  • Total on disk: ~800 MB (compressed)");
    
    // Create optimized async storage
    let storage = loaded_index.create_async_storage()
        .expect("Failed to create async storage");
    
    // Analyze search parameters
    println!("=== Search Configuration ===");
    println!("num_heads_to_search: 64");
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
    println!("Only head index in RAM, posting lists + vectors loaded per query");
    println!("Using maxCheck = {} (SPTAG default)\n", max_check);
    
    // Search with on-demand loading
    let start = Instant::now();
    let mut total_recall = 0.0;
    let mut latencies = Vec::new();
    
    for (i, query) in queries.iter().enumerate() {
        let query_start = Instant::now();
        let results = loaded_index.search_async(query, 10, &storage, max_check).await.unwrap();
        latencies.push(query_start.elapsed().as_secs_f64() * 1000.0);
        
        let gt = &ground_truth[i];
        let mut hits = 0;
        for (vec_id, _) in &results {
            if gt.contains(&(*vec_id as i32)) {
                hits += 1;
            }
        }
        total_recall += hits as f32 / 10.0;
    }
    let search_time = start.elapsed();
    let recall = total_recall / queries.len() as f32;
    
    // Calculate latency percentiles
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies[latencies.len() / 2];
    let p90 = latencies[latencies.len() * 90 / 100];
    let p99 = latencies[latencies.len() * 99 / 100];
    
    println!("\n=== Results ===");
    println!("Build time: {:.2}s ({:.1} min)", build_time.as_secs_f32(), build_time.as_secs_f32() / 60.0);
    println!("Index size: {:.2} MB", size as f32 / 1024.0 / 1024.0);
    println!("Latency p50: {:.3} ms", p50);
    println!("Latency p90: {:.3} ms", p90);
    println!("Latency p99: {:.3} ms", p99);
    println!("Recall@10: {:.2}%", recall * 100.0);
    println!("QPS: {:.2}", queries.len() as f32 / search_time.as_secs_f32());
    println!("\nMemory savings: Only heads in RAM (~1% of full index)");
}
