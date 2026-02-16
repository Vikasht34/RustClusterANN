use rustsptag::spann::{SPANNIndex, QuantizationType, DistanceMetric};
use std::time::Instant;
use std::fs;

fn read_fvecs(filename: &str) -> std::io::Result<Vec<Vec<f32>>> {
    use std::fs::File;
    use std::io::{BufReader, Read};
    
    let file = File::open(filename)?;
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();
    
    loop {
        let mut dim_bytes = [0u8; 4];
        if reader.read_exact(&mut dim_bytes).is_err() {
            break;
        }
        let dim = i32::from_le_bytes(dim_bytes) as usize;
        
        let mut vec = vec![0f32; dim];
        let mut buffer = vec![0u8; dim * 4];
        reader.read_exact(&mut buffer)?;
        
        for i in 0..dim {
            vec[i] = f32::from_le_bytes([
                buffer[i*4], buffer[i*4+1], buffer[i*4+2], buffer[i*4+3]
            ]);
        }
        vectors.push(vec);
    }
    Ok(vectors)
}

fn read_ivecs(filename: &str) -> std::io::Result<Vec<Vec<i32>>> {
    use std::fs::File;
    use std::io::{BufReader, Read};
    
    let file = File::open(filename)?;
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();
    
    loop {
        let mut dim_bytes = [0u8; 4];
        if reader.read_exact(&mut dim_bytes).is_err() {
            break;
        }
        let dim = i32::from_le_bytes(dim_bytes) as usize;
        
        let mut vec = vec![0i32; dim];
        let mut buffer = vec![0u8; dim * 4];
        reader.read_exact(&mut buffer)?;
        
        for i in 0..dim {
            vec[i] = i32::from_le_bytes([
                buffer[i*4], buffer[i*4+1], buffer[i*4+2], buffer[i*4+3]
            ]);
        }
        vectors.push(vec);
    }
    Ok(vectors)
}

fn percentile(data: &[f64], p: f64) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((data.len() as f64 * p) as usize).min(data.len() - 1);
    sorted[idx]
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let base_path = "/Users/viktari/rustsptag/data/cohere/base.1M.fvecs";
    let query_path = "/Users/viktari/rustsptag/data/cohere/queries.dev.fvecs";
    let gt_path = "/Users/viktari/rustsptag/data/cohere/gt.dev.ibin";
    
    println!("\n{}", "=".repeat(80));
    println!("COHERE 1M - Parameter Sweep");
    println!("{}\n", "=".repeat(80));
    
    // Load data
    println!("Loading dataset...");
    let base = read_fvecs(base_path)?;
    let queries = read_fvecs(query_path)?;
    
    // Load ground truth (binary format)
    let gt_data = std::fs::read(gt_path)?;
    let mut ground_truth = Vec::new();
    let mut offset = 0;
    while offset < gt_data.len() {
        if offset + 4 > gt_data.len() {
            break;
        }
        let count = i32::from_le_bytes([
            gt_data[offset],
            gt_data[offset + 1],
            gt_data[offset + 2],
            gt_data[offset + 3],
        ]) as usize;
        offset += 4;
        
        let mut gt_vec = Vec::new();
        for _ in 0..count {
            if offset + 4 > gt_data.len() {
                break;
            }
            let id = i32::from_le_bytes([
                gt_data[offset],
                gt_data[offset + 1],
                gt_data[offset + 2],
                gt_data[offset + 3],
            ]);
            gt_vec.push(id);
            offset += 4;
        }
        ground_truth.push(gt_vec);
    }
    
    println!("  Base: {} vectors, {} dims", base.len(), base[0].len());
    println!("  Queries: {} vectors", queries.len());
    println!("  Ground truth: {} entries\n", ground_truth.len());
    
    // Build index
    let index_path = "/tmp/spann_indexes/cohere_1bit_sweep.idx";
    
    if !std::path::Path::new(index_path).exists() {
        println!("Building index with 1-bit quantization...");
        let start = Instant::now();
        let mut index = SPANNIndex::new();
        index.set_metric(DistanceMetric::InnerProduct);
        index.set_quantization(QuantizationType::OneBit);
        index.build(base.clone());
        println!("  Build time: {:.2}s", start.elapsed().as_secs_f32());
        
        println!("\nSaving index...");
        index.save(index_path, true, false)?;
        let size = fs::metadata(index_path)?.len();
        println!("  Index size: {:.2} MB\n", size as f64 / 1024.0 / 1024.0);
    } else {
        println!("Using existing index at {}\n", index_path);
    }
    
    // Load index
    let loaded_index = SPANNIndex::load_with_mode(index_path, true)?;
    let (_postings, _full_vectors, list_infos) = rustsptag::spann::SPANNStorage::load(index_path)?;
    let storage = rustsptag::spann::OptimizedAsyncStorage::new(index_path.to_string(), list_infos, true);
    
    // Test configurations
    let configs = vec![
        (32, 1024),  // 32 heads, 1024 candidates
        (32, 2048),  // 32 heads, 2048 candidates
        (32, 4096),  // 32 heads, 4096 candidates
        (64, 2048),  // 64 heads, 2048 candidates (comparison)
        (64, 4096),  // 64 heads, 4096 candidates (baseline)
    ];
    
    println!("\n{}", "=".repeat(80));
    println!("PARAMETER SWEEP RESULTS");
    println!("{}\n", "=".repeat(80));
    
    for (num_heads, max_check) in configs {
        println!("\n--- Configuration: {} heads, maxCheck={} ---", num_heads, max_check);
        
        // Reload index for each configuration
        let mut test_index = SPANNIndex::load_with_mode(index_path, true)?;
        test_index.set_num_heads_to_search(num_heads);
        
        let mut latencies = Vec::new();
        let mut bytes_loaded = Vec::new();
        let mut recalls = Vec::new();
        
        let start = Instant::now();
        for (i, query) in queries.iter().enumerate() {
            if (i + 1) % 100 == 0 {
                print!("\r  Progress: {}/{}", i + 1, queries.len());
                use std::io::Write;
                std::io::stdout().flush().unwrap();
            }
            
            let query_start = Instant::now();
            let (results, stats) = test_index.search_async_with_stats(query, 10, &storage, max_check).await?;
            let query_time = query_start.elapsed().as_secs_f64() * 1000.0;
            
            latencies.push(query_time);
            bytes_loaded.push(stats.bytes_loaded_from_disk);
            
            // Calculate recall
            let gt = &ground_truth[i];
            let mut hits = 0;
            for (vec_id, _) in &results {
                if gt.contains(&(*vec_id as i32)) {
                    hits += 1;
                }
            }
            recalls.push(hits as f32 / 10.0);
        }
        let total_time = start.elapsed();
        println!("\r  Progress: {}/{} ✓", queries.len(), queries.len());
        
        // Calculate statistics
        let avg_recall = recalls.iter().sum::<f32>() / recalls.len() as f32;
        let qps = queries.len() as f64 / total_time.as_secs_f64();
        
        let bytes_f64: Vec<f64> = bytes_loaded.iter().map(|&x| x as f64).collect();
        
        println!("\n  Latency:");
        println!("    p50: {:.2} ms", percentile(&latencies, 0.5));
        println!("    p90: {:.2} ms", percentile(&latencies, 0.9));
        println!("    p99: {:.2} ms", percentile(&latencies, 0.99));
        println!("    avg: {:.2} ms", latencies.iter().sum::<f64>() / latencies.len() as f64);
        
        println!("  Data loaded per query:");
        println!("    p50: {:.2} MB", percentile(&bytes_f64, 0.5) / 1024.0 / 1024.0);
        println!("    p90: {:.2} MB", percentile(&bytes_f64, 0.9) / 1024.0 / 1024.0);
        println!("    avg: {:.2} MB", bytes_f64.iter().sum::<f64>() / bytes_f64.len() as f64 / 1024.0 / 1024.0);
        
        println!("  Recall@10: {:.2}%", avg_recall * 100.0);
        println!("  QPS: {:.1}", qps);
    }
    
    println!("\n{}", "=".repeat(80));
    
    Ok(())
}
