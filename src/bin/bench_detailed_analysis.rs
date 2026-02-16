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

#[derive(Default)]
struct DetailedStats {
    total_bytes_loaded: Vec<usize>,
    num_posting_lists: Vec<usize>,
    num_candidates: Vec<usize>,
    latencies_ms: Vec<f64>,
    recalls: Vec<f32>,
}

impl DetailedStats {
    fn add(&mut self, bytes: usize, postings: usize, candidates: usize, latency_ms: f64, recall: f32) {
        self.total_bytes_loaded.push(bytes);
        self.num_posting_lists.push(postings);
        self.num_candidates.push(candidates);
        self.latencies_ms.push(latency_ms);
        self.recalls.push(recall);
    }
    
    fn percentile(data: &[f64], p: f64) -> f64 {
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((data.len() as f64 * p) as usize).min(data.len() - 1);
        sorted[idx]
    }
    
    fn percentile_usize(data: &[usize], p: f64) -> usize {
        let mut sorted = data.to_vec();
        sorted.sort();
        let idx = ((data.len() as f64 * p) as usize).min(data.len() - 1);
        sorted[idx]
    }
    
    fn print_report(&self, dataset_name: &str) {
        println!("\n{}", "=".repeat(80));
        println!("DETAILED ANALYSIS: {}", dataset_name);
        println!("{}\n", "=".repeat(80));
        
        // Latency stats
        println!("=== Query Latency ===");
        println!("  p50: {:.3} ms", Self::percentile(&self.latencies_ms, 0.5));
        println!("  p90: {:.3} ms", Self::percentile(&self.latencies_ms, 0.9));
        println!("  p99: {:.3} ms", Self::percentile(&self.latencies_ms, 0.99));
        println!("  avg: {:.3} ms", self.latencies_ms.iter().sum::<f64>() / self.latencies_ms.len() as f64);
        
        // Bytes loaded per query
        let bytes_f64: Vec<f64> = self.total_bytes_loaded.iter().map(|&x| x as f64).collect();
        println!("\n=== Bytes Loaded Per Query ===");
        println!("  p50: {:.2} KB ({:.2} MB)", Self::percentile(&bytes_f64, 0.5) / 1024.0, Self::percentile(&bytes_f64, 0.5) / 1024.0 / 1024.0);
        println!("  p90: {:.2} KB ({:.2} MB)", Self::percentile(&bytes_f64, 0.9) / 1024.0, Self::percentile(&bytes_f64, 0.9) / 1024.0 / 1024.0);
        println!("  p99: {:.2} KB ({:.2} MB)", Self::percentile(&bytes_f64, 0.99) / 1024.0, Self::percentile(&bytes_f64, 0.99) / 1024.0 / 1024.0);
        println!("  avg: {:.2} KB ({:.2} MB)", 
                 self.total_bytes_loaded.iter().sum::<usize>() as f64 / self.total_bytes_loaded.len() as f64 / 1024.0,
                 self.total_bytes_loaded.iter().sum::<usize>() as f64 / self.total_bytes_loaded.len() as f64 / 1024.0 / 1024.0);
        
        // Posting lists loaded
        println!("\n=== Posting Lists Loaded Per Query ===");
        println!("  p50: {}", Self::percentile_usize(&self.num_posting_lists, 0.5));
        println!("  p90: {}", Self::percentile_usize(&self.num_posting_lists, 0.9));
        println!("  p99: {}", Self::percentile_usize(&self.num_posting_lists, 0.99));
        println!("  avg: {:.1}", self.num_posting_lists.iter().sum::<usize>() as f64 / self.num_posting_lists.len() as f64);
        
        // Candidates evaluated
        println!("\n=== Candidates Evaluated Per Query ===");
        println!("  p50: {}", Self::percentile_usize(&self.num_candidates, 0.5));
        println!("  p90: {}", Self::percentile_usize(&self.num_candidates, 0.9));
        println!("  p99: {}", Self::percentile_usize(&self.num_candidates, 0.99));
        println!("  avg: {:.1}", self.num_candidates.iter().sum::<usize>() as f64 / self.num_candidates.len() as f64);
        
        // Recall
        println!("\n=== Recall@10 ===");
        let avg_recall = self.recalls.iter().sum::<f32>() / self.recalls.len() as f32;
        println!("  Average: {:.2}%", avg_recall * 100.0);
        println!("  Min: {:.2}%", self.recalls.iter().cloned().fold(f32::INFINITY, f32::min) * 100.0);
        println!("  Max: {:.2}%", self.recalls.iter().cloned().fold(f32::NEG_INFINITY, f32::max) * 100.0);
    }
}

async fn benchmark_dataset(
    dataset_name: &str,
    base_path: &str,
    query_path: &str,
    gt_path: &str,
    index_path: &str,
    metric: DistanceMetric,
) -> std::io::Result<()> {
    println!("\n{}", "=".repeat(80));
    println!("BENCHMARKING: {}", dataset_name);
    println!("{}\n", "=".repeat(80));
    
    // Load data
    println!("Loading dataset...");
    let base = read_fvecs(base_path)?;
    let queries = read_fvecs(query_path)?;
    let ground_truth = read_ivecs(gt_path)?;
    
    println!("  Base: {} vectors, {} dims", base.len(), base[0].len());
    println!("  Queries: {} vectors", queries.len());
    
    // Build index
    println!("\nBuilding index with 1-bit quantization...");
    let start = Instant::now();
    let mut index = SPANNIndex::new();
    index.set_metric(metric);
    index.set_quantization(QuantizationType::OneBit);
    index.build(base.clone());
    let build_time = start.elapsed();
    println!("  Build time: {:.2}s", build_time.as_secs_f32());
    
    // Save index
    println!("\nSaving index...");
    let start = Instant::now();
    index.save(index_path, true, false)?;
    let save_time = start.elapsed();
    println!("  Save time: {:.2}s", save_time.as_secs_f32());
    
    // Analyze index file sizes
    println!("\n=== INDEX SIZE BREAKDOWN ===");
    let total_size = fs::metadata(index_path)?.len();
    println!("Total index size: {:.2} MB", total_size as f64 / 1024.0 / 1024.0);
    
    // Calculate theoretical sizes
    let num_vectors = base.len();
    let dim = base[0].len();
    let num_heads = (num_vectors as f64 * 0.01) as usize; // 1% heads
    
    println!("\n--- Theoretical Component Sizes ---");
    
    // Head vectors (full precision)
    let head_size = num_heads * dim * 4;
    println!("Head vectors ({} heads × {} dims × 4 bytes):", num_heads, dim);
    println!("  Uncompressed: {:.2} MB", head_size as f64 / 1024.0 / 1024.0);
    
    // Full vectors in posting lists
    let replica_count = 3;
    let full_vectors_in_postings = num_vectors * replica_count * dim * 4;
    println!("\nFull vectors in posting lists ({} vectors × {} replicas × {} dims × 4 bytes):", 
             num_vectors, replica_count, dim);
    println!("  Uncompressed: {:.2} MB", full_vectors_in_postings as f64 / 1024.0 / 1024.0);
    
    // Vector IDs in posting lists
    let vector_ids_size = num_vectors * replica_count * 4;
    println!("\nVector IDs in posting lists ({} vectors × {} replicas × 4 bytes):", 
             num_vectors, replica_count);
    println!("  Uncompressed: {:.2} MB", vector_ids_size as f64 / 1024.0 / 1024.0);
    
    // 1-bit quantized data
    let quantized_size = num_vectors * replica_count * ((dim + 7) / 8);
    println!("\n1-bit quantized vectors ({} vectors × {} replicas × {} bytes):", 
             num_vectors, replica_count, (dim + 7) / 8);
    println!("  Size: {:.2} MB", quantized_size as f64 / 1024.0 / 1024.0);
    
    // Total uncompressed
    let total_uncompressed = head_size + full_vectors_in_postings + vector_ids_size + quantized_size;
    println!("\nTotal uncompressed: {:.2} MB", total_uncompressed as f64 / 1024.0 / 1024.0);
    println!("Actual compressed: {:.2} MB", total_size as f64 / 1024.0 / 1024.0);
    println!("Compression ratio: {:.2}x", total_uncompressed as f64 / total_size as f64);
    
    // Load index in on-demand mode
    println!("\n\nLoading index in ON-DEMAND mode...");
    let loaded_index = SPANNIndex::load_with_mode(index_path, true)?;
    
    // Load list_infos for storage
    let (_postings, _full_vectors, list_infos) = rustsptag::spann::SPANNStorage::load(index_path)?;
    let storage = rustsptag::spann::OptimizedAsyncStorage::new(index_path.to_string(), list_infos, true);
    
    println!("\n=== MEMORY FOOTPRINT (On-Demand Mode) ===");
    let head_memory = num_heads * dim * 4;
    let metadata_memory = num_heads * 22; // ListInfo size
    println!("Head index in RAM: {:.2} MB", head_memory as f64 / 1024.0 / 1024.0);
    println!("Metadata in RAM: {:.2} KB", metadata_memory as f64 / 1024.0);
    println!("Total RAM: {:.2} MB", (head_memory + metadata_memory) as f64 / 1024.0 / 1024.0);
    println!("Posting lists on disk: {:.2} MB", (total_size - head_memory as u64) as f64 / 1024.0 / 1024.0);
    
    // Run detailed benchmark
    println!("\n\nRunning detailed benchmark on {} queries...", queries.len());
    let max_check = 4096;
    let mut stats = DetailedStats::default();
    
    for (i, query) in queries.iter().enumerate() {
        if (i + 1) % 100 == 0 {
            print!("\r  Progress: {}/{}", i + 1, queries.len());
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }
        
        let start = Instant::now();
        let (results, search_stats) = loaded_index.search_async_with_stats(query, 10, &storage, max_check).await?;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        
        // Calculate recall
        let gt = &ground_truth[i];
        let mut hits = 0;
        for (vec_id, _) in &results {
            if gt.contains(&(*vec_id as i32)) {
                hits += 1;
            }
        }
        let recall = hits as f32 / 10.0;
        
        stats.add(
            search_stats.bytes_loaded_from_disk,
            search_stats.num_posting_lists_loaded,
            search_stats.num_candidates_evaluated,
            latency_ms,
            recall,
        );
    }
    println!("\r  Progress: {}/{} ✓", queries.len(), queries.len());
    
    // Print detailed report
    stats.print_report(dataset_name);
    
    Ok(())
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    // SIFT 1M benchmark
    benchmark_dataset(
        "SIFT 1M (1-bit quantization)",
        "/Users/viktari/rustsptag/data/sift/sift_base.fvecs",
        "/Users/viktari/rustsptag/data/sift/sift_query.fvecs",
        "/Users/viktari/rustsptag/data/sift/sift_groundtruth.ivecs",
        "/tmp/spann_indexes/sift_1bit_detailed.idx",
        DistanceMetric::L2,
    ).await?;
    
    // Cohere 1M benchmark
    benchmark_dataset(
        "Cohere 1M (1-bit quantization)",
        "/Users/viktari/rustsptag/data/cohere/base.fvecs",
        "/Users/viktari/rustsptag/data/cohere/query.fvecs",
        "/Users/viktari/rustsptag/data/cohere/groundtruth.ivecs",
        "/tmp/spann_indexes/cohere_1bit_detailed.idx",
        DistanceMetric::InnerProduct,
    ).await?;
    
    Ok(())
}
