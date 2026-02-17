use rustsptag::spann::{SPANNIndex, DistanceMetric, QuantizationType};
use std::fs::File;
use std::io::{BufReader, Read};
use std::time::Instant;

fn read_fvecs(filename: &str) -> Vec<Vec<f32>> {
    let file = File::open(filename).unwrap();
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();
    
    loop {
        let mut dim_bytes = [0u8; 4];
        if reader.read_exact(&mut dim_bytes).is_err() {
            break;
        }
        let dim = i32::from_le_bytes(dim_bytes) as usize;
        
        let mut buffer = vec![0u8; dim * 4];
        if reader.read_exact(&mut buffer).is_err() {
            break;
        }
        
        let vec: Vec<f32> = buffer.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        vectors.push(vec);
    }
    
    vectors
}

fn read_ivecs(filename: &str) -> Vec<Vec<i32>> {
    let file = File::open(filename).unwrap();
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();
    
    loop {
        let mut dim_bytes = [0u8; 4];
        if reader.read_exact(&mut dim_bytes).is_err() {
            break;
        }
        let dim = i32::from_le_bytes(dim_bytes) as usize;
        
        let mut buffer = vec![0u8; dim * 4];
        if reader.read_exact(&mut buffer).is_err() {
            break;
        }
        
        let vec: Vec<i32> = buffer.chunks_exact(4)
            .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        vectors.push(vec);
    }
    
    vectors
}

#[tokio::main]
async fn main() {
    println!("=== SIFT 1M: 1-bit Quantization with Reranking ===\n");
    
    let data_path = "/Users/viktari/rustsptag/data/sift";
    let index_path = "/tmp/sift_1bit.idx";
    
    // Build index with 1-bit quantization
    println!("Loading SIFT base vectors...");
    let start = Instant::now();
    let base = read_fvecs(&format!("{}/sift_learn.fvecs", data_path));
    println!("  Loaded {} vectors ({}D) in {:.2}s\n", base.len(), base[0].len(), start.elapsed().as_secs_f32());
    
    println!("Building index with 1-bit quantization...");
    let start = Instant::now();
    let mut index = SPANNIndex::new();
    index.set_metric(DistanceMetric::L2);
    index.set_quantization(QuantizationType::OneBit);
    index.set_hbc_sample_size(Some(200_000));
    index.build(base);
    println!("Build time: {:.2}s\n", start.elapsed().as_secs_f32());
    
    // Save with compression
    println!("Saving index...");
    let start = Instant::now();
    index.save(index_path, true, true).unwrap();
    println!("Save time: {:.2}s\n", start.elapsed().as_secs_f32());
    
    // Report sizes
    let posting_size = std::fs::metadata(index_path).unwrap().len();
    let head_size = std::fs::metadata(&format!("{}.head_index", index_path)).unwrap().len();
    let vectors_size = std::fs::metadata(&format!("{}.vectors", index_path)).unwrap().len();
    let total_size = posting_size + head_size + vectors_size;
    
    println!("=== Index Size ===");
    println!("Posting lists (1-bit + zstd): {:.2} MB", posting_size as f32 / 1024.0 / 1024.0);
    println!("Head index: {:.2} MB", head_size as f32 / 1024.0 / 1024.0);
    println!("Full vectors (for rerank): {:.2} MB", vectors_size as f32 / 1024.0 / 1024.0);
    println!("Total: {:.2} MB\n", total_size as f32 / 1024.0 / 1024.0);
    
    // Load in on-demand mode
    println!("Loading index in on-demand mode...");
    let mut loaded_index = SPANNIndex::load_with_mode(index_path, true).unwrap();
    loaded_index.set_num_heads_to_search(128);
    loaded_index.set_max_check(8192);
    
    let storage = loaded_index.create_async_storage().unwrap();
    println!("Index loaded (heads in RAM, posting lists on disk)\n");
    
    // Load queries and ground truth
    println!("Loading queries...");
    let queries = read_fvecs(&format!("{}/sift_query.fvecs", data_path));
    println!("  Loaded {} queries\n", queries.len());
    
    println!("Loading ground truth...");
    let ground_truth = read_ivecs(&format!("{}/sift_groundtruth.ivecs", data_path));
    println!("  Loaded {} ground truth (top-{})\n", ground_truth.len(), ground_truth[0].len());
    
    // Test with rerank factor 3
    println!("=== Testing with rerank_factor=3 ===");
    println!("Stage 1: Get top-30 candidates with 1-bit quantization");
    println!("Stage 2: Rerank with full precision, return top-10\n");
    
    let start = Instant::now();
    let mut total_recall = 0.0;
    
    for (i, query) in queries.iter().enumerate() {
        let results = loaded_index.search_async_with_rerank(query, 10, 3, &storage).await.unwrap();
        
        let result_ids: std::collections::HashSet<_> = results.iter().map(|(id, _)| *id).collect();
        let gt_ids: std::collections::HashSet<_> = ground_truth[i].iter().take(10).map(|&id| id as usize).collect();
        let intersection = result_ids.intersection(&gt_ids).count();
        let recall = intersection as f32 / 10.0;
        total_recall += recall;
        
        if i < 3 {
            println!("Query {}: recall={:.1}%", i, recall * 100.0);
        }
    }
    
    let elapsed = start.elapsed();
    let avg_recall = total_recall / queries.len() as f32;
    let qps = queries.len() as f32 / elapsed.as_secs_f32();
    
    println!("\n=== Results ===");
    println!("Queries: {}", queries.len());
    println!("Recall@10: {:.2}%", avg_recall * 100.0);
    println!("Total time: {:.2}s", elapsed.as_secs_f32());
    println!("Latency: {:.2} ms/query", elapsed.as_secs_f32() * 1000.0 / queries.len() as f32);
    println!("QPS: {:.2}", qps);
}
