use rustsptag::spann::SPANNIndex;
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
    println!("=== SIFT: Testing saved 1-bit quantized index ===\n");
    
    let data_path = "/Users/viktari/rustsptag/data/sift";
    let index_path = "/tmp/sift_1bit.idx";
    
    // Load index
    println!("Loading index from {}...", index_path);
    let mut index = SPANNIndex::load_with_mode(index_path, true).unwrap();
    index.set_num_heads_to_search(512);
    index.set_max_check(16384);
    
    let storage = index.create_async_storage().unwrap();
    println!("Index loaded\n");
    
    // Load queries and ground truth
    println!("Loading queries...");
    let queries = read_fvecs(&format!("{}/sift_query.fvecs", data_path));
    println!("  Loaded {} queries\n", queries.len());
    
    println!("Loading ground truth...");
    let ground_truth = read_ivecs(&format!("{}/sift_groundtruth.ivecs", data_path));
    println!("  Loaded ground truth\n");
    
    // Test with rerank factor 3
    println!("Testing with rerank_factor=3...");
    let start = Instant::now();
    let mut total_recall = 0.0;
    
    for (i, query) in queries.iter().take(100).enumerate() {
        let results = index.search_async_with_rerank(query, 10, 20, &storage).await.unwrap();
        
        let result_ids: std::collections::HashSet<_> = results.iter().map(|(id, _)| *id).collect();
        let gt_ids: std::collections::HashSet<_> = ground_truth[i].iter().take(10).map(|&id| id as usize).collect();
        let intersection = result_ids.intersection(&gt_ids).count();
        let recall = intersection as f32 / 10.0;
        total_recall += recall;
        
        if i < 5 {
            println!("Query {}: recall={:.1}%, results={:?}", i, recall * 100.0, 
                results.iter().take(5).map(|(id, _)| id).collect::<Vec<_>>());
            println!("  Ground truth: {:?}", ground_truth[i].iter().take(5).collect::<Vec<_>>());
        }
    }
    
    let elapsed = start.elapsed();
    let avg_recall = total_recall / 100.0;
    
    println!("\n=== Results (100 queries) ===");
    println!("Recall@10: {:.2}%", avg_recall * 100.0);
    println!("Latency: {:.2} ms/query", elapsed.as_secs_f32() * 1000.0 / 100.0);
}
