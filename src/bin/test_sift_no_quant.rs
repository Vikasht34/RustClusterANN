use rustsptag::spann::{SPANNIndex, DistanceMetric};
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
    println!("=== SIFT: NO quantization (baseline) ===\n");
    
    let data_path = "/Users/viktari/pysptag/data/sift";
    let index_path = "/tmp/sift_no_quant.idx";
    
    // Build index WITHOUT quantization
    println!("Loading SIFT base vectors...");
    let start = Instant::now();
    let base = read_fvecs(&format!("{}/sift_base.fvecs", data_path));
    println!("  Loaded {} vectors ({}D) in {:.2}s\n", base.len(), base[0].len(), start.elapsed().as_secs_f32());
    
    println!("Building index WITHOUT quantization...");
    let start = Instant::now();
    let mut index = SPANNIndex::new();
    index.set_metric(DistanceMetric::L2);
    index.set_hbc_sample_size(Some(200_000));
    index.build(base);
    println!("Build time: {:.2}s\n", start.elapsed().as_secs_f32());
    
    // Save
    println!("Saving index...");
    index.save(index_path, true, true).unwrap();
    
    // Load in on-demand mode
    println!("Loading index...");
    let mut loaded_index = SPANNIndex::load_with_mode(index_path, true).unwrap();
    loaded_index.set_num_heads_to_search(128);
    loaded_index.set_max_check(8192);
    
    let storage = loaded_index.create_async_storage().unwrap();
    println!("Index loaded\n");
    
    // Load queries and ground truth
    println!("Loading queries...");
    let queries = read_fvecs(&format!("{}/sift_query.fvecs", data_path));
    let ground_truth = read_ivecs(&format!("{}/sift_groundtruth.ivecs", data_path));
    
    // Test
    println!("Testing 100 queries...");
    let start = Instant::now();
    let mut total_recall = 0.0;
    
    for (i, query) in queries.iter().take(100).enumerate() {
        let results = loaded_index.search_async(query, 10, &storage).await.unwrap();
        
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
    let avg_recall = total_recall / 100.0;
    
    println!("\n=== Results ===");
    println!("Recall@10: {:.2}%", avg_recall * 100.0);
    println!("Latency: {:.2} ms/query", elapsed.as_secs_f32() * 1000.0 / 100.0);
}
