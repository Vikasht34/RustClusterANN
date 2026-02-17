use rustsptag::spann::{SPANNIndex, DistanceMetric};
use std::fs::File;
use std::io::{BufReader, Read};

fn read_binary_vectors(filename: &str) -> Vec<Vec<f32>> {
    let file = File::open(filename).unwrap();
    let mut reader = BufReader::new(file);
    
    let mut header = [0u8; 8];
    reader.read_exact(&mut header).unwrap();
    let n = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let d = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
    
    let mut vectors = Vec::with_capacity(n);
    let mut buffer = vec![0u8; d * 4];
    
    for _ in 0..n {
        reader.read_exact(&mut buffer).unwrap();
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

fn read_binary_groundtruth(filename: &str, limit: usize) -> Vec<Vec<usize>> {
    let file = File::open(filename).unwrap();
    let mut reader = BufReader::new(file);
    
    let mut header = [0u8; 8];
    reader.read_exact(&mut header).unwrap();
    let n = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let k = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
    
    let n = n.min(limit);
    let mut groundtruth = Vec::with_capacity(n);
    let mut buffer = vec![0u8; k * 4];
    
    for _ in 0..n {
        reader.read_exact(&mut buffer).unwrap();
        let vec: Vec<usize> = (0..k)
            .map(|i| {
                let id = i32::from_le_bytes([
                    buffer[i * 4],
                    buffer[i * 4 + 1],
                    buffer[i * 4 + 2],
                    buffer[i * 4 + 3],
                ]);
                (id - 1).max(0) as usize
            })
            .collect();
        groundtruth.push(vec);
    }
    
    groundtruth
}

fn main() {
    println!("Building fresh index...");
    let base = read_binary_vectors("/Users/viktari/rustsptag/data/base.bin");
    println!("Loaded {} vectors", base.len());
    
    let mut index = SPANNIndex::new();
    index.set_metric(DistanceMetric::InnerProduct);
    index.set_hbc_sample_size(Some(200_000));
    index.build(base);
    
    // Set search parameters to match working test
    index.set_num_heads_to_search(128);
    index.set_max_check(8192);
    println!("Built!\n");
    
    println!("Loading 100 queries...");
    let queries = read_binary_vectors("/Users/viktari/rustsptag/data/query.bin");
    let queries: Vec<_> = queries.into_iter().take(100).collect();
    
    println!("Loading ground truth...");
    let ground_truth = read_binary_groundtruth("/Users/viktari/rustsptag/data/groundtruth.bin", 100);
    println!("Loaded!\n");
    
    println!("Testing recall on 100 queries...");
    let mut total_recall = 0.0;
    
    for (i, query) in queries.iter().enumerate() {
        let results = index.search(query, 10);
        
        let result_ids: std::collections::HashSet<_> = results.iter().map(|(id, _)| *id).collect();
        let gt_ids: std::collections::HashSet<_> = ground_truth[i].iter().take(10).copied().collect();
        let intersection = result_ids.intersection(&gt_ids).count();
        let recall = intersection as f32 / 10.0;
        total_recall += recall;
        
        if i < 3 {
            println!("Query {}: recall={:.1}%, results={:?}", i, recall * 100.0, 
                results.iter().take(5).map(|(id, _)| id).collect::<Vec<_>>());
        }
    }
    
    let avg_recall = total_recall / queries.len() as f32;
    println!("\nRecall@10: {:.2}%", avg_recall * 100.0);
}
