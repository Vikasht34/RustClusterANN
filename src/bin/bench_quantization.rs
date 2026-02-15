use rustsptag::spann::{SPANNIndex, DistanceMetric, QuantizationType};
use std::fs::File;
use std::io::{Read, BufReader};
use std::time::Instant;

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

fn test_quantization(quant_type: QuantizationType, base: Vec<Vec<f32>>, queries: &[Vec<f32>], ground_truth: &[Vec<i32>]) {
    println!("\n=== Testing {:?} ===", quant_type);
    
    let start = Instant::now();
    let mut index = SPANNIndex::new();
    index.set_metric(DistanceMetric::InnerProduct);
    index.set_hbc_sample_size(Some(200_000));
    index.set_quantization(quant_type);
    index.build(base);
    let build_time = start.elapsed();
    
    println!("Build time: {:.2}s ({:.2} min)", build_time.as_secs_f32(), build_time.as_secs_f32() / 60.0);
    
    // Search sample queries
    let sample_size = 1000.min(queries.len());
    let mut latencies = Vec::new();
    let mut recalls = Vec::new();
    
    for i in 0..sample_size {
        let start = Instant::now();
        let results = index.search(&queries[i], 10);
        latencies.push(start.elapsed().as_secs_f64() * 1000.0);
        
        let gt_set: std::collections::HashSet<_> = ground_truth[i].iter().take(10).collect();
        let found = results.iter().filter(|(id, _)| gt_set.contains(&(*id as i32))).count();
        recalls.push(found as f32 / 10.0);
    }
    
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = latencies[latencies.len() / 2];
    let p99 = latencies[latencies.len() * 99 / 100];
    let avg_recall = recalls.iter().sum::<f32>() / recalls.len() as f32;
    
    println!("Latency p50: {:.3} ms", p50);
    println!("Latency p99: {:.3} ms", p99);
    println!("Recall@10: {:.2}%", avg_recall * 100.0);
}

fn main() {
    println!("=== SPANN Quantization Comparison on Cohere 1M ===\n");
    
    println!("Loading Cohere 1M base vectors...");
    let start = Instant::now();
    let base = read_binary_vectors("data/cohere_base.bin");
    println!("  Loaded {} vectors ({}D) in {:.2}s", base.len(), base[0].len(), start.elapsed().as_secs_f32());
    
    println!("Loading Cohere queries...");
    let queries = read_binary_vectors("data/cohere_query.bin");
    println!("  Loaded {} queries", queries.len());
    
    println!("Loading ground truth...");
    let ground_truth = read_binary_groundtruth("data/cohere_groundtruth.bin");
    println!("  Loaded {} ground truth vectors (top-{})\n", ground_truth.len(), ground_truth[0].len());
    
    // Test each quantization level
    test_quantization(QuantizationType::None, base.clone(), &queries, &ground_truth);
    test_quantization(QuantizationType::OneBit, base.clone(), &queries, &ground_truth);
    test_quantization(QuantizationType::TwoBit, base.clone(), &queries, &ground_truth);
    test_quantization(QuantizationType::FourBit, base, &queries, &ground_truth);
}
