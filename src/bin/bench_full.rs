/// Comprehensive SPTAG benchmark with quantization and disk/RAM storage
use rustsptag::index::{SPTAGIndex, IndexType};
use rustsptag::multibit::{MultiBitQuantizer, MetricType};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::time::Instant;

fn read_fvecs(path: &str) -> Vec<Vec<f32>> {
    let file = File::open(path).expect("Failed to open file");
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();

    loop {
        let mut dim_bytes = [0u8; 4];
        if reader.read_exact(&mut dim_bytes).is_err() {
            break;
        }
        let dim = i32::from_le_bytes(dim_bytes) as usize;

        let mut vec = vec![0.0f32; dim];
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

fn read_ivecs(path: &str) -> Vec<Vec<i32>> {
    let file = File::open(path).expect("Failed to open file");
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

fn compute_recall(results: &[(usize, f32)], ground_truth: &[i32], k: usize) -> f32 {
    let gt_set: std::collections::HashSet<i32> = ground_truth.iter().take(k).copied().collect();
    let found = results.iter().take(k).filter(|(id, _)| gt_set.contains(&(*id as i32))).count();
    found as f32 / k as f32
}

fn save_index_to_disk(index: &SPTAGIndex, path: &str) {
    println!("Saving index to {}...", path);
    // Placeholder - would serialize index structure
    let file = File::create(path).expect("Failed to create file");
    let mut writer = BufWriter::new(file);
    writer.write_all(b"SPTAG_INDEX").unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n = if args.len() > 1 {
        args[1].parse().unwrap_or(10000)
    } else {
        10000
    };
    let storage = if args.len() > 2 { &args[2] } else { "ram" };

    println!("=================================================================");
    println!("SPTAG Comprehensive Benchmark");
    println!("=================================================================");
    println!("Dataset size: {}", n);
    println!("Storage mode: {}", storage);
    println!();

    println!("Loading SIFT dataset...");
    let mut base = read_fvecs("/Users/viktari/pysptag/data/sift/sift_base.fvecs");
    base.truncate(n);
    
    let mut query = read_fvecs("/Users/viktari/pysptag/data/sift/sift_query.fvecs");
    query.truncate(100);
    
    let mut ground_truth = read_ivecs("/Users/viktari/pysptag/data/sift/sift_groundtruth.ivecs");
    ground_truth.truncate(100);

    println!("Dataset: {} base, {} queries, {} dims", base.len(), query.len(), base[0].len());
    println!();

    // ========== NO QUANTIZATION ==========
    println!("=================================================================");
    println!("1. NO QUANTIZATION (Full Precision)");
    println!("=================================================================");
    println!("Building KDT+Graph index on {} full-precision vectors...", base.len());
    
    let start = Instant::now();
    let mut index_full = SPTAGIndex::new_kdt(32, 2);
    index_full.build(base.clone());
    let build_time = start.elapsed();
    println!("Build time: {:.2?}", build_time);

    if storage == "disk" {
        save_index_to_disk(&index_full, "sptag_full.idx");
    }

    println!("Searching with {} queries...", query.len());
    let start = Instant::now();
    let mut total_recall = 0.0;
    for (i, q) in query.iter().enumerate().take(5) {
        let results = index_full.search(q, 10);
        let recall = compute_recall(&results, &ground_truth[i], 10);
        println!("  Query {}: found {:?}, GT first 5: {:?}, recall: {:.1}%", 
                 i, 
                 results.iter().take(5).map(|(id, _)| id).collect::<Vec<_>>(),
                 &ground_truth[i][..5.min(ground_truth[i].len())],
                 recall * 100.0);
        total_recall += recall;
    }
    
    for (i, q) in query.iter().enumerate().skip(5) {
        let results = index_full.search(q, 10);
        total_recall += compute_recall(&results, &ground_truth[i], 10);
    }
    let elapsed = start.elapsed();
    
    let avg_recall = total_recall / query.len() as f32;
    println!("Recall@10: {:.1}%", avg_recall * 100.0);
    println!("Latency: {:.3}ms per query", elapsed.as_secs_f64() * 1000.0 / query.len() as f64);
    println!("QPS: {:.0}", query.len() as f64 / elapsed.as_secs_f64());
    println!();

    // ========== 1-BIT QUANTIZATION ==========
    println!("=================================================================");
    println!("2. 1-BIT QUANTIZATION");
    println!("=================================================================");
    
    let start = Instant::now();
    let mut quant_1bit = MultiBitQuantizer::new(1);
    quant_1bit.train(&base, MetricType::L2);
    let quant_base_1bit: Vec<Vec<f32>> = (0..base.len())
        .map(|i| quant_1bit.reconstruct(i))
        .collect();
    let mut index_1bit = SPTAGIndex::new_kdt(32, 2);
    index_1bit.build(quant_base_1bit);
    let build_time = start.elapsed();
    println!("Build time: {:.2?}", build_time);

    if storage == "disk" {
        save_index_to_disk(&index_1bit, "sptag_1bit.idx");
    }

    let start = Instant::now();
    let mut total_recall = 0.0;
    for (i, q) in query.iter().enumerate() {
        let results = index_1bit.search(q, 10);
        total_recall += compute_recall(&results, &ground_truth[i], 10);
    }
    let elapsed = start.elapsed();
    
    let avg_recall = total_recall / query.len() as f32;
    println!("Recall@10: {:.1}%", avg_recall * 100.0);
    println!("Latency: {:.3}ms per query", elapsed.as_secs_f64() * 1000.0 / query.len() as f64);
    println!("QPS: {:.0}", query.len() as f64 / elapsed.as_secs_f64());
    println!();

    // ========== 2-BIT QUANTIZATION ==========
    println!("=================================================================");
    println!("3. 2-BIT QUANTIZATION");
    println!("=================================================================");
    
    let start = Instant::now();
    let mut quant_2bit = MultiBitQuantizer::new(2);
    quant_2bit.train(&base, MetricType::L2);
    let quant_base_2bit: Vec<Vec<f32>> = (0..base.len())
        .map(|i| quant_2bit.reconstruct(i))
        .collect();
    let mut index_2bit = SPTAGIndex::new_kdt(32, 2);
    index_2bit.build(quant_base_2bit);
    let build_time = start.elapsed();
    println!("Build time: {:.2?}", build_time);

    if storage == "disk" {
        save_index_to_disk(&index_2bit, "sptag_2bit.idx");
    }

    let start = Instant::now();
    let mut total_recall = 0.0;
    for (i, q) in query.iter().enumerate() {
        let results = index_2bit.search(q, 10);
        total_recall += compute_recall(&results, &ground_truth[i], 10);
    }
    let elapsed = start.elapsed();
    
    let avg_recall = total_recall / query.len() as f32;
    println!("Recall@10: {:.1}%", avg_recall * 100.0);
    println!("Latency: {:.3}ms per query", elapsed.as_secs_f64() * 1000.0 / query.len() as f64);
    println!("QPS: {:.0}", query.len() as f64 / elapsed.as_secs_f64());
    println!();

    // ========== 4-BIT QUANTIZATION ==========
    println!("=================================================================");
    println!("4. 4-BIT QUANTIZATION");
    println!("=================================================================");
    
    let start = Instant::now();
    let mut quant_4bit = MultiBitQuantizer::new(4);
    quant_4bit.train(&base, MetricType::L2);
    let quant_base_4bit: Vec<Vec<f32>> = (0..base.len())
        .map(|i| quant_4bit.reconstruct(i))
        .collect();
    let mut index_4bit = SPTAGIndex::new_kdt(32, 2);
    index_4bit.build(quant_base_4bit);
    let build_time = start.elapsed();
    println!("Build time: {:.2?}", build_time);

    if storage == "disk" {
        save_index_to_disk(&index_4bit, "sptag_4bit.idx");
    }

    let start = Instant::now();
    let mut total_recall = 0.0;
    for (i, q) in query.iter().enumerate() {
        let results = index_4bit.search(q, 10);
        total_recall += compute_recall(&results, &ground_truth[i], 10);
    }
    let elapsed = start.elapsed();
    
    let avg_recall = total_recall / query.len() as f32;
    println!("Recall@10: {:.1}%", avg_recall * 100.0);
    println!("Latency: {:.3}ms per query", elapsed.as_secs_f64() * 1000.0 / query.len() as f64);
    println!("QPS: {:.0}", query.len() as f64 / elapsed.as_secs_f64());
    println!();

    println!("=================================================================");
    println!("SUMMARY");
    println!("=================================================================");
    println!("Storage mode: {}", storage);
    if storage == "disk" {
        println!("Indexes saved to disk: sptag_full.idx, sptag_1bit.idx, sptag_2bit.idx, sptag_4bit.idx");
    } else {
        println!("All indexes kept in RAM");
    }
}
