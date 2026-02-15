/// Benchmark SPTAG index on SIFT dataset
use rustsptag::index::{SPTAGIndex, IndexType};
use std::fs::File;
use std::io::{BufReader, Read};
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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n = if args.len() > 1 {
        args[1].parse().unwrap_or(10000)
    } else {
        10000
    };

    println!("Loading SIFT dataset (first {} vectors)...", n);
    let mut base = read_fvecs("/Users/viktari/pysptag/data/sift/sift_base.fvecs");
    base.truncate(n);
    
    let mut query = read_fvecs("/Users/viktari/pysptag/data/sift/sift_query.fvecs");
    query.truncate(100);

    println!("Dataset: {} base vectors, {} queries, dim={}", 
             base.len(), query.len(), base[0].len());

    // Test KDT+Graph
    println!("\n=================================================================");
    println!("KDT + Graph Index");
    println!("=================================================================");
    
    let start = Instant::now();
    let mut index_kdt = SPTAGIndex::new_kdt(32, 2);
    index_kdt.build(base.clone());
    println!("Build time: {:.2?}", start.elapsed());

    let start = Instant::now();
    let mut total_recall = 0.0;
    for q in &query {
        let results = index_kdt.search(q, 10);
        // Simple recall check - just count results
        total_recall += results.len() as f32 / 10.0;
    }
    let elapsed = start.elapsed();
    
    println!("Avg recall: {:.1}%", total_recall / query.len() as f32 * 100.0);
    println!("Latency: {:.3}ms per query", elapsed.as_secs_f64() * 1000.0 / query.len() as f64);
    println!("QPS: {:.0}", query.len() as f64 / elapsed.as_secs_f64());

    // Test BKT+Graph
    println!("\n=================================================================");
    println!("BKT + Graph Index");
    println!("=================================================================");
    
    let start = Instant::now();
    let mut index_bkt = SPTAGIndex::new_bkt(32, 2, 8);
    index_bkt.build(base.clone());
    println!("Build time: {:.2?}", start.elapsed());

    let start = Instant::now();
    let mut total_recall = 0.0;
    for q in &query {
        let results = index_bkt.search(q, 10);
        total_recall += results.len() as f32 / 10.0;
    }
    let elapsed = start.elapsed();
    
    println!("Avg recall: {:.1}%", total_recall / query.len() as f32 * 100.0);
    println!("Latency: {:.3}ms per query", elapsed.as_secs_f64() * 1000.0 / query.len() as f64);
    println!("QPS: {:.0}", query.len() as f64 / elapsed.as_secs_f64());
}
