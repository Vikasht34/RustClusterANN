use rustsptag::index::SPTAGBKTIndex;
use rustsptag::recall::{load_ground_truth, calculate_average_recall};
use std::env;
use std::fs::File;
use std::io::{BufReader, Read};

fn load_sift(path: &str, n: usize) -> Vec<Vec<f32>> {
    let file = File::open(path).expect("Failed to open SIFT file");
    let mut reader = BufReader::new(file);
    
    let mut data = Vec::with_capacity(n);
    let mut buffer = [0u8; 4];
    
    for _ in 0..n {
        // Read dimension (should be 128)
        reader.read_exact(&mut buffer).expect("Failed to read dimension");
        let dim = u32::from_le_bytes(buffer) as usize;
        
        // Read vector
        let mut vec = vec![0f32; dim];
        for j in 0..dim {
            reader.read_exact(&mut buffer).expect("Failed to read value");
            vec[j] = u32::from_le_bytes(buffer) as f32;
        }
        
        data.push(vec);
    }
    
    data
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let n = if args.len() > 1 {
        args[1].parse().unwrap_or(10000)
    } else {
        10000
    };
    
    println!("=================================================================");
    println!("SPTAG-BKT Benchmark (Tree + RNG + NPA)");
    println!("=================================================================");
    println!("Loading {} SIFT vectors...", n);
    
    let data = load_sift("/Users/viktari/pysptag/data/sift/sift_base.fvecs", n);
    println!("Loaded {} vectors of dimension {}", data.len(), data[0].len());
    
    // Build index
    println!("\n=================================================================");
    println!("Building Index");
    println!("=================================================================");
    println!("Parameters:");
    println!("  K (clusters per level): 32");
    println!("  Leaf size: 2000");
    println!("  Max degree: 32");
    println!("  RNG factor: 1.0");
    println!("  CEF (candidate expansion): 500");
    println!("  Refine iterations: 2");
    println!("  Number of trees: 32");
    
    let mut index = SPTAGBKTIndex::new(
        32,    // K
        2000,  // leaf_size (SPTAG default)
        32,    // max_degree
        1.0,   // rng_factor
        500,   // CEF (SPTAG default)
        32,    // num_trees (SPTAG default)
    );
    
    let build_start = std::time::Instant::now();
    index.build(data, 2); // Enable NPA to debug
    let build_time = build_start.elapsed();
    
    println!("\n=================================================================");
    println!("Build Complete");
    println!("=================================================================");
    println!("Total build time: {:.2}s", build_time.as_secs_f32());
    
    // Search benchmark
    println!("\n=================================================================");
    println!("Search Benchmark");
    println!("=================================================================");
    
    let queries = load_sift("/Users/viktari/pysptag/data/sift/sift_query.fvecs", 100);
    println!("Loaded {} query vectors", queries.len());
    
    // Load ground truth
    let ground_truth = load_ground_truth("/Users/viktari/pysptag/data/sift/sift_groundtruth.ivecs", 100);
    println!("Loaded ground truth for {} queries", ground_truth.len());
    
    let k = 10;
    let max_checks = 8192;
    
    let search_start = std::time::Instant::now();
    let mut all_results = Vec::new();
    
    // Debug first query
    let first_result = index.search(&queries[0], k, max_checks);
    eprintln!("First query result IDs: {:?}", first_result.iter().map(|(id, _)| id).take(10).collect::<Vec<_>>());
    eprintln!("First query ground truth: {:?}", &ground_truth[0][..10]);
    
    // Brute force check - find actual nearest neighbors in our dataset
    let mut brute_force: Vec<(usize, f32)> = (0..index.get_data().len())
        .map(|i| (i, rustsptag::simd::l2_distance(&queries[0], &index.get_data()[i])))
        .collect();
    brute_force.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let bf_ids: Vec<usize> = brute_force.iter().take(10).map(|(id, _)| *id).collect();
    let graph_ids: Vec<usize> = first_result.iter().take(10).map(|(id, _)| *id).collect();
    
    let mut matches = 0;
    for id in &graph_ids {
        if bf_ids.contains(id) {
            matches += 1;
        }
    }
    
    eprintln!("First query result IDs: {:?}", &graph_ids);
    eprintln!("Brute force top 10: {:?}", &bf_ids);
    eprintln!("Matches: {}/10 = {:.1}% recall", matches, matches as f32 / 10.0 * 100.0);
    
    all_results.push(first_result);
    
    for query in &queries[1..] {
        let results = index.search(query, k, max_checks);
        all_results.push(results);
    }
    
    let search_time = search_start.elapsed();
    let qps = queries.len() as f64 / search_time.as_secs_f64();
    
    // Calculate recall
    let recall = calculate_average_recall(&all_results, &ground_truth, k);
    
    println!("Queries: {}", queries.len());
    println!("K: {}", k);
    println!("Max checks: {}", max_checks);
    println!("Search time: {:.3}s", search_time.as_secs_f64());
    println!("QPS: {:.2}", qps);
    println!("Recall@{}: {:.4}", k, recall);
    
    println!("\n=================================================================");
    println!("Summary");
    println!("=================================================================");
    println!("Vectors: {}", n);
    println!("Build time: {:.2}s", build_time.as_secs_f32());
    println!("QPS: {:.2}", qps);
    println!("Recall@{}: {:.4}", k, recall);
}
