use rustsptag::spann::SPANNIndex;
use rustsptag::simd;
use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    let n = if args.len() > 1 {
        args[1].parse().unwrap_or(10000)
    } else {
        10000
    };

    println!("=== SPANN Benchmark (SPTAG defaults) ===");
    println!("Dataset size: {}", n);
    println!();

    // Generate random data
    println!("Generating {} random 128D vectors...", n);
    let data: Vec<Vec<f32>> = (0..n)
        .map(|_| (0..128).map(|_| rand::random::<f32>()).collect())
        .collect();

    // Build SPANN index
    println!("\nBuilding SPANN index...");
    let start = Instant::now();
    let mut index = SPANNIndex::new();
    index.build(data.clone());  // Clone so we can use data later
    let build_time = start.elapsed();
    println!("\nBuild time: {:.2}s", build_time.as_secs_f32());

    // Generate queries
    let num_queries = 100;
    let queries: Vec<Vec<f32>> = (0..num_queries)
        .map(|_| (0..128).map(|_| rand::random::<f32>()).collect())
        .collect();

    // Search
    println!("\nSearching {} queries...", num_queries);
    let start = Instant::now();
    let mut total_results = 0;
    for query in &queries {
        let results = index.search(query, 10);
        total_results += results.len();
    }
    let search_time = start.elapsed();
    let qps = num_queries as f32 / search_time.as_secs_f32();
    
    println!("Search time: {:.3}s", search_time.as_secs_f32());
    println!("QPS: {:.0}", qps);
    println!("Avg results per query: {:.1}", total_results as f32 / num_queries as f32);

    // Compute recall vs brute force
    println!("\nComputing recall vs brute force...");
    let query = &queries[0];
    
    // SPANN search
    let spann_results = index.search(query, 10);
    let spann_ids: Vec<usize> = spann_results.iter().map(|(id, _)| *id).collect();
    
    // Brute force
    let mut brute_force: Vec<(usize, f32)> = (0..index.len())
        .map(|i| (i, simd::l2_distance(query, &data[i])))
        .collect();
    brute_force.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let brute_force_ids: Vec<usize> = brute_force.iter().take(10).map(|(id, _)| *id).collect();
    
    let matches = spann_ids.iter()
        .filter(|id| brute_force_ids.contains(id))
        .count();
    
    println!("Recall@10: {:.1}% ({}/10)", matches as f32 / 10.0 * 100.0, matches);
    
    println!("\n=== Summary ===");
    println!("Index size: {} vectors", index.len());
    println!("Build time: {:.2}s", build_time.as_secs_f32());
    println!("QPS: {:.0}", qps);
    println!("Recall@10: {:.1}%", matches as f32 / 10.0 * 100.0);
}
