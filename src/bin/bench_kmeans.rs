/// Benchmark balanced k-means clustering
use rustsptag::index::BalancedKMeans;
use std::fs::File;
use std::io::{BufReader, Read};
use std::time::Instant;

fn read_fvecs(path: &str, limit: usize) -> Vec<Vec<f32>> {
    let file = File::open(path).expect("Failed to open file");
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();

    loop {
        if vectors.len() >= limit {
            break;
        }
        
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

    println!("=================================================================");
    println!("Balanced K-means Benchmark");
    println!("=================================================================\n");

    println!("Loading {} vectors from SIFT...", n);
    let data = read_fvecs("/Users/viktari/pysptag/data/sift/sift_base.fvecs", n);
    println!("Loaded {} vectors, dim={}\n", data.len(), data[0].len());

    // SPTAG rule: K = sqrt(N) for balanced clustering
    let k = (data.len() as f32).sqrt() as usize;
    println!("Using K={} clusters (sqrt of N)\n", k);

    // Test 1: Standard k-means (lambda=0)
    println!("=================================================================");
    println!("1. Standard K-means (lambda=0, no balancing)");
    println!("=================================================================");
    
    let start = Instant::now();
    let mut kmeans_std = BalancedKMeans::new(k, data[0].len());
    let obj = kmeans_std.fit_with_lambda(&data, 0.0, 100);
    let elapsed = start.elapsed();
    
    let (max, min, avg, std_ratio) = kmeans_std.stats();
    println!("Time: {:.2?}", elapsed);
    println!("Objective: {:.2}", obj);
    println!("Cluster sizes:");
    println!("  Max: {}", max);
    println!("  Min: {}", min);
    println!("  Avg: {:.1}", avg);
    println!("  Std/Avg: {:.3} (higher = more imbalanced)\n", std_ratio);

    // Test 2: Balanced k-means with auto lambda
    println!("=================================================================");
    println!("2. Balanced K-means (auto lambda selection)");
    println!("=================================================================");
    
    let start = Instant::now();
    let mut kmeans_balanced = BalancedKMeans::new(k, data[0].len());
    let obj = kmeans_balanced.fit(&data, 100);
    let elapsed = start.elapsed();
    
    let (max, min, avg, std_ratio) = kmeans_balanced.stats();
    println!("Time: {:.2?}", elapsed);
    println!("Objective: {:.2}", obj);
    println!("Cluster sizes:");
    println!("  Max: {}", max);
    println!("  Min: {}", min);
    println!("  Avg: {:.1}", avg);
    println!("  Std/Avg: {:.3} (lower = more balanced)\n", std_ratio);

    // Test 3: Different dataset sizes
    println!("=================================================================");
    println!("3. Scaling Analysis (SIFT dataset)");
    println!("=================================================================");
    
    for &size in &[1000, 10000, 100000, 1000000] {
        let test_data = if size <= data.len() {
            data.iter().take(size).cloned().collect()
        } else {
            println!("\nLoading {} vectors from SIFT...", size);
            read_fvecs("/Users/viktari/pysptag/data/sift/sift_base.fvecs", size)
        };
        
        let k = (test_data.len() as f32).sqrt() as usize;
        
        let start = Instant::now();
        let mut kmeans = BalancedKMeans::new(k, test_data[0].len());
        kmeans.fit(&test_data, 100);
        let elapsed = start.elapsed();
        
        let (max, min, avg, std_ratio) = kmeans.stats();
        println!("N={:7}, K={:4}, Time={:7.2?}, Max={:5}, Min={:4}, Avg={:6.1}, Std/Avg={:.3}", 
                 size, k, elapsed, max, min, avg, std_ratio);
    }

    println!("\n=================================================================");
    println!("Summary");
    println!("=================================================================");
    println!("Balanced k-means produces more uniform cluster sizes");
    println!("Lambda factor automatically selected for optimal balance");
    println!("SPTAG uses K=sqrt(N) for hierarchical clustering");
}
