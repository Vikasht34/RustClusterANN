/// Benchmark balanced k-means on SIFT 1M
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
    println!("=================================================================");
    println!("Balanced K-means on SIFT 1M");
    println!("=================================================================\n");

    println!("Loading 1,000,000 vectors from SIFT...");
    let start = Instant::now();
    let data = read_fvecs("/Users/viktari/pysptag/data/sift/sift_base.fvecs", 1_000_000);
    println!("Loaded {} vectors, dim={} in {:.2?}\n", data.len(), data[0].len(), start.elapsed());

    // SPTAG uses K = sqrt(N)
    let k = (data.len() as f32).sqrt() as usize;
    println!("Using K={} clusters (sqrt of N)\n", k);

    println!("Running balanced k-means...");
    let start = Instant::now();
    let mut kmeans = BalancedKMeans::new(k, data[0].len());
    let obj = kmeans.fit(&data, 100);
    let elapsed = start.elapsed();
    
    let (max, min, avg, std_ratio) = kmeans.stats();
    
    println!("\n=================================================================");
    println!("Results");
    println!("=================================================================");
    println!("Time: {:.2?}", elapsed);
    println!("Objective: {:.2}", obj);
    println!("Cluster sizes:");
    println!("  Max: {}", max);
    println!("  Min: {}", min);
    println!("  Avg: {:.1}", avg);
    println!("  Std/Avg: {:.3} (balanced clustering)", std_ratio);
    println!("\nNon-zero clusters: {}/{}", 
             kmeans.counts.iter().filter(|&&c| c > 0).count(), 
             k);
}
