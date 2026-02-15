/// Benchmark recursive BK-Tree on SIFT 1M
use rustsptag::index::BKTreeBuilder;
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
    println!("Recursive BK-Tree Benchmark (SPTAG Algorithm)");
    println!("=================================================================\n");

    let args: Vec<String> = std::env::args().collect();
    let n = if args.len() > 1 {
        args[1].parse().unwrap_or(1_000_000)
    } else {
        1_000_000
    };

    println!("Loading {} vectors from SIFT...", n);
    let start = Instant::now();
    let data = read_fvecs("/Users/viktari/pysptag/data/sift/sift_base.fvecs", n);
    println!("Loaded {} vectors, dim={} in {:.2?}\n", data.len(), data[0].len(), start.elapsed());

    println!("Building recursive BK-Tree...");
    println!("Parameters: K=32, leaf_size=8, samples=1000\n");
    
    let start = Instant::now();
    let mut builder = BKTreeBuilder::new();
    let nodes = builder.build(&data);
    let elapsed = start.elapsed();
    
    println!("\n=================================================================");
    println!("Results");
    println!("=================================================================");
    println!("Build time: {:.2?}", elapsed);
    println!("Tree nodes: {}", nodes.len());
    println!("Vectors: {}", data.len());
    println!("Nodes per vector: {:.2}", nodes.len() as f32 / data.len() as f32);
}
