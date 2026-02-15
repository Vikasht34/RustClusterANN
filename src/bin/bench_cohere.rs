use rustsptag::lib::MultiBitQuantizer;
use rustsptag::multibit::MetricType;
use std::time::Instant;

fn load_hdf5_cohere(path: &str, max_train: usize, max_test: usize) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<usize>>) {
    use std::process::Command;
    
    let output = Command::new("python3")
        .arg("-c")
        .arg(format!(r#"
import h5py
import json
f = h5py.File('{}', 'r')
train = f['train'][:{}].tolist()
test = f['test'][:{}].tolist()
neighbors = f['neighbors'][:{}].tolist()
print(json.dumps({{'train': train, 'test': test, 'neighbors': neighbors}}))
"#, path, max_train, max_test, max_test))
        .output()
        .expect("Failed to run Python");
    
    let json_str = String::from_utf8(output.stdout).expect("Invalid UTF-8");
    let data: serde_json::Value = serde_json::from_str(&json_str).expect("Invalid JSON");
    
    let train: Vec<Vec<f32>> = data["train"].as_array().unwrap()
        .iter().map(|v| v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()).collect();
    let test: Vec<Vec<f32>> = data["test"].as_array().unwrap()
        .iter().map(|v| v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()).collect();
    let neighbors: Vec<Vec<usize>> = data["neighbors"].as_array().unwrap()
        .iter().map(|v| v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect()).collect();
    
    (train, test, neighbors)
}

fn recall_at_k(approx: &[f32], ground_truth: &[usize], k: usize) -> f32 {
    let mut approx_idx: Vec<usize> = (0..approx.len()).collect();
    approx_idx.sort_by(|&a, &b| approx[a].partial_cmp(&approx[b]).unwrap());
    
    let gt_set: std::collections::HashSet<_> = ground_truth.iter().take(k).collect();
    let approx_top_k: Vec<_> = approx_idx.iter().take(k).collect();
    
    approx_top_k.iter().filter(|&&idx| gt_set.contains(idx)).count() as f32 / k as f32
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_train = if args.len() > 1 { args[1].parse().unwrap_or(10000) } else { 10000 };
    let n_test = if args.len() > 2 { args[2].parse().unwrap_or(100) } else { 100 };
    
    println!("=================================================================");
    println!("Cohere Embeddings Benchmark (embed-english-v3.0, 768-dim)");
    println!("=================================================================\n");
    
    println!("Loading Cohere embeddings from HDF5...");
    let (mut train, mut test, neighbors) = load_hdf5_cohere(
        "/Users/viktari/pysptag/data/cohere/documents-1m.hdf5",
        n_train,
        n_test
    );
    
    // Normalize vectors for cosine similarity (IP metric)
    for vec in train.iter_mut() {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in vec.iter_mut() {
                *x /= norm;
            }
        }
    }
    for vec in test.iter_mut() {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for x in vec.iter_mut() {
                *x /= norm;
            }
        }
    }
    
    println!("Loaded {} train vectors, {} test queries, dim={}\n", train.len(), test.len(), train[0].len());
    
    let dim = train[0].len();
    let centroid: Vec<f32> = (0..dim).map(|i| {
        train.iter().map(|v| v[i]).sum::<f32>() / train.len() as f32
    }).collect();
    
    // 2-bit
    println!("=================================================================");
    println!("2-bit Quantization");
    println!("=================================================================");
    
    let start = Instant::now();
    let mut twobit = MultiBitQuantizer::new(2);
    twobit.fit(&train, &centroid, MetricType::IP);
    println!("Build time: {:.2}ms", start.elapsed().as_secs_f64() * 1000.0);
    
    let mut total_recall = 0.0;
    
    let start = Instant::now();
    for (query, gt) in test.iter().zip(neighbors.iter()) {
        let approx = twobit.compute_distances(query, MetricType::IP);
        total_recall += recall_at_k(&approx, gt, 10);
    }
    let query_time = start.elapsed();
    
    println!("Recall@10: {:.1}%", (total_recall / test.len() as f32) * 100.0);
    println!("Latency: {:.3}ms per query", query_time.as_secs_f64() * 1000.0 / test.len() as f64);
    println!("QPS: {:.0}\n", test.len() as f64 / query_time.as_secs_f64());
    
    // 4-bit
    println!("=================================================================");
    println!("4-bit Quantization");
    println!("=================================================================");
    
    let start = Instant::now();
    let mut fourbit = MultiBitQuantizer::new(4);
    fourbit.fit(&train, &centroid, MetricType::IP);
    println!("Build time: {:.2}ms", start.elapsed().as_secs_f64() * 1000.0);
    
    let mut total_recall = 0.0;
    
    let start = Instant::now();
    for (query, gt) in test.iter().zip(neighbors.iter()) {
        let approx = fourbit.compute_distances(query, MetricType::IP);
        total_recall += recall_at_k(&approx, gt, 10);
    }
    let query_time = start.elapsed();
    
    println!("Recall@10: {:.1}%", (total_recall / test.len() as f32) * 100.0);
    println!("Latency: {:.3}ms per query", query_time.as_secs_f64() * 1000.0 / test.len() as f64);
    println!("QPS: {:.0}\n", test.len() as f64 / query_time.as_secs_f64());
}
