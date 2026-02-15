use rustsptag::lib::{OneBitQuantizer, MultiBitQuantizer, MultiBitMetricType};
use rustsptag::{MetricType, multibit::MetricType as MBMetricType};
use std::fs::File;
use std::io::{BufReader, Read};
use std::time::Instant;

fn load_fvecs(path: &str, max_vectors: Option<usize>) -> std::io::Result<Vec<Vec<f32>>> {
    let file = File::open(path)?;
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
        reader.read_exact(&mut bytes)?;
        
        for i in 0..dim {
            vec[i] = f32::from_le_bytes([
                bytes[i * 4],
                bytes[i * 4 + 1],
                bytes[i * 4 + 2],
                bytes[i * 4 + 3],
            ]);
        }
        
        vectors.push(vec);
        if let Some(max) = max_vectors {
            if vectors.len() >= max {
                break;
            }
        }
    }
    
    Ok(vectors)
}

fn recall_at_k(exact: &[f32], approx: &[f32], k: usize) -> f32 {
    let mut exact_idx: Vec<usize> = (0..exact.len()).collect();
    exact_idx.sort_by(|&a, &b| exact[a].partial_cmp(&exact[b]).unwrap());
    
    let mut approx_idx: Vec<usize> = (0..approx.len()).collect();
    approx_idx.sort_by(|&a, &b| approx[a].partial_cmp(&approx[b]).unwrap());
    
    let exact_top_k: std::collections::HashSet<_> = exact_idx.iter().take(k).collect();
    let approx_top_k: Vec<_> = approx_idx.iter().take(k).collect();
    
    approx_top_k.iter().filter(|&&idx| exact_top_k.contains(idx)).count() as f32 / k as f32
}

fn correlation(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len() as f32;
    let mean_x: f32 = x.iter().sum::<f32>() / n;
    let mean_y: f32 = y.iter().sum::<f32>() / n;
    
    let cov: f32 = x.iter().zip(y.iter()).map(|(xi, yi)| (xi - mean_x) * (yi - mean_y)).sum();
    let std_x: f32 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum::<f32>().sqrt();
    let std_y: f32 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum::<f32>().sqrt();
    
    cov / (std_x * std_y)
}

fn compute_l2_distances(query: &[f32], data: &[Vec<f32>]) -> Vec<f32> {
    data.iter().map(|vec| {
        query.iter().zip(vec.iter()).map(|(q, v)| (q - v).powi(2)).sum()
    }).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_base = if args.len() > 1 { args[1].parse().unwrap_or(100000) } else { 100000 };
    
    println!("=================================================================");
    println!("SIFT Dataset Benchmark");
    println!("=================================================================\n");
    
    println!("Loading SIFT dataset...");
    let base = load_fvecs("/Users/viktari/pysptag/data/sift/sift_base.fvecs", Some(n_base))
        .expect("Failed to load SIFT base");
    let queries = load_fvecs("/Users/viktari/pysptag/data/sift/sift_query.fvecs", Some(100))
        .expect("Failed to load SIFT queries");
    
    println!("Loaded {} base vectors, {} queries, dim={}\n", base.len(), queries.len(), base[0].len());
    
    let dim = base[0].len();
    let centroid: Vec<f32> = (0..dim).map(|i| {
        base.iter().map(|v| v[i]).sum::<f32>() / base.len() as f32
    }).collect();
    
    // 1-bit
    println!("=================================================================");
    println!("1-bit Quantization");
    println!("=================================================================");
    
    let start = Instant::now();
    let mut onebit = OneBitQuantizer::new(dim);
    onebit.fit(&base, &centroid, MetricType::L2);
    println!("Build time: {:.2}ms", start.elapsed().as_secs_f64() * 1000.0);
    
    let mut total_recall = 0.0;
    let mut total_corr = 0.0;
    
    let start = Instant::now();
    for query in &queries {
        let approx = onebit.compute_distances(query, MetricType::L2);
        let exact = compute_l2_distances(query, &base);
        total_recall += recall_at_k(&exact, &approx, 10);
        total_corr += correlation(&exact, &approx);
    }
    let query_time = start.elapsed();
    
    println!("Recall@10: {:.1}%", (total_recall / queries.len() as f32) * 100.0);
    println!("Correlation: {:.4}", total_corr / queries.len() as f32);
    println!("Latency: {:.3}ms per query", query_time.as_secs_f64() * 1000.0 / queries.len() as f64);
    println!("QPS: {:.0}\n", queries.len() as f64 / query_time.as_secs_f64());
    
    // 2-bit
    println!("=================================================================");
    println!("2-bit Quantization");
    println!("=================================================================");
    
    let start = Instant::now();
    let mut twobit = MultiBitQuantizer::new(2);
    twobit.fit(&base, &centroid, MBMetricType::L2);
    println!("Build time: {:.2}ms", start.elapsed().as_secs_f64() * 1000.0);
    
    let mut total_recall = 0.0;
    let mut total_corr = 0.0;
    
    let start = Instant::now();
    for query in &queries {
        let approx = twobit.compute_distances(query, MBMetricType::L2);
        let exact = compute_l2_distances(query, &base);
        total_recall += recall_at_k(&exact, &approx, 10);
        total_corr += correlation(&exact, &approx);
    }
    let query_time = start.elapsed();
    
    println!("Recall@10: {:.1}%", (total_recall / queries.len() as f32) * 100.0);
    println!("Correlation: {:.4}", total_corr / queries.len() as f32);
    println!("Latency: {:.3}ms per query", query_time.as_secs_f64() * 1000.0 / queries.len() as f64);
    println!("QPS: {:.0}\n", queries.len() as f64 / query_time.as_secs_f64());
    
    // 4-bit
    println!("=================================================================");
    println!("4-bit Quantization");
    println!("=================================================================");
    
    let start = Instant::now();
    let mut fourbit = MultiBitQuantizer::new(4);
    fourbit.fit(&base, &centroid, MBMetricType::L2);
    println!("Build time: {:.2}ms", start.elapsed().as_secs_f64() * 1000.0);
    
    let mut total_recall = 0.0;
    let mut total_corr = 0.0;
    
    let start = Instant::now();
    for query in &queries {
        let approx = fourbit.compute_distances(query, MBMetricType::L2);
        let exact = compute_l2_distances(query, &base);
        total_recall += recall_at_k(&exact, &approx, 10);
        total_corr += correlation(&exact, &approx);
    }
    let query_time = start.elapsed();
    
    println!("Recall@10: {:.1}%", (total_recall / queries.len() as f32) * 100.0);
    println!("Correlation: {:.4}", total_corr / queries.len() as f32);
    println!("Latency: {:.3}ms per query", query_time.as_secs_f64() * 1000.0 / queries.len() as f64);
    println!("QPS: {:.0}\n", queries.len() as f64 / query_time.as_secs_f64());
}
