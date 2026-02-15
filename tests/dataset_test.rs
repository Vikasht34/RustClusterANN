use rustsptag::lib::{OneBitQuantizer, MultiBitQuantizer, MultiBitMetricType};
use rustsptag::MetricType;
use std::fs::File;
use std::io::{BufReader, Read};

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

#[test]
#[ignore]
fn test_sift_1m_onebit() {
    let base = load_fvecs("sift/sift_base.fvecs", Some(10000))
        .expect("Download SIFT dataset from http://corpus-texmex.irisa.fr/");
    let queries = load_fvecs("sift/sift_query.fvecs", Some(100))
        .expect("SIFT query file not found");
    
    let dim = base[0].len();
    let centroid: Vec<f32> = (0..dim).map(|i| {
        base.iter().map(|v| v[i]).sum::<f32>() / base.len() as f32
    }).collect();
    
    let mut quantizer = OneBitQuantizer::new(dim);
    quantizer.fit(&base, &centroid, MetricType::L2);
    
    let mut total_recall = 0.0;
    let mut total_corr = 0.0;
    
    for query in queries.iter().take(10) {
        let exact = compute_l2_distances(query, &base);
        let approx = quantizer.compute_distances(query, MetricType::L2);
        
        total_recall += recall_at_k(&exact, &approx, 10);
        total_corr += correlation(&exact, &approx);
    }
    
    let avg_recall = total_recall / 10.0;
    let avg_corr = total_corr / 10.0;
    
    println!("SIFT 1-bit: Recall@10={:.2}%, Correlation={:.4}", avg_recall * 100.0, avg_corr);
    assert!(avg_recall >= 0.25, "SIFT 1-bit recall: {}", avg_recall);
    assert!(avg_corr >= 0.7, "SIFT 1-bit correlation: {}", avg_corr);
}

#[test]
#[ignore]
fn test_sift_1m_multibit() {
    let base = load_fvecs("sift/sift_base.fvecs", Some(10000))
        .expect("Download SIFT dataset from http://corpus-texmex.irisa.fr/");
    let queries = load_fvecs("sift/sift_query.fvecs", Some(100))
        .expect("SIFT query file not found");
    
    let dim = base[0].len();
    let centroid: Vec<f32> = (0..dim).map(|i| {
        base.iter().map(|v| v[i]).sum::<f32>() / base.len() as f32
    }).collect();
    
    let mut quantizer = MultiBitQuantizer::new(dim, 4);
    quantizer.fit(&base, &centroid, rustsptag::multibit::MetricType::IP);
    
    let mut total_recall = 0.0;
    let mut total_corr = 0.0;
    
    for query in queries.iter().take(10) {
        let exact = compute_l2_distances(query, &base);
        let approx = quantizer.compute_distances(query, MultiBitMetricType::L2);
        
        total_recall += recall_at_k(&exact, &approx, 10);
        total_corr += correlation(&exact, &approx);
    }
    
    let avg_recall = total_recall / 10.0;
    let avg_corr = total_corr / 10.0;
    
    println!("SIFT 4-bit: Recall@10={:.2}%, Correlation={:.4}", avg_recall * 100.0, avg_corr);
    assert!(avg_recall >= 0.6, "SIFT 4-bit recall: {}", avg_recall);
    assert!(avg_corr >= 0.95, "SIFT 4-bit correlation: {}", avg_corr);
}

#[test]
#[ignore]
fn test_cohere_embeddings() {
    // Simulate Cohere embed-english-v3.0 (1024-dim, normalized)
    let n_base = 5000;
    let n_queries = 20;
    let dim = 1024;
    
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hash, Hasher};
    
    let mut base = Vec::with_capacity(n_base);
    for i in 0..n_base {
        let mut vec = Vec::with_capacity(dim);
        for j in 0..dim {
            let mut hasher = RandomState::new().build_hasher();
            (42u64, i, j).hash(&mut hasher);
            vec.push((hasher.finish() % 10000) as f32 / 10000.0 - 0.5);
        }
        // Normalize
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut vec {
            *x /= norm;
        }
        base.push(vec);
    }
    
    let mut queries = Vec::with_capacity(n_queries);
    for i in 0..n_queries {
        let mut vec = Vec::with_capacity(dim);
        for j in 0..dim {
            let mut hasher = RandomState::new().build_hasher();
            (99u64, i, j).hash(&mut hasher);
            vec.push((hasher.finish() % 10000) as f32 / 10000.0 - 0.5);
        }
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut vec {
            *x /= norm;
        }
        queries.push(vec);
    }
    
    let centroid: Vec<f32> = (0..dim).map(|i| {
        base.iter().map(|v| v[i]).sum::<f32>() / base.len() as f32
    }).collect();
    
    // Test 4-bit quantization
    let mut quantizer = MultiBitQuantizer::new(dim, 4);
    quantizer.fit(&base, &centroid, rustsptag::multibit::MetricType::IP);
    
    let mut total_recall = 0.0;
    let mut total_corr = 0.0;
    
    for query in &queries {
        let exact: Vec<f32> = base.iter().map(|vec| {
            -query.iter().zip(vec.iter()).map(|(q, v)| q * v).sum::<f32>()
        }).collect();
        let approx = quantizer.compute_distances(query, MultiBitMetricType::IP);
        
        total_recall += recall_at_k(&exact, &approx, 10);
        total_corr += correlation(&exact, &approx);
    }
    
    let avg_recall = total_recall / queries.len() as f32;
    let avg_corr = total_corr / queries.len() as f32;
    
    println!("Cohere 4-bit: Recall@10={:.2}%, Correlation={:.4}", avg_recall * 100.0, avg_corr);
    assert!(avg_recall >= 0.7, "Cohere 4-bit recall: {}", avg_recall);
    assert!(avg_corr >= 0.98, "Cohere 4-bit correlation: {}", avg_corr);
}
