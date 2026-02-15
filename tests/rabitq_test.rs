use rustsptag::lib::{OneBitQuantizer, MultiBitQuantizer, MultiBitMetricType};
use rustsptag::MetricType;

fn generate_random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hash, Hasher};
    
    let mut vectors = Vec::with_capacity(n);
    for i in 0..n {
        let mut vec = Vec::with_capacity(dim);
        for j in 0..dim {
            let mut hasher = RandomState::new().build_hasher();
            (seed, i, j).hash(&mut hasher);
            let val = (hasher.finish() % 10000) as f32 / 10000.0;
            vec.push(val);
        }
        vectors.push(vec);
    }
    vectors
}

fn compute_exact_distances(query: &[f32], data: &[Vec<f32>], metric: MetricType) -> Vec<f32> {
    data.iter().map(|vec| {
        match metric {
            MetricType::IP => -query.iter().zip(vec.iter()).map(|(q, v)| q * v).sum::<f32>(),
            MetricType::L2 => query.iter().zip(vec.iter()).map(|(q, v)| (q - v).powi(2)).sum::<f32>(),
        }
    }).collect()
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

#[test]
fn test_onebit_quantizer_basic() {
    let data = vec![
        vec![1.0, 0.0, 1.0, 0.0],
        vec![0.0, 1.0, 0.0, 1.0],
        vec![1.0, 1.0, 0.0, 0.0],
    ];
    
    let centroid = vec![0.67, 0.67, 0.33, 0.33];
    let mut quantizer = OneBitQuantizer::new(4);
    quantizer.fit(&data, &centroid, MetricType::IP);
    
    assert_eq!(quantizer.n_vectors, 3);
    assert_eq!(quantizer.dim, 4);
    assert_eq!(quantizer.binary_codes.len(), 12);
}

#[test]
fn test_onebit_quantizer_recall() {
    let data = generate_random_vectors(1000, 128, 42);
    let queries = generate_random_vectors(10, 128, 99);
    
    let centroid: Vec<f32> = (0..128).map(|i| {
        data.iter().map(|v| v[i]).sum::<f32>() / data.len() as f32
    }).collect();
    
    let mut quantizer = OneBitQuantizer::new(128);
    quantizer.fit(&data, &centroid, MetricType::IP);
    
    let mut total_recall = 0.0;
    let mut total_corr = 0.0;
    
    for query in &queries {
        let exact = compute_exact_distances(query, &data, MetricType::IP);
        let approx = quantizer.compute_distances(query, MetricType::IP);
        
        total_recall += recall_at_k(&exact, &approx, 10);
        total_corr += correlation(&exact, &approx);
    }
    
    let avg_recall = total_recall / queries.len() as f32;
    let avg_corr = total_corr / queries.len() as f32;
    
    assert!(avg_recall >= 0.3, "1-bit recall too low: {}", avg_recall);
    assert!(avg_corr >= 0.75, "1-bit correlation too low: {}", avg_corr);
}

#[test]
fn test_multibit_2bit_recall() {
    let data = generate_random_vectors(1000, 128, 42);
    let queries = generate_random_vectors(10, 128, 99);
    
    let centroid: Vec<f32> = (0..128).map(|i| {
        data.iter().map(|v| v[i]).sum::<f32>() / data.len() as f32
    }).collect();
    
    let mut quantizer = MultiBitQuantizer::new(128, 2);
    quantizer.fit(&data, &centroid, rustsptag::multibit::MetricType::IP);
    
    let mut total_recall = 0.0;
    let mut total_corr = 0.0;
    
    for query in &queries {
        let exact = compute_exact_distances(query, &data, MetricType::IP);
        let approx = quantizer.compute_distances(query, MultiBitMetricType::IP);
        
        total_recall += recall_at_k(&exact, &approx, 10);
        total_corr += correlation(&exact, &approx);
    }
    
    let avg_recall = total_recall / queries.len() as f32;
    let avg_corr = total_corr / queries.len() as f32;
    
    assert!(avg_recall >= 0.35, "2-bit recall too low: {}", avg_recall);
    assert!(avg_corr >= 0.85, "2-bit correlation too low: {}", avg_corr);
}

#[test]
fn test_multibit_4bit_recall() {
    let data = generate_random_vectors(1000, 128, 42);
    let queries = generate_random_vectors(10, 128, 99);
    
    let centroid: Vec<f32> = (0..128).map(|i| {
        data.iter().map(|v| v[i]).sum::<f32>() / data.len() as f32
    }).collect();
    
    let mut quantizer = MultiBitQuantizer::new(128, 4);
    quantizer.fit(&data, &centroid, rustsptag::multibit::MetricType::IP);
    
    let mut total_recall = 0.0;
    let mut total_corr = 0.0;
    
    for query in &queries {
        let exact = compute_exact_distances(query, &data, MetricType::IP);
        let approx = quantizer.compute_distances(query, MultiBitMetricType::IP);
        
        total_recall += recall_at_k(&exact, &approx, 10);
        total_corr += correlation(&exact, &approx);
    }
    
    let avg_recall = total_recall / queries.len() as f32;
    let avg_corr = total_corr / queries.len() as f32;
    
    assert!(avg_recall >= 0.7, "4-bit recall too low: {}", avg_recall);
    assert!(avg_corr >= 0.98, "4-bit correlation too low: {}", avg_corr);
}

#[test]
fn test_multibit_better_than_onebit() {
    let data = generate_random_vectors(500, 64, 42);
    let queries = generate_random_vectors(5, 64, 99);
    
    let centroid: Vec<f32> = (0..64).map(|i| {
        data.iter().map(|v| v[i]).sum::<f32>() / data.len() as f32
    }).collect();
    
    let mut onebit = OneBitQuantizer::new(64);
    onebit.fit(&data, &centroid, MetricType::IP);
    
    let mut multibit = MultiBitQuantizer::new(64, 4);
    multibit.fit(&data, &centroid, rustsptag::multibit::MetricType::IP);
    
    let mut onebit_corr = 0.0;
    let mut multibit_corr = 0.0;
    
    for query in &queries {
        let exact = compute_exact_distances(query, &data, MetricType::IP);
        let onebit_approx = onebit.compute_distances(query, MetricType::IP);
        let multibit_approx = multibit.compute_distances(query, MultiBitMetricType::IP);
        
        onebit_corr += correlation(&exact, &onebit_approx);
        multibit_corr += correlation(&exact, &multibit_approx);
    }
    
    onebit_corr /= queries.len() as f32;
    multibit_corr /= queries.len() as f32;
    
    assert!(multibit_corr > onebit_corr, 
        "4-bit ({:.4}) should be better than 1-bit ({:.4})", multibit_corr, onebit_corr);
}

#[test]
fn test_quantizer_deterministic() {
    let data = generate_random_vectors(100, 32, 42);
    let query = &data[0];
    
    let centroid: Vec<f32> = (0..32).map(|i| {
        data.iter().map(|v| v[i]).sum::<f32>() / data.len() as f32
    }).collect();
    
    let mut q1 = OneBitQuantizer::new(32);
    q1.fit(&data, &centroid, MetricType::IP);
    let dist1 = q1.compute_distances(query, MetricType::IP);
    
    let mut q2 = OneBitQuantizer::new(32);
    q2.fit(&data, &centroid, MetricType::IP);
    let dist2 = q2.compute_distances(query, MetricType::IP);
    
    assert_eq!(dist1, dist2, "Quantizer should be deterministic");
}
