use rustsptag::lib::{OneBitQuantizer, MultiBitQuantizer};
use rustsptag::{MetricType as OneBitMetric};
use rustsptag::multibit::MetricType as MultiBitMetric;
use rand::Rng;

fn test_metric_ip() {
    println!("\n{}", "=".repeat(70));
    println!("Testing IP (Inner Product) metric");
    println!("{}", "=".repeat(70));
    
    let mut rng = rand::thread_rng();
    let dim = 128;
    let n_vectors = 1000;
    
    println!("Generating {} normalized vectors of dimension {}...", n_vectors, dim);
    
    // Generate normalized random vectors for IP
    let mut vectors: Vec<Vec<f32>> = Vec::new();
    for _ in 0..n_vectors {
        let mut vec: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        vec.iter_mut().for_each(|x| *x /= norm);
        vectors.push(vec);
    }
    
    let mut query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
    let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
    query.iter_mut().for_each(|x| *x /= norm);
    
    // Compute centroid
    let mut centroid = vec![0.0f32; dim];
    for vec in &vectors {
        for (i, &val) in vec.iter().enumerate() {
            centroid[i] += val;
        }
    }
    centroid.iter_mut().for_each(|x| *x /= n_vectors as f32);
    
    // Ground truth distances (negative dot product)
    let gt_dists: Vec<f32> = vectors.iter()
        .map(|v| {
            let dot: f32 = v.iter().zip(query.iter()).map(|(a, b)| a * b).sum();
            -dot
        })
        .collect();
    
    // Test 1-bit
    println!("\n1-bit Quantization:");
    let mut quantizer_1bit = OneBitQuantizer::new(dim);
    quantizer_1bit.fit(&vectors, &centroid, OneBitMetric::IP);
    let quant_dists = quantizer_1bit.compute_distances(&query, OneBitMetric::IP);
    print_metrics(&gt_dists, &quant_dists, n_vectors);
    
    // Test 2-bit
    println!("\n2-bit Quantization:");
    let mut quantizer_2bit = MultiBitQuantizer::new(2);
    quantizer_2bit.fit(&vectors, &centroid, MultiBitMetric::IP);
    let quant_dists = quantizer_2bit.compute_distances(&query, MultiBitMetric::IP);
    print_metrics(&gt_dists, &quant_dists, n_vectors);
    
    // Test 4-bit
    println!("\n4-bit Quantization:");
    let mut quantizer_4bit = MultiBitQuantizer::new(4);
    quantizer_4bit.fit(&vectors, &centroid, MultiBitMetric::IP);
    let quant_dists = quantizer_4bit.compute_distances(&query, MultiBitMetric::IP);
    print_metrics(&gt_dists, &quant_dists, n_vectors);
}

fn test_metric_l2() {
    println!("\n{}", "=".repeat(70));
    println!("Testing L2 (Euclidean) metric");
    println!("{}", "=".repeat(70));
    
    let mut rng = rand::thread_rng();
    let dim = 128;
    let n_vectors = 1000;
    
    println!("Generating {} unnormalized vectors of dimension {}...", n_vectors, dim);
    
    // Generate unnormalized random vectors for L2
    let mut vectors: Vec<Vec<f32>> = Vec::new();
    for _ in 0..n_vectors {
        let vec: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 100.0 - 50.0).collect();
        vectors.push(vec);
    }
    
    let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 100.0 - 50.0).collect();
    
    // Compute centroid
    let mut centroid = vec![0.0f32; dim];
    for vec in &vectors {
        for (i, &val) in vec.iter().enumerate() {
            centroid[i] += val;
        }
    }
    centroid.iter_mut().for_each(|x| *x /= n_vectors as f32);
    
    // Ground truth distances (L2 squared)
    let gt_dists: Vec<f32> = vectors.iter()
        .map(|v| {
            v.iter().zip(query.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f32>()
        })
        .collect();
    
    // Test 1-bit
    println!("\n1-bit Quantization:");
    let mut quantizer_1bit = OneBitQuantizer::new(dim);
    quantizer_1bit.fit(&vectors, &centroid, OneBitMetric::L2);
    let quant_dists = quantizer_1bit.compute_distances(&query, OneBitMetric::L2);
    print_metrics(&gt_dists, &quant_dists, n_vectors);
    
    // Test 2-bit
    println!("\n2-bit Quantization:");
    let mut quantizer_2bit = MultiBitQuantizer::new(2);
    quantizer_2bit.fit(&vectors, &centroid, MultiBitMetric::L2);
    let quant_dists = quantizer_2bit.compute_distances(&query, MultiBitMetric::L2);
    print_metrics(&gt_dists, &quant_dists, n_vectors);
    
    // Test 4-bit
    println!("\n4-bit Quantization:");
    let mut quantizer_4bit = MultiBitQuantizer::new(4);
    quantizer_4bit.fit(&vectors, &centroid, MultiBitMetric::L2);
    let quant_dists = quantizer_4bit.compute_distances(&query, MultiBitMetric::L2);
    print_metrics(&gt_dists, &quant_dists, n_vectors);
}

fn print_metrics(gt_dists: &[f32], quant_dists: &[f32], n_vectors: usize) {
    // Recall@10
    let mut gt_indices: Vec<usize> = (0..n_vectors).collect();
    gt_indices.sort_by(|&a, &b| gt_dists[a].partial_cmp(&gt_dists[b]).unwrap());
    let gt_top10: std::collections::HashSet<usize> = gt_indices[..10].iter().cloned().collect();
    
    let mut quant_indices: Vec<usize> = (0..n_vectors).collect();
    quant_indices.sort_by(|&a, &b| quant_dists[a].partial_cmp(&quant_dists[b]).unwrap());
    let quant_top10: std::collections::HashSet<usize> = quant_indices[..10].iter().cloned().collect();
    
    let recall = gt_top10.intersection(&quant_top10).count() as f32 / 10.0;
    
    // Correlation
    let mean_gt: f32 = gt_dists.iter().sum::<f32>() / n_vectors as f32;
    let mean_quant: f32 = quant_dists.iter().sum::<f32>() / n_vectors as f32;
    let mut cov = 0.0f32;
    let mut var_gt = 0.0f32;
    let mut var_quant = 0.0f32;
    for i in 0..n_vectors {
        let dgt = gt_dists[i] - mean_gt;
        let dq = quant_dists[i] - mean_quant;
        cov += dgt * dq;
        var_gt += dgt * dgt;
        var_quant += dq * dq;
    }
    let correlation = cov / (var_gt * var_quant).sqrt();
    
    println!("  Recall@10: {:.1}%", recall * 100.0);
    println!("  Correlation: {:.4}", correlation);
}

fn main() {
    println!("======================================================================");
    println!("RaBitQ Rust Implementation - Synthetic Data Test");
    println!("======================================================================");
    
    test_metric_ip();
    test_metric_l2();
}
