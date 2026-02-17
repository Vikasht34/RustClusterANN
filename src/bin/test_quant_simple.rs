use rustsptag::multibit::{MultiBitQuantizer, MetricType};

fn main() {
    println!("=== Testing RaBitQ Quantization ===\n");
    
    // Simple 2D test case
    let dim = 2;
    let centroid = vec![0.0, 0.0];
    
    // Test vectors at different positions
    let test_vecs = vec![
        vec![1.0, 0.0],   // Right
        vec![0.0, 1.0],   // Up
        vec![-1.0, 0.0],  // Left
        vec![0.0, -1.0],  // Down
        vec![0.7, 0.7],   // Diagonal
    ];
    
    let query = vec![0.9, 0.1];  // Should be closest to [1.0, 0.0]
    
    for bits in [2, 3, 5] {  // 1-bit, 2-bit, 4-bit
        println!("Testing {}-bit quantization:", bits - 1);
        
        let mut quantizer = MultiBitQuantizer::new(bits);
        quantizer.fit(&test_vecs, &centroid, MetricType::L2);
        
        // Debug: print quantization parameters
        for i in 0..test_vecs.len() {
            quantizer.debug_params(i);
        }
        
        let distances = quantizer.compute_distances(&query, MetricType::L2);
        
        // Compute true L2 distances
        let mut true_dists = Vec::new();
        for vec in &test_vecs {
            let dist: f32 = query.iter().zip(vec.iter())
                .map(|(q, v)| (q - v) * (q - v))
                .sum::<f32>()
                .sqrt();
            true_dists.push(dist);
        }
        
        println!("  Vector | True Dist | Quant Dist | Error");
        println!("  -------|-----------|------------|-------");
        for i in 0..test_vecs.len() {
            let error = (distances[i] - true_dists[i] * true_dists[i]).abs();  // Compare squared distances
            println!("  {:6?} | {:9.4} | {:10.4} | {:.4}", 
                test_vecs[i], true_dists[i], distances[i], error);
        }
        
        // Check if ranking is preserved
        let true_order: Vec<_> = (0..test_vecs.len())
            .map(|i| (i, true_dists[i]))
            .collect::<Vec<_>>();
        let mut true_sorted = true_order.clone();
        true_sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        let quant_order: Vec<_> = (0..test_vecs.len())
            .map(|i| (i, distances[i]))
            .collect::<Vec<_>>();
        let mut quant_sorted = quant_order.clone();
        quant_sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        println!("  True order:  {:?}", true_sorted.iter().map(|(i, _)| i).collect::<Vec<_>>());
        println!("  Quant order: {:?}", quant_sorted.iter().map(|(i, _)| i).collect::<Vec<_>>());
        
        let rank_match = true_sorted[0].0 == quant_sorted[0].0;
        println!("  Top-1 match: {}\n", if rank_match { "✓" } else { "✗" });
    }
}
