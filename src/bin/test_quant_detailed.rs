use rustsptag::multibit::{MultiBitQuantizer, MetricType};

fn main() {
    println!("=== Detailed RaBitQ Step-by-Step Test ===\n");
    
    // Simple 2D test
    let dim = 2;
    let data = vec![1.0, 0.5];
    let centroid = vec![0.0, 0.0];
    let query = vec![0.9, 0.4];
    
    println!("Setup:");
    println!("  data: {:?}", data);
    println!("  centroid: {:?}", centroid);
    println!("  query: {:?}", query);
    
    // Compute exact L2 distance
    let exact_l2_sq: f32 = data.iter().zip(query.iter())
        .map(|(d, q)| (d - q) * (d - q))
        .sum();
    println!("\nExact L2^2 distance: {:.6}", exact_l2_sq);
    
    for bits in [2, 3, 5] {
        let ex_bits = bits - 1;
        println!("\n=== Testing {}-bit ({} ex_bits) ===", bits, ex_bits);
        
        // Step 1: Compute residual
        let residual: Vec<f32> = data.iter().zip(centroid.iter())
            .map(|(d, c)| d - c)
            .collect();
        println!("1. Residual: {:?}", residual);
        
        // Step 2: Normalize residual
        let norm: f32 = residual.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized: Vec<f32> = residual.iter().map(|x| x / norm).collect();
        println!("2. Norm: {:.6}, Normalized: {:?}", norm, normalized);
        
        // Step 3: Quantize normalized residual
        let abs_res: Vec<f32> = normalized.iter().map(|x| x.abs()).collect();
        println!("3. Abs residual: {:?}", abs_res);
        
        // Step 4: Compute codes (simplified - just show the concept)
        let mut ex_code = vec![0u8; dim];
        let mut signs = vec![0i8; dim];
        for i in 0..dim {
            // Simplified quantization
            let code_val = (abs_res[i] * ((1 << ex_bits) - 1) as f32) as u8;
            ex_code[i] = if normalized[i] < 0.0 {
                ((1 << ex_bits) - 1 - code_val as i32) as u8
            } else {
                code_val
            };
            signs[i] = if normalized[i] >= 0.0 { 1 } else { -1 };
        }
        println!("4. ex_code: {:?}, signs: {:?}", ex_code, signs);
        
        // Step 5: Compute total_code
        let mut total_code = vec![0i32; dim];
        for i in 0..dim {
            total_code[i] = ex_code[i] as i32;
            if residual[i] >= 0.0 {
                total_code[i] += 1 << ex_bits;
            }
        }
        println!("5. total_code: {:?}", total_code);
        
        // Step 6: Center codes
        let cb = -((1 << ex_bits) as f32 - 0.5);
        let xu_cb: Vec<f32> = total_code.iter().map(|&tc| tc as f32 + cb).collect();
        println!("6. cb: {:.2}, xu_cb: {:?}", cb, xu_cb);
        
        // Step 7: Compute factors
        let l2_sqr: f32 = residual.iter().map(|x| x * x).sum();
        let l2_norm = l2_sqr.sqrt();
        let ip_resi_xucb: f32 = residual.iter().zip(xu_cb.iter()).map(|(r, x)| r * x).sum();
        let ip_cent_xucb: f32 = centroid.iter().zip(xu_cb.iter()).map(|(c, x)| c * x).sum();
        
        println!("7. l2_sqr: {:.6}, l2_norm: {:.6}", l2_sqr, l2_norm);
        println!("   ip_resi_xucb: {:.6}, ip_cent_xucb: {:.6}", ip_resi_xucb, ip_cent_xucb);
        
        let ip_resi_xucb = if ip_resi_xucb == 0.0 { f32::INFINITY } else { ip_resi_xucb };
        
        // L2 formula
        let f_add_ex = l2_sqr + (2.0 * l2_sqr * ip_cent_xucb / ip_resi_xucb);
        let f_rescale_ex = -2.0 * l2_norm / ip_resi_xucb;  // Simplified: ipnorm_inv = 1/ip_resi_xucb
        
        println!("8. f_add_ex: {:.6}, f_rescale_ex: {:.6}", f_add_ex, f_rescale_ex);
        
        // Step 8: Compute distance for query
        let g_add: f32 = query.iter().map(|x| x * x).sum();
        println!("9. g_add (||query||^2): {:.6}", g_add);
        
        // Compute combined_ip (simplified - without binary codes)
        let mut ip_exbits_raw = 0.0f32;
        for d in 0..dim {
            let tc = total_code[d] as f32;
            ip_exbits_raw += query[d] * tc;
        }
        println!("10. ip_exbits_raw: {:.6}", ip_exbits_raw);
        
        let combined_ip = ip_exbits_raw + cb * query.iter().sum::<f32>();
        println!("11. combined_ip: {:.6}", combined_ip);
        
        let approx_dist = f_add_ex + g_add + f_rescale_ex * combined_ip;
        println!("12. Approx L2^2: {:.6}", approx_dist);
        println!("    Error: {:.6} ({:.2}%)", 
            approx_dist - exact_l2_sq,
            (approx_dist - exact_l2_sq) / exact_l2_sq * 100.0);
        
        // Now test with our implementation
        println!("\n--- Our Implementation ---");
        let mut quantizer = MultiBitQuantizer::new(bits);
        quantizer.fit(&vec![data.clone()], &centroid, MetricType::L2);
        quantizer.debug_params(0);
        quantizer.debug_codes(0);
        let distances = quantizer.compute_distances(&query, MetricType::L2);
        println!("Our result: {:.6}", distances[0]);
        println!("Difference from manual: {:.6}", distances[0] - approx_dist);
    }
}
