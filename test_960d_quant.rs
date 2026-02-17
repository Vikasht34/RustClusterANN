use std::time::Instant;

fn main() {
    println!("=== Testing 960D Quantization ===\n");
    
    let dim = 960;
    let n_vectors = 1000;
    
    println!("Creating {} vectors of {}D...", n_vectors, dim);
    let vectors: Vec<Vec<f32>> = (0..n_vectors)
        .map(|i| {
            (0..dim).map(|j| ((i + j) as f32) / (dim as f32)).collect()
        })
        .collect();
    
    let centroid: Vec<f32> = vec![0.5; dim];
    
    println!("Testing quantization...");
    let start = Instant::now();
    
    for (i, vec) in vectors.iter().enumerate() {
        if i % 100 == 0 {
            println!("  Quantized {}/{} vectors", i, n_vectors);
        }
        let _ = quantize_simple(vec, &centroid, 4);
    }
    
    let elapsed = start.elapsed();
    println!("\nCompleted {} vectors in {:?}", n_vectors, elapsed);
    println!("Average: {:.3}ms per vector", elapsed.as_secs_f64() * 1000.0 / n_vectors as f64);
}

fn quantize_simple(vector: &[f32], centroid: &[f32], bits: usize) -> (Vec<u8>, Vec<u8>) {
    let dim = vector.len();
    
    // Compute residual
    let residual: Vec<f32> = vector.iter()
        .zip(centroid.iter())
        .map(|(v, c)| v - c)
        .collect();
    
    // Binary code
    let binary_bytes = (dim + 7) / 8;
    let mut binary_code = vec![0u8; binary_bytes];
    
    for (i, &val) in residual.iter().enumerate() {
        if val >= 0.0 {
            binary_code[i / 8] |= 1 << (i % 8);
        }
    }
    
    // Ex-code
    let ex_bits = bits - 1;
    let ex_bytes = (dim * ex_bits + 7) / 8;
    let mut ex_code = vec![0u8; ex_bytes];
    
    let max_val = (1 << ex_bits) - 1;
    
    for (i, &val) in residual.iter().enumerate() {
        let abs_val = val.abs();
        let quantized = (abs_val * max_val as f32).min(max_val as f32) as u8;
        
        let bit_offset = i * ex_bits;
        let byte_offset = bit_offset / 8;
        let bit_in_byte = bit_offset % 8;
        
        if byte_offset < ex_code.len() {
            ex_code[byte_offset] |= quantized << bit_in_byte;
            if bit_in_byte + ex_bits > 8 && byte_offset + 1 < ex_code.len() {
                ex_code[byte_offset + 1] |= quantized >> (8 - bit_in_byte);
            }
        }
    }
    
    (binary_code, ex_code)
}
