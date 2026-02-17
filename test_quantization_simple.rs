use std::time::Instant;

fn main() {
    println!("=== Simple Quantization Test ===\n");
    
    // Test with different dimensions
    let test_dims = vec![128, 768, 960];
    
    for dim in test_dims {
        println!("Testing dimension: {}", dim);
        
        // Create a simple vector
        let vector: Vec<f32> = (0..dim).map(|i| (i as f32) / (dim as f32)).collect();
        let centroid: Vec<f32> = vec![0.5; dim];
        
        // Test quantization
        let start = Instant::now();
        let (binary_code, ex_code) = quantize_simple(&vector, &centroid, 4);
        let quant_time = start.elapsed();
        
        println!("  Quantization time: {:?}", quant_time);
        println!("  Binary code size: {} bytes", binary_code.len());
        println!("  Ex code size: {} bytes", ex_code.len());
        println!("  Expected binary: {} bytes", dim / 8);
        println!("  Expected ex: {} bytes\n", (dim * 3) / 8);
    }
}

fn quantize_simple(vector: &[f32], centroid: &[f32], bits: usize) -> (Vec<u8>, Vec<u8>) {
    let dim = vector.len();
    
    // Step 1: Compute residual
    let residual: Vec<f32> = vector.iter()
        .zip(centroid.iter())
        .map(|(v, c)| v - c)
        .collect();
    
    // Step 2: Binary code (1-bit)
    let binary_bytes = (dim + 7) / 8;
    let mut binary_code = vec![0u8; binary_bytes];
    
    for (i, &val) in residual.iter().enumerate() {
        if val >= 0.0 {
            binary_code[i / 8] |= 1 << (i % 8);
        }
    }
    
    // Step 3: Ex-code (remaining bits)
    let ex_bits = bits - 1;
    let ex_bytes = (dim * ex_bits + 7) / 8;
    let mut ex_code = vec![0u8; ex_bytes];
    
    // Simple quantization: map absolute values to ex_bits range
    let max_val = (1 << ex_bits) - 1;
    
    for (i, &val) in residual.iter().enumerate() {
        let abs_val = val.abs();
        let quantized = (abs_val * max_val as f32).min(max_val as f32) as u8;
        
        // Pack into ex_code
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
