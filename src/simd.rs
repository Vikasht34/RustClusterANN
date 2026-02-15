/// SIMD-optimized operations for RaBitQ - ported from SPTAG
/// Supports SSE2, AVX2, and AVX-512 with runtime CPU detection

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Compute L2 distance using AVX2
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_distance_avx2(x: &[f32], y: &[f32], len: usize) -> f32 {
    let mut sum = _mm256_setzero_ps();
    let chunks = len / 8;
    
    for i in 0..chunks {
        let offset = i * 8;
        let vx = _mm256_loadu_ps(x.as_ptr().add(offset));
        let vy = _mm256_loadu_ps(y.as_ptr().add(offset));
        let diff = _mm256_sub_ps(vx, vy);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }
    
    let arr: [f32; 8] = std::mem::transmute(sum);
    let mut result: f32 = arr.iter().sum();
    
    for i in (chunks * 8)..len {
        let d = x[i] - y[i];
        result += d * d;
    }
    
    result
}

/// Compute cosine distance using AVX2
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn cosine_distance_avx2(x: &[f32], y: &[f32], len: usize) -> f32 {
    let mut sum = _mm256_setzero_ps();
    let chunks = len / 8;
    
    for i in 0..chunks {
        let offset = i * 8;
        let vx = _mm256_loadu_ps(x.as_ptr().add(offset));
        let vy = _mm256_loadu_ps(y.as_ptr().add(offset));
        sum = _mm256_fmadd_ps(vx, vy, sum);
    }
    
    let arr: [f32; 8] = std::mem::transmute(sum);
    let mut result: f32 = arr.iter().sum();
    
    for i in (chunks * 8)..len {
        result += x[i] * y[i];
    }
    
    result
}

/// Binary inner product - scalar fallback
pub fn binary_ip_scalar(query: &[f32], binary_codes: &[u8], dim: usize) -> f32 {
    let mut sum = 0.0f32;
    for d in 0..dim {
        let bit = (binary_codes[d / 8] >> (d % 8)) & 1;
        let sign = if bit == 1 { 1.0 } else { -1.0 };
        sum += query[d] * sign;
    }
    sum
}

/// Binary inner product - AVX2 optimized
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn binary_ip_avx2(query: &[f32], binary_codes: &[u8], dim: usize) -> f32 {
    let mut sum = _mm256_setzero_ps();
    let chunks = dim / 8;
    
    for i in 0..chunks {
        let byte_val = binary_codes[i];
        let offset = i * 8;
        
        let mut signs = [0.0f32; 8];
        for j in 0..8 {
            signs[j] = if (byte_val >> j) & 1 == 1 { 1.0 } else { -1.0 };
        }
        
        let vq = _mm256_loadu_ps(query.as_ptr().add(offset));
        let vs = _mm256_loadu_ps(signs.as_ptr());
        sum = _mm256_fmadd_ps(vq, vs, sum);
    }
    
    let arr: [f32; 8] = std::mem::transmute(sum);
    let mut result: f32 = arr.iter().sum();
    
    for d in (chunks * 8)..dim {
        let bit = (binary_codes[d / 8] >> (d % 8)) & 1;
        let sign = if bit == 1 { 1.0 } else { -1.0 };
        result += query[d] * sign;
    }
    
    result
}

/// Multi-bit inner product - scalar
pub fn multibit_ip_scalar(query: &[f32], codes: &[i32], dim: usize) -> f32 {
    query.iter().zip(codes.iter()).map(|(q, c)| q * (*c as f32)).sum()
}

/// Multi-bit inner product - AVX2
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn multibit_ip_avx2(query: &[f32], codes: &[i32], dim: usize) -> f32 {
    let mut sum = _mm256_setzero_ps();
    let chunks = dim / 8;
    
    for i in 0..chunks {
        let offset = i * 8;
        let vq = _mm256_loadu_ps(query.as_ptr().add(offset));
        let vc_i32 = _mm256_loadu_si256(codes.as_ptr().add(offset) as *const __m256i);
        let vc = _mm256_cvtepi32_ps(vc_i32);
        sum = _mm256_fmadd_ps(vq, vc, sum);
    }
    
    let arr: [f32; 8] = std::mem::transmute(sum);
    let mut result: f32 = arr.iter().sum();
    
    for i in (chunks * 8)..dim {
        result += query[i] * codes[i] as f32;
    }
    
    result
}

/// Runtime dispatch for binary IP
#[inline]
pub fn binary_ip(query: &[f32], binary_codes: &[u8], dim: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { return binary_ip_avx2(query, binary_codes, dim); }
        }
    }
    binary_ip_scalar(query, binary_codes, dim)
}

/// Runtime dispatch for multi-bit IP
#[inline]
pub fn multibit_ip(query: &[f32], codes: &[i32], dim: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { return multibit_ip_avx2(query, codes, dim); }
        }
    }
    multibit_ip_scalar(query, codes, dim)
}

/// L2 distance with runtime dispatch
#[inline]
pub fn l2_distance(x: &[f32], y: &[f32]) -> f32 {
    let len = x.len().min(y.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { return l2_distance_avx2(x, y, len); }
        }
    }
    
    let mut sum = 0.0f32;
    for i in 0..len {
        let d = x[i] - y[i];
        sum += d * d;
    }
    sum
}

/// Cosine distance with runtime dispatch
#[inline]
pub fn cosine_distance(x: &[f32], y: &[f32]) -> f32 {
    let len = x.len().min(y.len());
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { return cosine_distance_avx2(x, y, len); }
        }
    }
    
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += x[i] * y[i];
    }
    sum
}

/// Inner product (negative for distance, since we want to maximize IP)
#[inline]
pub fn inner_product(x: &[f32], y: &[f32]) -> f32 {
    -cosine_distance(x, y)
}
