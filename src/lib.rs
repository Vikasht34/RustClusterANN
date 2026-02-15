pub mod multibit;
pub mod simd;
pub mod index;
pub mod spann;
pub mod recall;

use std::collections::BinaryHeap;
use std::cmp::Ordering;

const K_CONST_EPSILON: f32 = 1.9;

#[derive(Clone, Copy)]
pub enum MetricType {
    L2,
    IP,
}

pub use multibit::{MultiBitQuantizer, MetricType as MultiBitMetricType};

#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// One-bit quantization with factors
pub struct OneBitQuantizer {
    pub dim: usize,
    pub centroid: Vec<f32>,
    pub binary_codes: Vec<u8>,  // 0 or 1 per dimension
    pub f_add: Vec<f32>,
    pub f_rescale: Vec<f32>,
    pub f_error: Vec<f32>,
    pub n_vectors: usize,
}

impl OneBitQuantizer {
    pub fn new(dim: usize) -> Self {
        OneBitQuantizer {
            dim,
            centroid: vec![0.0; dim],
            binary_codes: Vec::new(),
            f_add: Vec::new(),
            f_rescale: Vec::new(),
            f_error: Vec::new(),
            n_vectors: 0,
        }
    }
    
    /// Quantize vectors with 1-bit codes and compute factors
    pub fn fit(&mut self, vectors: &[Vec<f32>], centroid: &[f32], metric: MetricType) {
        self.n_vectors = vectors.len();
        self.centroid = centroid.to_vec();
        
        let mut binary_codes = vec![0; self.n_vectors * self.dim];
        let mut f_add = vec![0.0; self.n_vectors];
        let mut f_rescale = vec![0.0; self.n_vectors];
        let mut f_error = vec![0.0; self.n_vectors];
        
        for (i, vec) in vectors.iter().enumerate() {
            let offset = i * self.dim;
            Self::one_bit_code_with_factor_static(
                vec,
                &self.centroid,
                self.dim,
                &mut binary_codes[offset..offset + self.dim],
                &mut f_add[i],
                &mut f_rescale[i],
                &mut f_error[i],
                metric,
            );
        }
        
        self.binary_codes = binary_codes;
        self.f_add = f_add;
        self.f_rescale = f_rescale;
        self.f_error = f_error;
    }
    
    /// Core 1-bit quantization with factor computation (static version)
    fn one_bit_code_with_factor_static(
        data: &[f32],
        centroid: &[f32],
        dim: usize,
        binary_code: &mut [u8],
        f_add: &mut f32,
        f_rescale: &mut f32,
        f_error: &mut f32,
        metric_type: MetricType,
    ) {
        // Compute residual: data - centroid
        let mut residual = vec![0.0f32; dim];
        for i in 0..dim {
            residual[i] = data[i] - centroid[i];
        }
        
        // Binary code: record sign of each coordinate
        for i in 0..dim {
            binary_code[i] = if residual[i] > 0.0 { 1 } else { 0 };
        }
        
        // xu_cb = x_u + cb, where cb = -((1 << 1) - 1) / 2 = -0.5
        let cb = -0.5f32;
        let mut xu_cb = vec![0.0f32; dim];
        for i in 0..dim {
            xu_cb[i] = binary_code[i] as f32 + cb;
        }
        
        // Compute norms and inner products
        let l2_sqr = dot_product(&residual, &residual);
        let l2_norm = l2_sqr.sqrt();
        
        let mut ip_resi_xucb = dot_product(&residual, &xu_cb);
        let ip_cent_xucb = dot_product(centroid, &xu_cb);
        
        // Handle corner case
        if ip_resi_xucb == 0.0 {
            ip_resi_xucb = f32::INFINITY;
        }
        
        // Compute error bound
        let xu_cb_norm_sqr = dot_product(&xu_cb, &xu_cb);
        let tmp_error = l2_norm * K_CONST_EPSILON *
            ((((l2_sqr * xu_cb_norm_sqr) / (ip_resi_xucb * ip_resi_xucb)) - 1.0) / 
             ((dim - 1) as f32)).sqrt();
        
        // Compute factors based on metric
        match metric_type {
            MetricType::L2 => {
                *f_add = l2_sqr + (2.0 * l2_sqr * ip_cent_xucb / ip_resi_xucb);
                *f_rescale = -2.0 * l2_sqr / ip_resi_xucb;
                *f_error = 2.0 * tmp_error;
            }
            MetricType::IP => {
                let ip_res_cent = dot_product(&residual, centroid);
                *f_add = 1.0 - ip_res_cent + (l2_sqr * ip_cent_xucb / ip_resi_xucb);
                *f_rescale = -l2_sqr / ip_resi_xucb;
                *f_error = tmp_error;
            }
        }
    }
    
    /// Compute distances using factor-based formula
    pub fn compute_distances(&self, query: &[f32], metric: MetricType) -> Vec<f32> {
        // Compute query-dependent term (G_add)
        let g_add = match metric {
            MetricType::IP => -dot_product(query, &self.centroid),
            MetricType::L2 => dot_product(query, query),
        };
        
        let mut distances = vec![0.0f32; self.n_vectors];
        
        for i in 0..self.n_vectors {
            let offset = i * self.dim;
            let codes = &self.binary_codes[offset..offset + self.dim];
            
            // Compute inner product: <query, xu_cb>
            // where xu_cb = codes - 0.5
            let mut ip_query_xucb = 0.0f32;
            for d in 0..self.dim {
                let xu_cb = codes[d] as f32 - 0.5;
                ip_query_xucb += query[d] * xu_cb;
            }
            
            // Distance formula: f_add + G_add + f_rescale * ip_query_xucb
            distances[i] = self.f_add[i] + g_add + self.f_rescale[i] * ip_query_xucb;
        }
        
        distances
    }
}

/// Multi-bit quantization with optimal rescaling
/// SIMD inner product for 2-bit codes
#[inline]
fn ip_fxu2_simd(query: &[f32], codes: &[u8]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        ip_fxu2_neon(query, codes)
    }
    
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if is_x86_feature_detected!("avx2") {
            ip_fxu2_avx2(query, codes)
        } else {
            ip_fxu2_scalar(query, codes)
        }
    }
    
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        ip_fxu2_scalar(query, codes)
    }
}

/// SIMD inner product for 4-bit codes
#[inline]
fn ip_fxu4_simd(query: &[f32], codes: &[u8]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        ip_fxu4_neon(query, codes)
    }
    
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if is_x86_feature_detected!("avx2") {
            ip_fxu4_avx2(query, codes)
        } else {
            ip_fxu4_scalar(query, codes)
        }
    }
    
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        ip_fxu4_scalar(query, codes)
    }
}

// Scalar fallback implementations
#[inline]
fn ip_fxu2_scalar(query: &[f32], codes: &[u8]) -> f32 {
    query.iter().zip(codes.iter()).map(|(q, &c)| q * c as f32).sum()
}

#[inline]
fn ip_fxu4_scalar(query: &[f32], codes: &[u8]) -> f32 {
    query.iter().zip(codes.iter()).map(|(q, &c)| q * c as f32).sum()
}

// ARM NEON implementations
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn ip_fxu2_neon(query: &[f32], codes: &[u8]) -> f32 {
    use std::arch::aarch64::*;
    
    let len = query.len();
    let mut sum = vdupq_n_f32(0.0);
    let mut i = 0;
    
    while i + 4 <= len {
        let q = vld1q_f32(query.as_ptr().add(i));
        
        // Load 4 uint8 codes and convert to f32
        let codes_u8 = vld1_u8(codes.as_ptr().add(i));
        let codes_u16 = vmovl_u8(codes_u8);
        let codes_u32 = vmovl_u16(vget_low_u16(codes_u16));
        let codes_f32 = vcvtq_f32_u32(codes_u32);
        
        sum = vfmaq_f32(sum, q, codes_f32);
        i += 4;
    }
    
    let mut result = vaddvq_f32(sum);
    
    while i < len {
        result += query[i] * codes[i] as f32;
        i += 1;
    }
    
    result
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn ip_fxu4_neon(query: &[f32], codes: &[u8]) -> f32 {
    use std::arch::aarch64::*;
    
    let len = query.len();
    let mut sum = vdupq_n_f32(0.0);
    let mut i = 0;
    
    while i + 4 <= len {
        let q = vld1q_f32(query.as_ptr().add(i));
        
        let codes_u8 = vld1_u8(codes.as_ptr().add(i));
        let codes_u16 = vmovl_u8(codes_u8);
        let codes_u32 = vmovl_u16(vget_low_u16(codes_u16));
        let codes_f32 = vcvtq_f32_u32(codes_u32);
        
        sum = vfmaq_f32(sum, q, codes_f32);
        i += 4;
    }
    
    let mut result = vaddvq_f32(sum);
    
    while i < len {
        result += query[i] * codes[i] as f32;
        i += 1;
    }
    
    result
}

// x86 AVX2 implementations
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn ip_fxu2_avx2(query: &[f32], codes: &[u8]) -> f32 {
    use std::arch::x86_64::*;
    
    let len = query.len();
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;
    
    while i + 8 <= len {
        let q = _mm256_loadu_ps(query.as_ptr().add(i));
        
        // Load 8 uint8 codes
        let codes_i64 = std::ptr::read_unaligned(codes.as_ptr().add(i) as *const i64);
        let codes_i128 = _mm_set1_epi64x(codes_i64);
        
        // Convert u8 to i32 then to f32
        let codes_i32 = _mm256_cvtepu8_epi32(codes_i128);
        let codes_f32 = _mm256_cvtepi32_ps(codes_i32);
        
        sum = _mm256_fmadd_ps(q, codes_f32, sum);
        i += 8;
    }
    
    // Horizontal sum
    let sum_high = _mm256_extractf128_ps(sum, 1);
    let sum_low = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(sum_low, sum_high);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x55));
    let mut result = _mm_cvtss_f32(sum32);
    
    while i < len {
        result += query[i] * codes[i] as f32;
        i += 1;
    }
    
    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn ip_fxu4_avx2(query: &[f32], codes: &[u8]) -> f32 {
    use std::arch::x86_64::*;
    
    let len = query.len();
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;
    
    while i + 8 <= len {
        let q = _mm256_loadu_ps(query.as_ptr().add(i));
        
        let codes_i64 = std::ptr::read_unaligned(codes.as_ptr().add(i) as *const i64);
        let codes_i128 = _mm_set1_epi64x(codes_i64);
        let codes_i32 = _mm256_cvtepu8_epi32(codes_i128);
        let codes_f32 = _mm256_cvtepi32_ps(codes_i32);
        
        sum = _mm256_fmadd_ps(q, codes_f32, sum);
        i += 8;
    }
    
    let sum_high = _mm256_extractf128_ps(sum, 1);
    let sum_low = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(sum_low, sum_high);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 0x55));
    let mut result = _mm_cvtss_f32(sum32);
    
    while i < len {
        result += query[i] * codes[i] as f32;
        i += 1;
    }
    
    result
}

pub mod lib {
    pub use super::*;
}
