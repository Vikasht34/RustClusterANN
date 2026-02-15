/// RaBitQ Multi-bit - Direct C++ translation
/// Based on rabitqlib/quantization/rabitq_impl.hpp

use std::collections::BinaryHeap;
use std::cmp::Ordering;

const K_TIGHT_START: [f32; 9] = [0.0, 0.15, 0.20, 0.52, 0.59, 0.71, 0.75, 0.77, 0.81];

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MetricType {
    L2,
    IP,
}

/// Multi-bit quantizer with 1-bit + ex_bits (RaBitQ architecture)
pub struct MultiBitQuantizer {
    dim: usize,
    bits: usize,  // total_bits (1-bit + ex_bits)
    centroid: Vec<f32>,
    // 1-bit base quantization
    binary_codes: Vec<u8>,  // Packed binary codes
    // ex_bits refinement
    ex_codes: Vec<u8>,
    ex_signs: Vec<i8>,
    f_add_ex: Vec<f32>,
    f_rescale_ex: Vec<f32>,
    n_vectors: usize,
}

impl MultiBitQuantizer {
    pub fn new(bits: usize) -> Self {
        MultiBitQuantizer {
            dim: 0,
            bits,
            centroid: Vec::new(),
            binary_codes: Vec::new(),
            ex_codes: Vec::new(),
            ex_signs: Vec::new(),
            f_add_ex: Vec::new(),
            f_rescale_ex: Vec::new(),
            n_vectors: 0,
        }
    }

    pub fn train(&mut self, vectors: &[Vec<f32>], metric: MetricType) {
        self.dim = vectors[0].len();
        
        // Compute centroid
        self.centroid = vec![0.0; self.dim];
        for v in vectors {
            for d in 0..self.dim {
                self.centroid[d] += v[d];
            }
        }
        for d in 0..self.dim {
            self.centroid[d] /= vectors.len() as f32;
        }
        
        self.fit(vectors, &self.centroid.clone(), metric);
    }
    
    pub fn fit(&mut self, vectors: &[Vec<f32>], centroid: &[f32], metric: MetricType) {
        self.n_vectors = vectors.len();
        self.centroid = centroid.to_vec();
        
        let ex_bits = self.bits - 1;  // RaBitQ: ex_bits = total_bits - 1
        
        // Allocate storage
        let bytes_per_vec = (self.dim + 7) / 8;
        self.binary_codes = vec![0u8; self.n_vectors * bytes_per_vec];
        self.ex_codes = vec![0u8; self.n_vectors * self.dim];
        self.ex_signs = vec![0i8; self.n_vectors * self.dim];
        self.f_add_ex = vec![0.0; self.n_vectors];
        self.f_rescale_ex = vec![0.0; self.n_vectors];
        
        for i in 0..self.n_vectors {
            // Compute residual
            let residual: Vec<f32> = vectors[i].iter().zip(centroid.iter())
                .map(|(v, c)| v - c).collect();
            
            // Pack 1-bit codes
            let offset = i * bytes_per_vec;
            for d in 0..self.dim {
                if residual[d] >= 0.0 {
                    self.binary_codes[offset + d / 8] |= 1 << (d % 8);
                }
            }
            
            // Compute ex_bits quantization with factors
            let ex_offset = i * self.dim;
            Self::ex_bits_code_with_factor_static(
                &vectors[i],
                &self.centroid,
                self.dim,
                ex_bits,
                &mut self.ex_codes[ex_offset..ex_offset + self.dim],
                &mut self.ex_signs[ex_offset..ex_offset + self.dim],
                &mut self.f_add_ex[i],
                &mut self.f_rescale_ex[i],
                metric,
            );
        }
    }
    
    /// Direct translation of ex_bits_code_with_factor from rabitq_impl.hpp
    /// Note: ex_bits parameter is total_bits - 1
    fn ex_bits_code_with_factor_static(
        data: &[f32],
        centroid: &[f32],
        dim: usize,
        ex_bits: usize,  // This is total_bits - 1
        ex_code: &mut [u8],
        signs: &mut [i8],
        f_add_ex: &mut f32,
        f_rescale_ex: &mut f32,
        metric: MetricType,
    ) {
        // Compute residual
        let mut residual = vec![0.0f32; dim];
        for i in 0..dim {
            residual[i] = data[i] - centroid[i];
        }
        
        // Normalize residual
        let norm: f32 = residual.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut normalized = residual.clone();
        if norm > 1e-10 {
            for i in 0..dim {
                normalized[i] /= norm;
            }
        }
        
        // Get absolute values and flip codes for negative dimensions
        let mut abs_res = vec![0.0f32; dim];
        for i in 0..dim {
            abs_res[i] = normalized[i].abs();
        }
        
        // Quantize
        let ipnorm_inv = Self::quantize_ex_static(&abs_res, ex_bits, ex_code);
        
        // Flip codes for negative dimensions (as in RaBitQ)
        let mask = ((1 << ex_bits) - 1) as u8;
        for i in 0..dim {
            if normalized[i] < 0.0 {
                ex_code[i] = (!ex_code[i]) & mask;
            }
            signs[i] = if normalized[i] >= 0.0 { 1 } else { -1 };
        }
        
        // Compute total_code: add (1 << ex_bits) for positive residuals
        let mut total_code = vec![0i32; dim];
        for i in 0..dim {
            total_code[i] = ex_code[i] as i32;
            if residual[i] >= 0.0 {
                total_code[i] += 1 << ex_bits;
            }
        }
        
        // Center codes using ex_bits: xu_cb = total_code - (2^ex_bits - 0.5)
        let cb = -((1 << ex_bits) as f32 - 0.5);
        let mut xu_cb = vec![0.0f32; dim];
        for i in 0..dim {
            xu_cb[i] = total_code[i] as f32 + cb;
        }
        
        // Compute factors using xu_cb
        let l2_sqr: f32 = residual.iter().map(|x| x * x).sum();
        let l2_norm = l2_sqr.sqrt();
        let ip_resi_xucb: f32 = residual.iter().zip(xu_cb.iter()).map(|(r, x)| r * x).sum();
        let ip_cent_xucb: f32 = centroid.iter().zip(xu_cb.iter()).map(|(c, x)| c * x).sum();
        
        let ip_resi_xucb = if ip_resi_xucb == 0.0 { f32::INFINITY } else { ip_resi_xucb };
        
        // Compute factors based on metric
        match metric {
            MetricType::L2 => {
                *f_add_ex = l2_sqr + (2.0 * l2_sqr * ip_cent_xucb / ip_resi_xucb);
                *f_rescale_ex = ipnorm_inv * -2.0 * l2_norm;
            }
            MetricType::IP => {
                let ip_res_cent: f32 = residual.iter().zip(centroid.iter()).map(|(r, c)| r * c).sum();
                *f_add_ex = 1.0 - ip_res_cent + (l2_sqr * ip_cent_xucb / ip_resi_xucb);
                *f_rescale_ex = ipnorm_inv * -l2_norm;
            }
        }
    }
    
    /// Direct translation of quantize_ex from rabitq_impl.hpp
    fn quantize_ex_static(o_abs: &[f32], bits: usize, code: &mut [u8]) -> f32 {
        let dim = o_abs.len();
        let t = Self::best_rescale_factor_static(o_abs, bits);
        
        let mut ipnorm = 0.0f64;
        let max_code = (1 << bits) - 1;
        
        for i in 0..dim {
            let tmp_code = ((t * o_abs[i] as f64) + 1e-5) as i32;
            let final_code = tmp_code.min(max_code);
            code[i] = final_code as u8;
            ipnorm += (final_code as f64 + 0.5) * o_abs[i] as f64;
        }
        
        let ipnorm_inv = (1.0 / ipnorm) as f32;
        if !ipnorm_inv.is_normal() { 1.0 } else { ipnorm_inv }
    }
    
    /// Direct translation of best_rescale_factor from rabitq_impl.hpp
    fn best_rescale_factor_static(o_abs: &[f32], bits: usize) -> f64 {
        const K_EPS: f64 = 1e-5;
        const K_N_ENUM: i32 = 10;
        
        let dim = o_abs.len();
        let max_o = o_abs.iter().cloned().fold(0.0f32, f32::max) as f64;
        let t_end = (((1 << bits) - 1) + K_N_ENUM) as f64 / max_o;
        let t_start = t_end * K_TIGHT_START[bits] as f64;
        
        let mut cur_o_bar = vec![0i32; dim];
        let mut sqr_denominator = (dim as f64) * 0.25;
        let mut numerator = 0.0f64;
        
        for i in 0..dim {
            let cur = ((t_start * o_abs[i] as f64) + K_EPS) as i32;
            cur_o_bar[i] = cur;
            sqr_denominator += (cur * cur) as f64 + cur as f64;
            numerator += (cur as f64 + 0.5) * o_abs[i] as f64;
        }
        
        #[derive(PartialEq)]
        struct Item(f64, usize);
        impl Eq for Item {}
        impl PartialOrd for Item {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                other.0.partial_cmp(&self.0)
            }
        }
        impl Ord for Item {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap()
            }
        }
        
        let mut next_t = BinaryHeap::new();
        for i in 0..dim {
            let t_next = (cur_o_bar[i] + 1) as f64 / o_abs[i] as f64;
            next_t.push(Item(t_next, i));
        }
        
        let mut max_ip = 0.0f64;
        let mut best_t = 0.0f64;
        
        while let Some(Item(cur_t, update_id)) = next_t.pop() {
            cur_o_bar[update_id] += 1;
            let update_o_bar = cur_o_bar[update_id];
            sqr_denominator += 2.0 * update_o_bar as f64;
            numerator += o_abs[update_id] as f64;
            
            let cur_ip = numerator / sqr_denominator.sqrt();
            if cur_ip > max_ip {
                max_ip = cur_ip;
                best_t = cur_t;
            }
            
            if update_o_bar < (1 << bits) - 1 {
                let t_next = (update_o_bar + 1) as f64 / o_abs[update_id] as f64;
                if t_next < t_end {
                    next_t.push(Item(t_next, update_id));
                }
            }
        }
        
        best_t
    }
    
    /// Compute distances using split_distance_boosting formula
    pub fn compute_distances(&self, query: &[f32], metric: MetricType) -> Vec<f32> {
        let g_add = match metric {
            MetricType::IP => {
                let dot: f32 = query.iter().zip(self.centroid.iter()).map(|(q, c)| q * c).sum();
                dot
            }
            MetricType::L2 => {
                query.iter().map(|x| x * x).sum()
            }
        };
        
        let query_for_ip = query;
        let ex_bits = self.bits - 1;
        let bytes_per_vec = (self.dim + 7) / 8;
        
        let c_b = -((1 << (ex_bits + 1)) - 1) as f32 / 2.0;
        let sumq: f32 = query_for_ip.iter().sum();
        let kbxsumq = sumq * c_b;
        
        let mut distances = vec![0.0f32; self.n_vectors];
        
        for i in 0..self.n_vectors {
            // 1. Compute 1-bit IP using SIMD
            let bin_offset = i * bytes_per_vec;
            let ip_1bit = crate::simd::binary_ip(
                query_for_ip,
                &self.binary_codes[bin_offset..bin_offset + bytes_per_vec],
                self.dim
            );
            
            // 2. Compute ex_bits IP - inline for speed
            let ex_offset = i * self.dim;
            let codes = &self.ex_codes[ex_offset..ex_offset + self.dim];
            let signs = &self.ex_signs[ex_offset..ex_offset + self.dim];
            
            let mut ip_exbits_raw = 0.0f32;
            for d in 0..self.dim {
                let total_code = codes[d] as i32 + if signs[d] > 0 { 1 << ex_bits } else { 0 };
                ip_exbits_raw += query_for_ip[d] * total_code as f32;
            }
            
            // 3. Combine
            let combined_ip = (1 << ex_bits) as f32 * ip_1bit + ip_exbits_raw + kbxsumq;
            distances[i] = self.f_add_ex[i] + g_add + self.f_rescale_ex[i] * combined_ip;
        }
        
        distances
    }

    pub fn reconstruct(&self, idx: usize) -> Vec<f32> {
        let mut result = self.centroid.clone();
        let ex_bits = self.bits - 1;
        let bytes_per_vec = (self.dim + 7) / 8;
        
        let bin_offset = idx * bytes_per_vec;
        let ex_offset = idx * self.dim;
        
        for d in 0..self.dim {
            let bit = (self.binary_codes[bin_offset + d / 8] >> (d % 8)) & 1;
            let sign = if bit == 1 { 1.0 } else { -1.0 };
            
            let code = self.ex_codes[ex_offset + d];
            let total_code = code as f32 + if self.ex_signs[ex_offset + d] > 0 { 1 << ex_bits } else { 0 } as f32;
            
            result[d] += sign * total_code * self.f_rescale_ex[idx];
        }
        
        result
    }
}

pub mod lib {
    pub use super::*;
}
