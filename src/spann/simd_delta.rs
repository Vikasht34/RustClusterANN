/// SIMD-optimized delta encoding/decoding

#[inline]
pub fn encode_delta_simd(vector: &[f32], head: &[f32], output: &mut [f32]) {
    debug_assert_eq!(vector.len(), head.len());
    debug_assert_eq!(vector.len(), output.len());
    
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { encode_delta_avx2(vector, head, output) };
        return;
    }
    
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    {
        unsafe { encode_delta_sse2(vector, head, output) };
        return;
    }
    
    // Fallback
    for i in 0..vector.len() {
        output[i] = vector[i] - head[i];
    }
}

#[inline]
pub fn decode_delta_simd(delta: &[f32], head: &[f32], output: &mut [f32]) {
    debug_assert_eq!(delta.len(), head.len());
    debug_assert_eq!(delta.len(), output.len());
    
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { decode_delta_avx2(delta, head, output) };
        return;
    }
    
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    {
        unsafe { decode_delta_sse2(delta, head, output) };
        return;
    }
    
    // Fallback
    for i in 0..delta.len() {
        output[i] = delta[i] + head[i];
    }
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn encode_delta_avx2(vector: &[f32], head: &[f32], output: &mut [f32]) {
    let len = vector.len();
    let chunks = len / 8;
    
    for i in 0..chunks {
        let offset = i * 8;
        let v = _mm256_loadu_ps(vector.as_ptr().add(offset));
        let h = _mm256_loadu_ps(head.as_ptr().add(offset));
        let delta = _mm256_sub_ps(v, h);
        _mm256_storeu_ps(output.as_mut_ptr().add(offset), delta);
    }
    
    // Handle remainder
    for i in (chunks * 8)..len {
        output[i] = vector[i] - head[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decode_delta_avx2(delta: &[f32], head: &[f32], output: &mut [f32]) {
    let len = delta.len();
    let chunks = len / 8;
    
    for i in 0..chunks {
        let offset = i * 8;
        let d = _mm256_loadu_ps(delta.as_ptr().add(offset));
        let h = _mm256_loadu_ps(head.as_ptr().add(offset));
        let result = _mm256_add_ps(d, h);
        _mm256_storeu_ps(output.as_mut_ptr().add(offset), result);
    }
    
    // Handle remainder
    for i in (chunks * 8)..len {
        output[i] = delta[i] + head[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn encode_delta_sse2(vector: &[f32], head: &[f32], output: &mut [f32]) {
    let len = vector.len();
    let chunks = len / 4;
    
    for i in 0..chunks {
        let offset = i * 4;
        let v = _mm_loadu_ps(vector.as_ptr().add(offset));
        let h = _mm_loadu_ps(head.as_ptr().add(offset));
        let delta = _mm_sub_ps(v, h);
        _mm_storeu_ps(output.as_mut_ptr().add(offset), delta);
    }
    
    for i in (chunks * 4)..len {
        output[i] = vector[i] - head[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn decode_delta_sse2(delta: &[f32], head: &[f32], output: &mut [f32]) {
    let len = delta.len();
    let chunks = len / 4;
    
    for i in 0..chunks {
        let offset = i * 4;
        let d = _mm_loadu_ps(delta.as_ptr().add(offset));
        let h = _mm_loadu_ps(head.as_ptr().add(offset));
        let result = _mm_add_ps(d, h);
        _mm_storeu_ps(output.as_mut_ptr().add(offset), result);
    }
    
    for i in (chunks * 4)..len {
        output[i] = delta[i] + head[i];
    }
}
