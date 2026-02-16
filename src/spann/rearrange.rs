/// Posting list rearrangement for better cache locality
use crate::spann::PostingList;

pub fn rearrange_posting_list(
    posting: &mut PostingList,
    full_vectors: &[Vec<f32>],
    head_vec: &[f32],
    metric: crate::spann::DistanceMetric,
) {
    if posting.vector_ids.is_empty() {
        return;
    }
    
    // Calculate distances to head
    let mut distances: Vec<(usize, f32)> = posting.vector_ids
        .iter()
        .map(|&id| {
            let vec = &full_vectors[id];
            let dist = match metric {
                crate::spann::DistanceMetric::L2 => l2_distance(vec, head_vec),
                crate::spann::DistanceMetric::InnerProduct => -inner_product(vec, head_vec),
            };
            (id, dist)
        })
        .collect();
    
    // Sort by distance (closest first)
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    
    // Update vector_ids in sorted order
    posting.vector_ids = distances.into_iter().map(|(id, _)| id).collect();
}

#[inline]
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

#[inline]
fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
