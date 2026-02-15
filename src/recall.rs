use std::fs::File;
use std::io::{BufReader, Read};

/// Load ground truth from ivecs file
pub fn load_ground_truth(path: &str, n: usize) -> Vec<Vec<usize>> {
    let file = File::open(path).expect("Failed to open ground truth file");
    let mut reader = BufReader::new(file);
    
    let mut ground_truth = Vec::with_capacity(n);
    let mut buffer = [0u8; 4];
    
    for _ in 0..n {
        // Read dimension (number of neighbors)
        reader.read_exact(&mut buffer).expect("Failed to read dimension");
        let dim = u32::from_le_bytes(buffer) as usize;
        
        // Read neighbor IDs
        let mut neighbors = vec![0usize; dim];
        for j in 0..dim {
            reader.read_exact(&mut buffer).expect("Failed to read neighbor");
            neighbors[j] = u32::from_le_bytes(buffer) as usize;
        }
        
        ground_truth.push(neighbors);
    }
    
    ground_truth
}

/// Calculate recall@k
pub fn calculate_recall(results: &[(usize, f32)], ground_truth: &[usize], k: usize) -> f32 {
    let result_ids: Vec<usize> = results.iter().take(k).map(|(id, _)| *id).collect();
    let gt_ids: Vec<usize> = ground_truth.iter().take(k).copied().collect();
    
    let mut matches = 0;
    for &result_id in &result_ids {
        if gt_ids.contains(&result_id) {
            matches += 1;
        }
    }
    
    matches as f32 / k as f32
}

/// Calculate average recall@k over all queries
pub fn calculate_average_recall(
    all_results: &[Vec<(usize, f32)>],
    ground_truth: &[Vec<usize>],
    k: usize,
) -> f32 {
    let mut total_recall = 0.0;
    let n = all_results.len().min(ground_truth.len());
    
    for i in 0..n {
        total_recall += calculate_recall(&all_results[i], &ground_truth[i], k);
    }
    
    total_recall / n as f32
}
