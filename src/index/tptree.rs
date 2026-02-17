/// TP-Tree (Tree Projection Tree) - Simple variance-based binary tree
/// Used by SPTAG for fast KNNG construction
use crate::dataset::Dataset;
use rand::Rng;

pub struct TPTree {
    leaf_size: usize,
    samples: usize,
    top_dims: usize,
}

impl TPTree {
    pub fn new(leaf_size: usize, samples: usize, top_dims: usize) -> Self {
        Self {
            leaf_size,
            samples,
            top_dims,
        }
    }
    
    /// Build TP-Tree and return leaf ranges
    pub fn build(&self, data: &Dataset, indices: &mut [usize]) -> Vec<(usize, usize)> {
        let mut leaves = Vec::new();
        self.build_recursive(data, indices, 0, indices.len(), &mut leaves);
        leaves
    }
    
    fn build_recursive(
        &self,
        data: &Dataset,
        indices: &mut [usize],
        first: usize,
        last: usize,
        leaves: &mut Vec<(usize, usize)>,
    ) {
        let size = last - first;
        
        // Leaf node
        if size <= self.leaf_size {
            leaves.push((first, last));
            return;
        }
        
        // Sample vectors for split calculation
        let sample_size = self.samples.min(size);
        let mut rng = rand::thread_rng();
        
        // Calculate mean and variance for each dimension
        let dim = data.cols();
        let mut mean = vec![0.0f32; dim];
        let mut variance = vec![0.0f32; dim];
        
        for _ in 0..sample_size {
            let idx = rng.gen_range(0..size);
            let vec = data.at(indices[first + idx]);
            for d in 0..dim {
                mean[d] += vec[d];
            }
        }
        
        for d in 0..dim {
            mean[d] /= sample_size as f32;
        }
        
        for _ in 0..sample_size {
            let idx = rng.gen_range(0..size);
            let vec = data.at(indices[first + idx]);
            for d in 0..dim {
                let diff = vec[d] - mean[d];
                variance[d] += diff * diff;
            }
        }
        
        // Find top dimensions with highest variance
        let mut dim_variance: Vec<(usize, f32)> = variance.iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        dim_variance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        let top_dims: Vec<usize> = dim_variance.iter()
            .take(self.top_dims)
            .map(|(i, _)| *i)
            .collect();
        
        // Try random weighted combinations to find best split
        let mut best_variance = 0.0f32;
        let mut best_weights = vec![1.0f32; self.top_dims];
        let mut best_mean = mean[top_dims[0]];
        
        for _ in 0..100 {
            // Random weights
            let mut weights: Vec<f32> = (0..self.top_dims)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();
            
            // Normalize
            let sum: f32 = weights.iter().map(|w| w * w).sum::<f32>().sqrt();
            for w in &mut weights {
                *w /= sum;
            }
            
            // Calculate projected mean and variance
            let mut proj_mean = 0.0f32;
            let mut proj_vals = Vec::with_capacity(sample_size);
            
            for _ in 0..sample_size {
                let idx = rng.gen_range(0..size);
                let vec = data.at(indices[first + idx]);
                let mut val = 0.0f32;
                for (i, &d) in top_dims.iter().enumerate() {
                    val += weights[i] * vec[d];
                }
                proj_vals.push(val);
                proj_mean += val;
            }
            proj_mean /= sample_size as f32;
            
            let mut proj_var = 0.0f32;
            for val in &proj_vals {
                let diff = val - proj_mean;
                proj_var += diff * diff;
            }
            
            if proj_var > best_variance {
                best_variance = proj_var;
                best_weights = weights;
                best_mean = proj_mean;
            }
        }
        
        // Partition indices in-place based on projection
        let mut left = first;
        let mut right = last - 1;
        
        while left < right {
            let vec = data.at(indices[left]);
            let mut proj = 0.0f32;
            for (j, &d) in top_dims.iter().enumerate() {
                proj += best_weights[j] * vec[d];
            }
            
            if proj < best_mean {
                left += 1;
            } else {
                indices.swap(left, right);
                right -= 1;
            }
        }
        
        // Handle degenerate splits
        if left == first || left == last {
            leaves.push((first, last));
            return;
        }
        
        // Recurse on partitions
        self.build_recursive(data, indices, first, left, leaves);
        self.build_recursive(data, indices, left, last, leaves);
    }
}
