/// HBC (Hierarchical Balanced Clustering) head selection
/// Matches SPTAG's SelectHeadDynamically algorithm
use crate::index::bktree_recursive::BKTreeBuilder;
use crate::dataset::Dataset;

pub struct HBCSelector {
    select_threshold: usize,  // Min cluster size to select (default: 6)
    split_threshold: usize,   // Max cluster size before splitting (default: 25)
    split_factor: usize,      // Sub-heads from large clusters (default: 5)
}

impl HBCSelector {
    pub fn new() -> Self {
        Self {
            select_threshold: 6,   // SPTAG default
            split_threshold: 25,   // SPTAG default
            split_factor: 5,       // SPTAG default
        }
    }
    
    /// Select heads using HBC algorithm
    pub fn select_heads(
        &self,
        vectors: &[Vec<f32>],
        target_ratio: f32,
    ) -> Vec<usize> {
        self.select_heads_with_sampling(vectors, target_ratio, None)
    }
    
    /// Select heads using HBC algorithm with optional sampling
    pub fn select_heads_with_sampling(
        &self,
        vectors: &[Vec<f32>],
        target_ratio: f32,
        sample_size: Option<usize>,
    ) -> Vec<usize> {
        let n = vectors.len();
        
        // Determine if we should sample
        let actual_sample_size = match sample_size {
            Some(size) => size.min(n),
            None => n,  // No sampling by default (SPTAG behavior)
        };
        
        let use_sampling = actual_sample_size < n;
        
        if use_sampling {
            println!("  Using sampling for HBC: {} / {} vectors", actual_sample_size, n);
        }
        
        // Sample vectors for tree building
        let sampled_indices: Vec<usize> = if use_sampling {
            (0..n).step_by(n / actual_sample_size).take(actual_sample_size).collect()
        } else {
            (0..n).collect()
        };
        
        // Convert to Dataset for zero-copy tree building
        let sampled_vectors: Vec<Vec<f32>> = sampled_indices.iter()
            .map(|&i| vectors[i].clone())
            .collect();
        let sampled_dataset = crate::dataset::Dataset::from_vectors(&sampled_vectors);
        
        // Build BKT tree on sampled vectors
        let mut tree = BKTreeBuilder::new(32, 8, 1000);
        let _leaves = tree.build(&sampled_dataset);
        
        // Tune thresholds to hit target ratio
        let (select_thresh, split_thresh) = self.tune_thresholds(
            &tree,
            actual_sample_size,
            target_ratio,
        );
        
        // Select heads from sampled vectors
        let mut selected_in_sample = Vec::new();
        self.select_recursive(
            &tree,
            0,
            select_thresh,
            split_thresh,
            &mut selected_in_sample,
        );
        
        // Map back to original indices
        let selected: Vec<usize> = selected_in_sample.iter()
            .map(|&i| sampled_indices[i])
            .collect();
        
        selected
    }
    
    fn tune_thresholds(
        &self,
        tree: &BKTreeBuilder,
        total_size: usize,
        target_ratio: f32,
    ) -> (usize, usize) {
        // Auto-compute initial thresholds from ratio (SPTAG does this)
        let select_threshold = if self.select_threshold == 0 {
            (1.0 / target_ratio) as usize
        } else {
            self.select_threshold
        };
        
        let split_threshold = if self.split_threshold == 0 {
            select_threshold * 2
        } else {
            self.split_threshold
        };
        
        let split_factor = if self.split_factor == 0 {
            (1.0 / target_ratio + 0.5) as usize
        } else {
            self.split_factor
        };
        
        let target_count = (total_size as f32 * target_ratio) as usize;
        let mut best_select = select_threshold;
        let mut best_split = split_threshold;
        let mut min_diff = f32::MAX;
        
        // Binary search for optimal thresholds (SPTAG algorithm)
        for select in 2..=select_threshold.min(6) {
            let mut l = split_factor;
            let mut r = split_threshold;
            
            while l < r - 1 {
                let split = (l + r) / 2;
                let mut selected = Vec::new();
                self.select_recursive(tree, 0, select, split, &mut selected);
                
                let diff = (selected.len() as f32 / total_size as f32 - target_ratio).abs();
                
                if diff < min_diff {
                    min_diff = diff;
                    best_select = select;
                    best_split = split;
                }
                
                if selected.len() > target_count {
                    l = (l + r) / 2;
                } else {
                    r = (l + r) / 2;
                }
            }
        }
        
        println!("    Tuned thresholds: select={}, split={}, diff={:.2}%", 
                 best_select, best_split, min_diff * 100.0);
        
        (best_select, best_split)
    }
    
    fn select_recursive(
        &self,
        tree: &BKTreeBuilder,
        node_idx: usize,
        select_threshold: usize,
        split_threshold: usize,
        selected: &mut Vec<usize>,
    ) -> usize {
        // Get node info from tree
        let (center_id, children) = tree.get_node_info(node_idx);
        let mut children_size = 1;
        
        // Recurse on children
        let mut child_sizes = Vec::new();
        for &child_idx in &children {
            let size = self.select_recursive(
                tree,
                child_idx,
                select_threshold,
                split_threshold,
                selected,
            );
            if size > 0 {
                child_sizes.push((child_idx, size));
                children_size += size;
            }
        }
        
        // Select this node if cluster is large enough
        if children_size >= select_threshold {
            selected.push(center_id);
            
            // If cluster too large, also select from largest children
            if children_size > split_threshold {
                child_sizes.sort_by(|a, b| b.1.cmp(&a.1));
                let select_cnt = (children_size as f32 / self.split_factor as f32).ceil() as usize;
                
                for i in 0..select_cnt.min(child_sizes.len()) {
                    let (child_idx, _) = child_sizes[i];
                    let (child_center, _) = tree.get_node_info(child_idx);
                    selected.push(child_center);
                }
            }
            
            return 0;  // Don't count this subtree again
        }
        
        children_size
    }
}
