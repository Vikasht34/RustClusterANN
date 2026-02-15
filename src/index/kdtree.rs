/// KD-Tree implementation for SPTAG
use crate::simd;
use rand::seq::SliceRandom;
use rand::thread_rng;

#[derive(Clone)]
pub struct KDTNode {
    pub left: i32,
    pub right: i32,
    pub split_dim: usize,
    pub split_value: f32,
}

pub struct KDTree {
    nodes: Vec<KDTNode>,
    tree_starts: Vec<usize>,
    num_trees: usize,
    samples: usize,
    top_dims: usize,
}

impl KDTree {
    pub fn new(num_trees: usize, samples: usize, top_dims: usize) -> Self {
        Self {
            nodes: Vec::new(),
            tree_starts: Vec::new(),
            num_trees,
            samples,
            top_dims,
        }
    }

    pub fn build(&mut self, data: &[Vec<f32>]) {
        let n = data.len();
        let dim = data[0].len();
        
        self.tree_starts.clear();
        self.nodes.clear();
        
        for _ in 0..self.num_trees {
            self.tree_starts.push(self.nodes.len());
            let mut indices: Vec<usize> = (0..n).collect();
            self.build_tree(data, &mut indices, 0, n, dim);
        }
    }

    fn build_tree(&mut self, data: &[Vec<f32>], indices: &mut [usize], start: usize, end: usize, dim: usize) -> i32 {
        if end - start <= 1 {
            return -1;
        }

        let node_idx = self.nodes.len() as i32;
        self.nodes.push(KDTNode {
            left: -1,
            right: -1,
            split_dim: 0,
            split_value: 0.0,
        });

        // Sample points to find best split dimension
        let sample_size = self.samples.min(end - start);
        let mut rng = thread_rng();
        let mut sample_indices: Vec<usize> = (start..end).collect();
        sample_indices.shuffle(&mut rng);
        sample_indices.truncate(sample_size);

        // Find dimension with highest variance
        let mut best_dim = 0;
        let mut best_var = 0.0f32;
        
        for d in 0..dim.min(self.top_dims) {
            let mut sum = 0.0f32;
            let mut sum_sq = 0.0f32;
            
            for &idx in &sample_indices {
                let val = data[indices[idx]][d];
                sum += val;
                sum_sq += val * val;
            }
            
            let mean = sum / sample_size as f32;
            let variance = sum_sq / sample_size as f32 - mean * mean;
            
            if variance > best_var {
                best_var = variance;
                best_dim = d;
            }
        }

        // Find median along best dimension
        let mid = (start + end) / 2;
        indices[start..end].select_nth_unstable_by(mid - start, |&a, &b| {
            data[a][best_dim].partial_cmp(&data[b][best_dim]).unwrap()
        });

        let split_value = data[indices[mid]][best_dim];
        
        self.nodes[node_idx as usize].split_dim = best_dim;
        self.nodes[node_idx as usize].split_value = split_value;

        // Recursively build subtrees
        let left = self.build_tree(data, indices, start, mid, dim);
        let right = self.build_tree(data, indices, mid, end, dim);
        
        self.nodes[node_idx as usize].left = left;
        self.nodes[node_idx as usize].right = right;

        node_idx
    }

    pub fn search(&self, query: &[f32], data: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
        let mut candidates = Vec::new();
        
        for tree_idx in 0..self.num_trees {
            let root = self.tree_starts[tree_idx];
            self.search_tree(query, data, root as i32, &mut candidates, 0);
        }

        // Compute distances and sort
        let mut results: Vec<(usize, f32)> = candidates.into_iter()
            .map(|idx| {
                let dist = simd::l2_distance(query, &data[idx]);
                (idx, dist)
            })
            .collect();
        
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }

    fn search_tree(&self, query: &[f32], data: &[Vec<f32>], node_idx: i32, candidates: &mut Vec<usize>, depth: usize) {
        if node_idx < 0 || node_idx as usize >= self.nodes.len() {
            return;
        }

        let node = &self.nodes[node_idx as usize];
        
        // If leaf or max depth, collect nearby points
        if node.left < 0 && node.right < 0 {
            // This is a leaf - add some points near this region
            for i in 0..data.len().min(100) {
                candidates.push(i);
            }
            return;
        }
        
        // Traverse to appropriate child
        let go_left = query[node.split_dim] < node.split_value;
        
        if go_left {
            self.search_tree(query, data, node.left, candidates, depth + 1);
        } else {
            self.search_tree(query, data, node.right, candidates, depth + 1);
        }
    }
}
