/// BK-Tree (Ball Tree with K-means) implementation for SPTAG
use crate::simd;
use rand::seq::SliceRandom;
use rand::thread_rng;

#[derive(Clone)]
pub struct BKTNode {
    pub center_id: i32,
    pub child_start: i32,
    pub child_end: i32,
}

pub struct BKTree {
    nodes: Vec<BKTNode>,
    num_trees: usize,
    k_means: usize,
    samples: usize,
}

impl BKTree {
    pub fn new(num_trees: usize, k_means: usize, samples: usize) -> Self {
        Self {
            nodes: Vec::new(),
            num_trees,
            k_means,
            samples,
        }
    }

    pub fn build(&mut self, data: &[Vec<f32>]) {
        self.nodes.clear();
        
        for _ in 0..self.num_trees {
            let mut indices: Vec<usize> = (0..data.len()).collect();
            self.build_tree(data, &mut indices);
        }
    }

    fn build_tree(&mut self, data: &[Vec<f32>], indices: &mut Vec<usize>) {
        if indices.len() <= self.k_means {
            for &idx in indices.iter() {
                self.nodes.push(BKTNode {
                    center_id: idx as i32,
                    child_start: -1,
                    child_end: -1,
                });
            }
            return;
        }

        // K-means clustering
        let labels = self.kmeans(data, indices);
        
        // Group by cluster
        let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); self.k_means];
        for (i, &label) in labels.iter().enumerate() {
            clusters[label].push(indices[i]);
        }

        // Find cluster centers
        let node_start = self.nodes.len();
        for cluster in &clusters {
            if cluster.is_empty() {
                continue;
            }
            
            let center_id = self.find_medoid(data, cluster);
            let node_idx = self.nodes.len();
            
            self.nodes.push(BKTNode {
                center_id: center_id as i32,
                child_start: -1,
                child_end: -1,
            });
        }

        // Recursively build children
        let mut child_start = self.nodes.len();
        for (i, mut cluster) in clusters.into_iter().enumerate() {
            if cluster.is_empty() {
                continue;
            }
            
            let node_idx = node_start + i;
            self.nodes[node_idx].child_start = self.nodes.len() as i32;
            
            self.build_tree(data, &mut cluster);
            
            self.nodes[node_idx].child_end = self.nodes.len() as i32;
        }
    }

    fn kmeans(&self, data: &[Vec<f32>], indices: &[usize]) -> Vec<usize> {
        let k = self.k_means.min(indices.len());
        let dim = data[0].len();
        let max_iters = 100;

        // Initialize centers randomly
        let mut rng = thread_rng();
        let mut center_indices: Vec<usize> = indices.to_vec();
        center_indices.shuffle(&mut rng);
        center_indices.truncate(k);

        let mut centers: Vec<Vec<f32>> = center_indices.iter()
            .map(|&i| data[i].clone())
            .collect();

        let mut labels = vec![0; indices.len()];

        for _ in 0..max_iters {
            let mut changed = false;

            // Assign to nearest center
            for (i, &idx) in indices.iter().enumerate() {
                let mut best_label = 0;
                let mut best_dist = f32::MAX;

                for (j, center) in centers.iter().enumerate() {
                    let dist = simd::l2_distance(&data[idx], center);
                    if dist < best_dist {
                        best_dist = dist;
                        best_label = j;
                    }
                }

                if labels[i] != best_label {
                    labels[i] = best_label;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update centers
            let mut new_centers = vec![vec![0.0; dim]; k];
            let mut counts = vec![0; k];

            for (i, &idx) in indices.iter().enumerate() {
                let label = labels[i];
                counts[label] += 1;
                for d in 0..dim {
                    new_centers[label][d] += data[idx][d];
                }
            }

            for j in 0..k {
                if counts[j] > 0 {
                    for d in 0..dim {
                        centers[j][d] = new_centers[j][d] / counts[j] as f32;
                    }
                }
            }
        }

        labels
    }

    fn find_medoid(&self, data: &[Vec<f32>], indices: &[usize]) -> usize {
        let mut best_idx = indices[0];
        let mut best_sum = f32::MAX;

        for &i in indices {
            let mut sum = 0.0;
            for &j in indices {
                sum += simd::l2_distance(&data[i], &data[j]);
            }
            if sum < best_sum {
                best_sum = sum;
                best_idx = i;
            }
        }

        best_idx
    }

    pub fn search(&self, query: &[f32], data: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
        let mut candidates = Vec::new();
        self.search_node(query, data, 0, &mut candidates, k);
        
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates.truncate(k);
        candidates
    }

    fn search_node(&self, query: &[f32], data: &[Vec<f32>], node_idx: usize, candidates: &mut Vec<(usize, f32)>, k: usize) {
        if node_idx >= self.nodes.len() {
            return;
        }

        let node = &self.nodes[node_idx];
        let center_id = node.center_id as usize;
        let dist = simd::l2_distance(query, &data[center_id]);
        
        candidates.push((center_id, dist));

        if node.child_start >= 0 {
            for child_idx in node.child_start as usize..node.child_end as usize {
                self.search_node(query, data, child_idx, candidates, k);
            }
        }
    }
}
