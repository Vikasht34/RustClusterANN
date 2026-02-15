/// Head Index - small in-memory graph on cluster centers
use crate::index::NeighborhoodGraph;
use crate::simd;

pub struct HeadIndex {
    centers: Vec<Vec<f32>>,
    center_ids: Vec<usize>,
    graph: NeighborhoodGraph,
}

impl HeadIndex {
    pub fn new(k_neighbors: usize) -> Self {
        Self {
            centers: Vec::new(),
            center_ids: Vec::new(),
            graph: NeighborhoodGraph::new(k_neighbors),
        }
    }

    pub fn build_from_kmeans(vectors: &[Vec<f32>], num_clusters: usize) -> Self {
        let mut head = Self::new(32);
        
        // K-means clustering to find centers
        let (centers, assignments) = kmeans(vectors, num_clusters);
        
        // Store centers and their IDs
        head.centers = centers.clone();
        head.center_ids = (0..centers.len()).collect();
        
        // Build graph on centers
        head.graph.build(&centers);
        head.graph.refine(&centers, 3);
        
        head
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<usize> {
        if self.centers.is_empty() {
            return Vec::new();
        }
        
        let results = self.graph.search(query, &self.centers, k, 0, 1000);
        // Return cluster IDs (which are just indices 0..num_clusters)
        results.into_iter().map(|(idx, _)| idx).collect()
    }

    pub fn len(&self) -> usize {
        self.centers.len()
    }

    pub fn get_center(&self, idx: usize) -> &[f32] {
        &self.centers[idx]
    }
}

fn kmeans(vectors: &[Vec<f32>], k: usize) -> (Vec<Vec<f32>>, Vec<usize>) {
    use rand::seq::SliceRandom;
    use rand::thread_rng;
    
    let dim = vectors[0].len();
    let max_iters = 100;
    
    // Initialize centers randomly
    let mut rng = thread_rng();
    let mut center_indices: Vec<usize> = (0..vectors.len()).collect();
    center_indices.shuffle(&mut rng);
    center_indices.truncate(k);
    
    let mut centers: Vec<Vec<f32>> = center_indices.iter()
        .map(|&i| vectors[i].clone())
        .collect();
    
    let mut assignments = vec![0; vectors.len()];
    
    for _ in 0..max_iters {
        let mut changed = false;
        
        // Assign to nearest center
        for (i, vec) in vectors.iter().enumerate() {
            let mut best_center = 0;
            let mut best_dist = f32::MAX;
            
            for (j, center) in centers.iter().enumerate() {
                let dist = simd::l2_distance(vec, center);
                if dist < best_dist {
                    best_dist = dist;
                    best_center = j;
                }
            }
            
            if assignments[i] != best_center {
                assignments[i] = best_center;
                changed = true;
            }
        }
        
        if !changed {
            break;
        }
        
        // Update centers
        let mut new_centers = vec![vec![0.0; dim]; k];
        let mut counts = vec![0; k];
        
        for (i, vec) in vectors.iter().enumerate() {
            let cluster = assignments[i];
            counts[cluster] += 1;
            for d in 0..dim {
                new_centers[cluster][d] += vec[d];
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
    
    (centers, assignments)
}
