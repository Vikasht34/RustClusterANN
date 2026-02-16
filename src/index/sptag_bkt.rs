/// Complete SPTAG index with BK-Tree + KNNG + NPA
use super::bktree_recursive::BKTreeBuilder;
use super::rng::RNGGraph;
use crate::simd;

pub struct SPTAGBKTIndex {
    tree: BKTreeBuilder,
    pub graph: RNGGraph,
    data: Vec<Vec<f32>>,
    num_trees: usize,  // Number of trees to build (SPTAG default: 32)
}

impl SPTAGBKTIndex {
    pub fn new(k: usize, leaf_size: usize, max_degree: usize, rng_factor: f32, cef: usize, num_trees: usize) -> Self {
        Self {
            tree: BKTreeBuilder::new(k, leaf_size, 1000),
            graph: RNGGraph::new(max_degree, rng_factor, cef),
            data: Vec::new(),
            num_trees,
        }
    }

    pub fn build(&mut self, data: Vec<Vec<f32>>, refine_iters: usize) {
        self.build_with_verbosity(data, refine_iters, true);
    }
    
    pub fn build_quiet(&mut self, data: Vec<Vec<f32>>, refine_iters: usize) {
        self.build_with_verbosity(data, refine_iters, false);
    }
    
    fn build_with_verbosity(&mut self, data: Vec<Vec<f32>>, refine_iters: usize, verbose: bool) {
        if verbose {
            println!("Building SPTAG-BKT index for {} vectors...", data.len());
            println!("Using {} trees for better coverage", self.num_trees);
        }
        
        let start = std::time::Instant::now();
        self.data = data;
        
        let n = self.data.len();
        let max_degree = self.graph.max_degree;
        
        // Initialize graph
        let mut neighbors = vec![Vec::new(); n];
        let mut neighbor_dists = vec![vec![f32::MAX; max_degree]; n];
        
        // Build multiple trees and accumulate neighbors
        for tree_idx in 0..self.num_trees {
            println!("\n{}. Building Tree {}/{}...", tree_idx + 1, tree_idx + 1, self.num_trees);
            let tree_start = std::time::Instant::now();
            
            // Shuffle indices for this tree
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            let mut shuffled_data: Vec<usize> = (0..n).collect();
            shuffled_data.shuffle(&mut rng);
            
            // Create shuffled view
            let shuffled_vecs: Vec<Vec<f32>> = shuffled_data.iter()
                .map(|&i| self.data[i].clone())
                .collect();
            
            // Build tree on shuffled data
            let leaves = self.tree.build(&shuffled_vecs);
            let indices = self.tree.get_indices();
            
            println!("   Tree built in {:.2}s with {} leaves", tree_start.elapsed().as_secs_f32(), leaves.len());
            
            // Build KNNG from leaves
            let mut pairs = 0;
            for &(first, last) in &leaves {
                for i in first..last {
                    for j in (i + 1)..last {
                        // Map back to original indices
                        let p1 = shuffled_data[indices[i]];
                        let p2 = shuffled_data[indices[j]];
                        let dist = simd::l2_distance(&self.data[p1], &self.data[p2]);
                        
                        Self::add_neighbor(p1, p2, dist, &mut neighbors, &mut neighbor_dists, max_degree);
                        Self::add_neighbor(p2, p1, dist, &mut neighbors, &mut neighbor_dists, max_degree);
                        pairs += 1;
                    }
                }
            }
            println!("   Computed {} pairs", pairs);
        }
        
        self.graph.neighbors = neighbors;
        
        // Debug: Check graph statistics
        let mut total_neighbors = 0;
        let mut min_neighbors = usize::MAX;
        let mut max_neighbors = 0;
        let mut nodes_with_neighbors = 0;
        
        for node_neighbors in &self.graph.neighbors {
            let count = node_neighbors.len();
            if count > 0 {
                nodes_with_neighbors += 1;
                total_neighbors += count;
                min_neighbors = min_neighbors.min(count);
                max_neighbors = max_neighbors.max(count);
            }
        }
        
        println!("\nKNNG Statistics:");
        println!("  Nodes with neighbors: {}/{}", nodes_with_neighbors, n);
        println!("  Avg neighbors: {:.2}", total_neighbors as f32 / n as f32);
        println!("  Min neighbors: {}", if min_neighbors == usize::MAX { 0 } else { min_neighbors });
        println!("  Max neighbors: {}", max_neighbors);
        
        // Step 3: Refine with NPA
        if refine_iters > 0 {
            println!("\nRefining graph with NPA...");
            let refine_start = std::time::Instant::now();
            self.graph.refine_graph(&self.data, refine_iters);
            
            // Check graph after NPA
            let mut total = 0;
            let mut min = usize::MAX;
            let mut max = 0;
            for neighbors in &self.graph.neighbors {
                let count = neighbors.len();
                total += count;
                if count > 0 {
                    min = min.min(count);
                    max = max.max(count);
                }
            }
            println!("  After NPA: avg={:.2}, min={}, max={}", 
                     total as f32 / n as f32, 
                     if min == usize::MAX { 0 } else { min }, 
                     max);
            println!("  Graph refined in {:.2}s", refine_start.elapsed().as_secs_f32());
        }
        
        println!("\nTotal build time: {:.2}s", start.elapsed().as_secs_f32());
    }

    fn add_neighbor(
        node: usize,
        neighbor: usize,
        dist: f32,
        neighbors: &mut [Vec<usize>],
        neighbor_dists: &mut [Vec<f32>],
        max_degree: usize,
    ) {
        // Check if neighbor already exists
        if neighbors[node].contains(&neighbor) {
            return;
        }
        
        // Find insertion position
        let mut pos = max_degree;
        for i in 0..max_degree {
            if dist < neighbor_dists[node][i] {
                pos = i;
                break;
            }
        }
        
        if pos < max_degree {
            // Shift distances
            for i in (pos + 1..max_degree).rev() {
                neighbor_dists[node][i] = neighbor_dists[node][i - 1];
            }
            neighbor_dists[node][pos] = dist;
            
            // Insert neighbor
            if neighbors[node].len() < max_degree {
                neighbors[node].insert(pos, neighbor);
            } else {
                // Shift neighbors
                for i in (pos + 1..max_degree).rev() {
                    neighbors[node][i] = neighbors[node][i - 1];
                }
                neighbors[node][pos] = neighbor;
            }
        }
    }

    pub fn search(&self, query: &[f32], k: usize, max_checks: usize) -> Vec<(usize, f32)> {
        // Use 50 random entry points (SPTAG default)
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut entry_points = Vec::new();
        
        let num_entry_points = 50.min(self.data.len());
        for _ in 0..num_entry_points {
            let id = rng.gen_range(0..self.data.len());
            let dist = simd::l2_distance(query, &self.data[id]);
            entry_points.push((id, dist));
        }
        
        self.graph.search(query, &self.data, k, &entry_points, max_checks)
    }
    
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn get_data(&self) -> &[Vec<f32>] {
        &self.data
    }
}
