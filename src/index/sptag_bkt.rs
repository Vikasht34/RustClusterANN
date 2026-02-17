/// Complete SPTAG-BKT index: BK-Tree + TP-Trees + RNG Graph
use super::bktree_recursive::BKTreeBuilder;
use super::tptree::TPTree;
use super::rng::RNGGraph;
use crate::simd;
use crate::dataset::Dataset;

pub struct SPTAGBKTIndex {
    tree: BKTreeBuilder,           // BK-Tree for search (K=32, leaf_size=8)
    pub graph: RNGGraph,           // RNG graph for neighbor search
    data: Vec<Vec<f32>>,           // Original data
    num_tpt_trees: usize,          // Number of TP-Trees for KNNG (default: 32)
}

impl SPTAGBKTIndex {
    pub fn new(k: usize, bkt_leaf_size: usize, max_degree: usize, rng_factor: f32, cef: usize, num_tpt_trees: usize) -> Self {
        Self {
            tree: BKTreeBuilder::new(k, bkt_leaf_size, 1000),
            graph: RNGGraph::new(max_degree, rng_factor, cef),
            data: Vec::new(),
            num_tpt_trees,
        }
    }

    pub fn build(&mut self, data: Vec<Vec<f32>>, refine_iters: usize, skip_knng: bool) {
        println!("\n=== SPTAG-BKT Index Build ===");
        println!("Vectors: {}, Dim: {}", data.len(), data[0].len());
        println!("BK-Tree: K=32, leaf_size=8");
        if !skip_knng {
            println!("TP-Trees: {} trees, leaf_size=2000", self.num_tpt_trees);
            println!("RNG: max_degree={}, refine_iters={}", self.graph.max_degree, refine_iters);
        } else {
            println!("KNNG: Skipped (on-demand loading mode)");
        }
        
        let start = std::time::Instant::now();
        self.data = data;
        let n = self.data.len();
        let max_degree = self.graph.max_degree;
        
        // Convert to Dataset for zero-copy operations
        let dataset = Dataset::from_vectors(&self.data);
        
        // Step 1: Build BK-Tree for search routing
        println!("\n[1/3] Building BK-Tree for search...");
        let tree_start = std::time::Instant::now();
        self.tree.build(&dataset);
        println!("  ✓ BK-Tree built in {:.2}s", tree_start.elapsed().as_secs_f32());
        
        if skip_knng {
            println!("\n[2/3] Skipping KNNG construction (on-demand mode)");
            println!("\n[3/3] Skipping RNG refinement (on-demand mode)");
            println!("\n✓ Index built in {:.2}s ({:.2} min)", start.elapsed().as_secs_f32(), start.elapsed().as_secs_f32() / 60.0);
            return;
        }
        
        // Step 2: Build KNNG using TP-Trees
        println!("\n[2/3] Building KNNG with {} TP-Trees...", self.num_tpt_trees);
        let knng_start = std::time::Instant::now();
        
        let mut neighbors = vec![Vec::new(); n];
        let mut neighbor_dists = vec![vec![f32::MAX; max_degree]; n];
        
        let tptree = TPTree::new(2000, 1000, 5); // SPTAG defaults
        
        for tree_idx in 0..self.num_tpt_trees {
            print!("  Tree {}/{}: ", tree_idx + 1, self.num_tpt_trees);
            std::io::Write::flush(&mut std::io::stdout()).ok();
            
            // Shuffle indices for this tree
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            let mut shuffled_indices: Vec<usize> = (0..n).collect();
            shuffled_indices.shuffle(&mut rng);
            
            // Build TP-Tree
            let leaves = tptree.build(&dataset, &mut shuffled_indices);
            
            // Process leaves to build KNNG
            let mut pairs = 0;
            for &(first, last) in &leaves {
                for i in first..last {
                    for j in (i + 1)..last {
                        let p1 = shuffled_indices[i];
                        let p2 = shuffled_indices[j];
                        let dist = simd::l2_distance(&self.data[p1], &self.data[p2]);
                        
                        Self::add_neighbor(p1, p2, dist, &mut neighbors, &mut neighbor_dists, max_degree);
                        Self::add_neighbor(p2, p1, dist, &mut neighbors, &mut neighbor_dists, max_degree);
                        pairs += 1;
                    }
                }
            }
            println!("{} leaves, {} pairs", leaves.len(), pairs);
        }
        
        self.graph.neighbors = neighbors;
        
        // Check graph statistics
        let mut total_neighbors = 0;
        let mut nodes_with_neighbors = 0;
        for node_neighbors in &self.graph.neighbors {
            if !node_neighbors.is_empty() {
                nodes_with_neighbors += 1;
                total_neighbors += node_neighbors.len();
            }
        }
        println!("  ✓ KNNG built in {:.2}s", knng_start.elapsed().as_secs_f32());
        println!("    Nodes with neighbors: {}/{}", nodes_with_neighbors, n);
        println!("    Avg neighbors: {:.2}", total_neighbors as f32 / n as f32);
        
        // Step 3: Refine graph with RNG/NPA
        if refine_iters > 0 {
            println!("\n[3/3] Refining graph with RNG ({} iterations)...", refine_iters);
            let refine_start = std::time::Instant::now();
            self.graph.refine_graph(&self.data, refine_iters);
            
            let mut total = 0;
            for neighbors in &self.graph.neighbors {
                total += neighbors.len();
            }
            println!("  ✓ Graph refined in {:.2}s", refine_start.elapsed().as_secs_f32());
            println!("    Avg neighbors after refinement: {:.2}", total as f32 / n as f32);
        }
        
        println!("\n=== Build Complete in {:.2}s ===\n", start.elapsed().as_secs_f32());
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
    
    /// Search using BK-Tree only (no KNNG graph)
    pub fn search_bktree_only(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        self.tree.search(query, &self.data, k)
    }
    
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn get_data(&self) -> &[Vec<f32>] {
        &self.data
    }
    
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;
        
        let mut file = File::create(path)?;
        
        self.tree.save(&mut file)?;
        self.graph.save(&mut file)?;
        
        let num_vecs = self.data.len() as u32;
        file.write_all(&num_vecs.to_le_bytes())?;
        
        if num_vecs > 0 {
            let dim = self.data[0].len() as u32;
            file.write_all(&dim.to_le_bytes())?;
            
            for vec in &self.data {
                for &val in vec {
                    file.write_all(&val.to_le_bytes())?;
                }
            }
        }
        
        file.write_all(&(self.num_tpt_trees as u32).to_le_bytes())?;
        
        Ok(())
    }
    
    pub fn load(path: &str) -> std::io::Result<Self> {
        use std::fs::File;
        use std::io::Read;
        
        let mut file = File::open(path)?;
        
        let tree = BKTreeBuilder::load(&mut file)?;
        let graph = RNGGraph::load(&mut file)?;
        
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf)?;
        let num_vecs = u32::from_le_bytes(buf) as usize;
        
        let mut data = Vec::with_capacity(num_vecs);
        if num_vecs > 0 {
            file.read_exact(&mut buf)?;
            let dim = u32::from_le_bytes(buf) as usize;
            
            for _ in 0..num_vecs {
                let mut vec = Vec::with_capacity(dim);
                for _ in 0..dim {
                    file.read_exact(&mut buf)?;
                    vec.push(f32::from_le_bytes(buf));
                }
                data.push(vec);
            }
        }
        
        file.read_exact(&mut buf)?;
        let num_tpt_trees = u32::from_le_bytes(buf) as usize;
        
        Ok(Self {
            tree,
            graph,
            data,
            num_tpt_trees,
        })
    }
}
