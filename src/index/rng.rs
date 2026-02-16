/// Relative Neighborhood Graph (RNG) with NPA refinement
use crate::simd;
use std::collections::{BinaryHeap, HashSet};
use std::cmp::Ordering;

#[derive(Clone)]
struct Candidate {
    id: usize,
    dist: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.dist.partial_cmp(&self.dist) // Reverse for min-heap
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

pub struct RNGGraph {
    pub neighbors: Vec<Vec<usize>>,
    pub max_degree: usize,
    pub rng_factor: f32,
    pub cef: usize,
}

impl RNGGraph {
    pub fn new(max_degree: usize, rng_factor: f32, cef: usize) -> Self {
        Self {
            neighbors: Vec::new(),
            max_degree,
            rng_factor,
            cef,
        }
    }

    pub fn build_from_tree<F>(&mut self, data: &[Vec<f32>], tree_search: F)
    where
        F: Fn(&[f32]) -> Vec<(usize, f32)>,
    {
        let n = data.len();
        self.neighbors = vec![Vec::new(); n];

        for i in 0..n {
            let candidates = tree_search(&data[i]);
            self.rebuild_neighbors(i, &candidates, data);
        }
    }

    pub fn rebuild_neighbors(&mut self, node: usize, candidates: &[(usize, f32)], data: &[Vec<f32>]) {
        let mut neighbors = Vec::new();
        let mut rejected = 0;

        for &(j, dist_j) in candidates {
            if j == node {
                continue;
            }
            if neighbors.len() >= self.max_degree {
                break;
            }

            let mut is_rng = true;
            let vec_j: &Vec<f32> = <[Vec<f32>]>::get(data, j).unwrap();
            for &k in &neighbors {
                let vec_k: &Vec<f32> = <[Vec<f32>]>::get(data, k).unwrap();
                let dist_kj: f32 = simd::l2_distance(vec_k.as_slice(), vec_j.as_slice());
                if self.rng_factor * dist_kj < dist_j {
                    is_rng = false;
                    rejected += 1;
                    break;
                }
            }

            if is_rng {
                neighbors.push(j);
            }
        }
        
        if node == 0 {
            eprintln!("RebuildNeighbors node 0: {} candidates, {} accepted, {} rejected", 
                     candidates.len(), neighbors.len(), rejected);
        }

        self.neighbors[node] = neighbors;
    }

    pub fn refine_graph(&mut self, data: &[Vec<f32>], iters: usize) {
        println!("Refining graph with {} iterations...", iters);
        
        for iter in 0..iters {
            let start = std::time::Instant::now();
            let cef = if iter < iters - 1 { self.cef * 4 } else { self.cef };
            
            println!("  Iter {}: CEF={}, searching from each node...", iter, cef);
            
            for i in 0..data.len() {
                // Check neighbors before refinement
                let before_count = self.neighbors[i].len();
                
                // SPTAG: Search from node i itself
                let entry_points = vec![(i, 0.0)];
                let candidates = self.search(&data[i], data, cef, &entry_points, cef * 2);
                
                if i == 0 {
                    println!("    Node 0: {} neighbors before, {} candidates found", before_count, candidates.len());
                }
                
                self.rebuild_neighbors(i, &candidates, data);
                
                if i == 0 {
                    println!("    Node 0: {} neighbors after RNG filtering", self.neighbors[i].len());
                }
                
                if i % (data.len() / 10).max(1) == 0 {
                    println!("  Iter {}: {:.1}%", iter, i as f32 / data.len() as f32 * 100.0);
                }
            }
            
            println!("  Iter {} completed in {:.2}s", iter, start.elapsed().as_secs_f32());
        }
    }

    fn search_internal(&self, query: &[f32], data: &[Vec<f32>], k: usize, start: usize) -> Vec<(usize, f32)> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut results = Vec::new();

        let dist = simd::l2_distance(query, &data[start]);
        candidates.push(Candidate { id: start, dist });
        visited.insert(start);

        while let Some(Candidate { id: current, dist: current_dist }) = candidates.pop() {
            results.push((current, current_dist));
            
            if results.len() >= k {
                break;
            }

            for &neighbor in &self.neighbors[current] {
                if visited.contains(&neighbor) {
                    continue;
                }
                visited.insert(neighbor);

                let dist = simd::l2_distance(query, &data[neighbor]);
                candidates.push(Candidate { id: neighbor, dist });
            }
        }

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }

    pub fn search(&self, query: &[f32], data: &[Vec<f32>], k: usize, entry_points: &[(usize, f32)], max_checks: usize) -> Vec<(usize, f32)> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut checks = 0;

        for &(id, dist) in entry_points {
            if !visited.contains(&id) {
                candidates.push(Candidate { id, dist });
                visited.insert(id);
            }
        }

        let mut results = Vec::new();
        let mut expansions = 0;
        let mut loop_count = 0;

        while let Some(Candidate { id: current, dist: current_dist }) = candidates.pop() {
            loop_count += 1;
            results.push((current, current_dist));
            checks += 1;

            if results.len() >= k || checks >= max_checks {
                break;
            }

            let neighbor_count = self.neighbors[current].len();
            let mut added = 0;
            let mut skipped_visited = 0;
            for &neighbor in &self.neighbors[current] {
                if visited.contains(&neighbor) {
                    skipped_visited += 1;
                    continue;
                }
                visited.insert(neighbor);

                let dist = simd::l2_distance(query, &data[neighbor]);
                candidates.push(Candidate { id: neighbor, dist });
                added += 1;
                expansions += 1;
            }
            
            // Debug first search
            if loop_count == 1 {
                eprintln!("Loop 1: node {} has {} neighbors, added={}, skipped_visited={}, candidates.len()={}", 
                         current, neighbor_count, added, skipped_visited, candidates.len());
                eprintln!("  First 10 neighbors: {:?}", &self.neighbors[current][..10.min(neighbor_count)]);
                eprintln!("  Visited set size: {}", visited.len());
            }
            if loop_count == 2 {
                eprintln!("Loop 2: node {} has {} neighbors, added={}, skipped_visited={}, candidates.len()={}", 
                         current, neighbor_count, added, skipped_visited, candidates.len());
            }
        }
        
        if results.len() < 10 {
            eprintln!("Search ended: {} results, {} loops, {} checks, {} expansions, candidates_empty={}", 
                     results.len(), loop_count, checks, expansions, loop_count == results.len());
        }

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }

    pub fn get_neighbors(&self, node: usize) -> &[usize] {
        &self.neighbors[node]
    }
    
    /// Save graph to binary format
    pub fn save(&self, writer: &mut dyn std::io::Write) -> std::io::Result<()> {
        use std::io::Write;
        
        // Write parameters
        writer.write_all(&(self.max_degree as u32).to_le_bytes())?;
        writer.write_all(&self.rng_factor.to_le_bytes())?;
        writer.write_all(&(self.cef as u32).to_le_bytes())?;
        
        // Write neighbors
        let num_nodes = self.neighbors.len() as u32;
        writer.write_all(&num_nodes.to_le_bytes())?;
        
        for neighbors in &self.neighbors {
            let count = neighbors.len() as u32;
            writer.write_all(&count.to_le_bytes())?;
            for &neighbor in neighbors {
                writer.write_all(&(neighbor as u32).to_le_bytes())?;
            }
        }
        
        Ok(())
    }
    
    /// Load graph from binary format
    pub fn load(reader: &mut dyn std::io::Read) -> std::io::Result<Self> {
        use std::io::Read;
        
        let mut buf = [0u8; 4];
        
        // Read parameters
        reader.read_exact(&mut buf)?;
        let max_degree = u32::from_le_bytes(buf) as usize;
        
        reader.read_exact(&mut buf)?;
        let rng_factor = f32::from_le_bytes(buf);
        
        reader.read_exact(&mut buf)?;
        let cef = u32::from_le_bytes(buf) as usize;
        
        // Read neighbors
        reader.read_exact(&mut buf)?;
        let num_nodes = u32::from_le_bytes(buf) as usize;
        
        let mut neighbors = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            reader.read_exact(&mut buf)?;
            let count = u32::from_le_bytes(buf) as usize;
            
            let mut node_neighbors = Vec::with_capacity(count);
            for _ in 0..count {
                reader.read_exact(&mut buf)?;
                node_neighbors.push(u32::from_le_bytes(buf) as usize);
            }
            neighbors.push(node_neighbors);
        }
        
        Ok(Self {
            neighbors,
            max_degree,
            rng_factor,
            cef,
        })
    }
}
