/// Recursive BK-Tree builder (SPTAG algorithm)
use crate::index::BalancedKMeans;
use crate::simd;

#[derive(Clone)]
pub struct BKTNode {
    pub center_id: i32,
    pub child_start: i32,
    pub child_end: i32,
    pub data_start: usize,  // Start index in reordered data
    pub data_end: usize,    // End index in reordered data
}

pub struct BKTreeBuilder {
    nodes: Vec<BKTNode>,
    indices: Vec<usize>,  // Reordered indices after tree building
    leaves: Vec<(usize, usize)>,  // Leaf ranges (first, last)
    k: usize,
    leaf_size: usize,
    samples: usize,
    lambda: f32,
}

impl BKTreeBuilder {
    pub fn new(k: usize, leaf_size: usize, samples: usize) -> Self {
        Self {
            nodes: Vec::new(),
            indices: Vec::new(),
            leaves: Vec::new(),
            k,
            leaf_size,
            samples,
            lambda: -1.0,
        }
    }

    /// Build BK-Tree recursively (SPTAG algorithm)
    pub fn build(&mut self, data: &[Vec<f32>]) -> Vec<(usize, usize)> {
        println!("Building BK-Tree on {} vectors...", data.len());
        println!("  K={}, leaf_size={}, samples={}", self.k, self.leaf_size, self.samples);
        
        self.nodes.clear();
        self.leaves.clear();
        self.indices = (0..data.len()).collect();
        
        // Select lambda once for entire tree
        if self.lambda < 0.0 {
            println!("  Selecting lambda factor...");
            self.lambda = self.select_lambda(data, &self.indices.clone(), 0, data.len());
            println!("  Selected lambda: {:.3}", self.lambda);
        }
        
        // Build tree recursively
        self.build_recursive(data, 0, data.len(), 0);
        
        println!("Built BK-Tree with {} nodes, {} leaves", self.nodes.len(), self.leaves.len());
        self.leaves.clone()
    }

    pub fn get_indices(&self) -> &[usize] {
        &self.indices
    }
    
    pub fn get_node_info(&self, node_idx: usize) -> (usize, Vec<usize>) {
        if node_idx >= self.nodes.len() {
            return (0, Vec::new());
        }
        
        let node = &self.nodes[node_idx];
        let center_id = node.center_id as usize;
        
        // Get children indices
        let mut children = Vec::new();
        if node.child_start >= 0 && node.child_end > node.child_start {
            for i in node.child_start..node.child_end {
                children.push(i as usize);
            }
        }
        
        (center_id, children)
    }

    fn select_lambda(&self, data: &[Vec<f32>], indices: &[usize], first: usize, last: usize) -> f32 {
        use rand::seq::SliceRandom;
        use rand::thread_rng;
        
        let mut rng = thread_rng();
        let sample_size = self.samples.min(last - first);
        
        // Sample from partition
        let mut sample_indices: Vec<usize> = indices[first..last].to_vec();
        sample_indices.shuffle(&mut rng);
        sample_indices.truncate(sample_size);
        
        let sample_data: Vec<Vec<f32>> = sample_indices.iter()
            .map(|&i| data[i].clone())
            .collect();
        
        // Try lambda values
        let mut best_lambda = 1.0;
        let mut best_std = f32::MAX;
        
        for exp in -3..=3 {
            let lambda = 10.0_f32.powi(exp);
            let mut kmeans = BalancedKMeans::new(self.k, data[0].len());
            kmeans.fit_with_lambda(&sample_data, lambda, 20);
            
            let avg = sample_data.len() as f32 / self.k as f32;
            let variance: f32 = kmeans.counts.iter()
                .map(|&c| (c as f32 - avg).powi(2))
                .sum::<f32>() / self.k as f32;
            let std_ratio = variance.sqrt() / avg;
            
            if std_ratio < best_std {
                best_std = std_ratio;
                best_lambda = lambda;
            }
        }
        
        best_lambda
    }

    fn build_recursive(&mut self, data: &[Vec<f32>], first: usize, last: usize, depth: usize) -> i32 {
        let size = last - first;
        
        // Create node
        let node_idx = self.nodes.len() as i32;
        self.nodes.push(BKTNode {
            center_id: -1,
            child_start: -1,
            child_end: -1,
            data_start: first,
            data_end: last,
        });
        
        // Leaf node: store all vectors (or max depth reached)
        if size <= self.leaf_size || depth >= 50 {
            if depth >= 50 {
                println!("  Depth {}: Max depth reached, creating leaf with {} vectors", depth, size);
            } else {
                println!("  Depth {}: Leaf with {} vectors", depth, size);
            }
            self.nodes[node_idx as usize].center_id = self.indices[first] as i32;
            self.leaves.push((first, last));
            return node_idx;
        }
        
        println!("  Depth {}: Clustering {} vectors into {} clusters", depth, size, self.k);
        
        // Cluster this partition
        let partition_data: Vec<Vec<f32>> = self.indices[first..last].iter()
            .map(|&i| data[i].clone())
            .collect();
        
        let mut kmeans = BalancedKMeans::new(self.k, data[0].len());
        
        // Use samples for training
        use rand::seq::SliceRandom;
        use rand::thread_rng;
        let mut rng = thread_rng();
        
        let sample_size = self.samples.min(partition_data.len());
        let mut sample_indices_local: Vec<usize> = (0..partition_data.len()).collect();
        sample_indices_local.shuffle(&mut rng);
        sample_indices_local.truncate(sample_size);
        
        let sample_data: Vec<Vec<f32>> = sample_indices_local.iter()
            .map(|&i| partition_data[i].clone())
            .collect();
        
        // Train on samples
        kmeans.fit_with_lambda(&sample_data, self.lambda, 100);
        
        // Assign ALL vectors in partition to clusters
        let mut assignments = vec![0; partition_data.len()];
        let mut counts = vec![0; self.k];
        
        for (i, vec) in partition_data.iter().enumerate() {
            let mut best_cluster = 0;
            let mut best_dist = f32::MAX;
            
            for j in 0..self.k {
                let dist = simd::l2_distance(vec, &kmeans.centers[j]);
                if dist < best_dist {
                    best_dist = dist;
                    best_cluster = j;
                }
            }
            
            assignments[i] = best_cluster;
            counts[best_cluster] += 1;
        }
        
        // Reorder indices by cluster (SPTAG Shuffle)
        let mut reordered = Vec::with_capacity(partition_data.len());
        for cluster in 0..self.k {
            for i in 0..partition_data.len() {
                if assignments[i] == cluster {
                    reordered.push(self.indices[first + i]);
                }
            }
        }
        self.indices[first..last].copy_from_slice(&reordered);
        
        // Set center as first vector of largest cluster
        let max_cluster = counts.iter().enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| i)
            .unwrap_or(0);
        self.nodes[node_idx as usize].center_id = self.indices[first] as i32;
        
        // Recursively build children
        self.nodes[node_idx as usize].child_start = self.nodes.len() as i32;
        
        let mut offset = first;
        for cluster in 0..self.k {
            if counts[cluster] == 0 {
                continue;
            }
            
            if counts[cluster] > 1 {
                self.build_recursive(data, offset, offset + counts[cluster], depth + 1);
            }
            
            offset += counts[cluster];
        }
        
        self.nodes[node_idx as usize].child_end = self.nodes.len() as i32;
        
        node_idx
    }

    /// Search tree for k nearest neighbors using greedy descent
    pub fn search(&self, query: &[f32], data: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
        if self.nodes.is_empty() || data.is_empty() {
            return Vec::new();
        }
        
        let mut results = Vec::new();
        
        // Greedy descent to find best leaf
        let mut node_idx = 0;
        loop {
            if node_idx >= self.nodes.len() {
                break;
            }
            
            let node = &self.nodes[node_idx];
            
            // Add center to results
            if node.center_id >= 0 {
                let center_id = node.center_id as usize;
                if center_id < data.len() {
                    let dist = simd::l2_distance(query, &data[center_id]);
                    results.push((center_id, dist));
                }
            }
            
            // If leaf, we're done
            if node.child_start < 0 {
                break;
            }
            
            // Find closest child
            let child_start = node.child_start as usize;
            let child_end = node.child_end as usize;
            
            if child_start >= self.nodes.len() {
                break;
            }
            
            let mut best_child = child_start;
            let mut best_dist = f32::MAX;
            
            for child_idx in child_start..child_end.min(self.nodes.len()) {
                let child = &self.nodes[child_idx];
                if child.center_id >= 0 {
                    let center_id = child.center_id as usize;
                    if center_id < data.len() {
                        let dist = simd::l2_distance(query, &data[center_id]);
                        if dist < best_dist {
                            best_dist = dist;
                            best_child = child_idx;
                        }
                    }
                }
            }
            
            node_idx = best_child;
        }
        
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }
    
    /// Save BKTree structure to binary format (SPTAG-compatible)
    pub fn save(&self, writer: &mut dyn std::io::Write) -> std::io::Result<()> {
        use std::io::Write;
        
        // Write number of nodes
        let num_nodes = self.nodes.len() as u32;
        writer.write_all(&num_nodes.to_le_bytes())?;
        
        // Write nodes
        for node in &self.nodes {
            writer.write_all(&node.center_id.to_le_bytes())?;
            writer.write_all(&node.child_start.to_le_bytes())?;
            writer.write_all(&node.child_end.to_le_bytes())?;
            writer.write_all(&(node.data_start as u32).to_le_bytes())?;
            writer.write_all(&(node.data_end as u32).to_le_bytes())?;
        }
        
        // Write indices
        let num_indices = self.indices.len() as u32;
        writer.write_all(&num_indices.to_le_bytes())?;
        for &idx in &self.indices {
            writer.write_all(&(idx as u32).to_le_bytes())?;
        }
        
        // Write leaves
        let num_leaves = self.leaves.len() as u32;
        writer.write_all(&num_leaves.to_le_bytes())?;
        for &(start, end) in &self.leaves {
            writer.write_all(&(start as u32).to_le_bytes())?;
            writer.write_all(&(end as u32).to_le_bytes())?;
        }
        
        // Write parameters
        writer.write_all(&(self.k as u32).to_le_bytes())?;
        writer.write_all(&(self.leaf_size as u32).to_le_bytes())?;
        writer.write_all(&(self.samples as u32).to_le_bytes())?;
        writer.write_all(&self.lambda.to_le_bytes())?;
        
        Ok(())
    }
    
    /// Load BKTree structure from binary format
    pub fn load(reader: &mut dyn std::io::Read) -> std::io::Result<Self> {
        use std::io::Read;
        
        // Read number of nodes
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        let num_nodes = u32::from_le_bytes(buf) as usize;
        
        // Read nodes
        let mut nodes = Vec::with_capacity(num_nodes);
        for _ in 0..num_nodes {
            reader.read_exact(&mut buf)?;
            let center_id = i32::from_le_bytes(buf);
            
            reader.read_exact(&mut buf)?;
            let child_start = i32::from_le_bytes(buf);
            
            reader.read_exact(&mut buf)?;
            let child_end = i32::from_le_bytes(buf);
            
            reader.read_exact(&mut buf)?;
            let data_start = u32::from_le_bytes(buf) as usize;
            
            reader.read_exact(&mut buf)?;
            let data_end = u32::from_le_bytes(buf) as usize;
            
            nodes.push(BKTNode {
                center_id,
                child_start,
                child_end,
                data_start,
                data_end,
            });
        }
        
        // Read indices
        reader.read_exact(&mut buf)?;
        let num_indices = u32::from_le_bytes(buf) as usize;
        let mut indices = Vec::with_capacity(num_indices);
        for _ in 0..num_indices {
            reader.read_exact(&mut buf)?;
            indices.push(u32::from_le_bytes(buf) as usize);
        }
        
        // Read leaves
        reader.read_exact(&mut buf)?;
        let num_leaves = u32::from_le_bytes(buf) as usize;
        let mut leaves = Vec::with_capacity(num_leaves);
        for _ in 0..num_leaves {
            reader.read_exact(&mut buf)?;
            let start = u32::from_le_bytes(buf) as usize;
            reader.read_exact(&mut buf)?;
            let end = u32::from_le_bytes(buf) as usize;
            leaves.push((start, end));
        }
        
        // Read parameters
        reader.read_exact(&mut buf)?;
        let k = u32::from_le_bytes(buf) as usize;
        
        reader.read_exact(&mut buf)?;
        let leaf_size = u32::from_le_bytes(buf) as usize;
        
        reader.read_exact(&mut buf)?;
        let samples = u32::from_le_bytes(buf) as usize;
        
        reader.read_exact(&mut buf)?;
        let lambda = f32::from_le_bytes(buf);
        
        Ok(Self {
            nodes,
            indices,
            leaves,
            k,
            leaf_size,
            samples,
            lambda,
        })
    }
}
