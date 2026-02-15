/// K-Nearest Neighborhood Graph (KNNG) implementation
use crate::simd;
use std::collections::{BinaryHeap, HashSet};
use std::cmp::Ordering;

#[derive(Clone)]
struct HeapNode {
    id: usize,
    dist: f32,
}

impl PartialEq for HeapNode {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}

impl Eq for HeapNode {}

impl PartialOrd for HeapNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.dist.partial_cmp(&self.dist)
    }
}

impl Ord for HeapNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

pub struct NeighborhoodGraph {
    graph: Vec<Vec<usize>>,
    k: usize,
}

impl NeighborhoodGraph {
    pub fn new(k: usize) -> Self {
        Self {
            graph: Vec::new(),
            k,
        }
    }

    pub fn build(&mut self, data: &[Vec<f32>]) {
        let n = data.len();
        self.graph = vec![Vec::new(); n];

        for i in 0..n {
            let mut heap = BinaryHeap::new();
            
            for j in 0..n {
                if i == j {
                    continue;
                }

                let dist = simd::l2_distance(&data[i], &data[j]);
                heap.push(HeapNode { id: j, dist });

                if heap.len() > self.k {
                    heap.pop();
                }
            }

            self.graph[i] = heap.into_iter().map(|node| node.id).collect();
        }
    }

    pub fn refine(&mut self, data: &[Vec<f32>], iterations: usize) {
        for _ in 0..iterations {
            let mut new_graph = self.graph.clone();

            for i in 0..data.len() {
                let mut candidates = HashSet::new();
                
                // Add current neighbors
                for &neighbor in &self.graph[i] {
                    candidates.insert(neighbor);
                    
                    // Add neighbors of neighbors
                    for &nn in &self.graph[neighbor] {
                        if nn != i {
                            candidates.insert(nn);
                        }
                    }
                }

                // Find k nearest from candidates
                let mut heap = BinaryHeap::new();
                for &candidate in &candidates {
                    let dist = simd::l2_distance(&data[i], &data[candidate]);
                    heap.push(HeapNode { id: candidate, dist });

                    if heap.len() > self.k {
                        heap.pop();
                    }
                }

                new_graph[i] = heap.into_iter().map(|node| node.id).collect();
            }

            self.graph = new_graph;
        }
    }

    pub fn get_neighbors(&self, node: usize) -> &[usize] {
        &self.graph[node]
    }

    pub fn search(&self, query: &[f32], data: &[Vec<f32>], k: usize, entry_point: usize, max_checks: usize) -> Vec<(usize, f32)> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut result = BinaryHeap::new();

        let dist = simd::l2_distance(query, &data[entry_point]);
        candidates.push(HeapNode { id: entry_point, dist });
        result.push(HeapNode { id: entry_point, dist });
        visited.insert(entry_point);

        let mut checks = 0;

        while !candidates.is_empty() && checks < max_checks {
            let current = candidates.pop().unwrap();

            if result.len() >= k && current.dist > result.peek().unwrap().dist {
                break;
            }

            for &neighbor in &self.graph[current.id] {
                if visited.contains(&neighbor) {
                    continue;
                }
                visited.insert(neighbor);
                checks += 1;

                let dist = simd::l2_distance(query, &data[neighbor]);
                let node = HeapNode { id: neighbor, dist };

                if result.len() < k || dist < result.peek().unwrap().dist {
                    candidates.push(node.clone());
                    result.push(node);

                    if result.len() > k {
                        result.pop();
                    }
                }
            }
        }

        let mut output: Vec<(usize, f32)> = result.into_iter()
            .map(|node| (node.id, node.dist))
            .collect();
        output.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        output.truncate(k);
        output
    }
}
