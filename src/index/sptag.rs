/// SPTAG Index - combines KDTree/BKTree with graph-based search
use crate::index::{KDTree, BKTree, NeighborhoodGraph};
use crate::simd;

pub enum IndexType {
    KDT,
    BKT,
}

pub struct SPTAGIndex {
    data: Vec<Vec<f32>>,
    kdtree: Option<KDTree>,
    bktree: Option<BKTree>,
    graph: NeighborhoodGraph,
    index_type: IndexType,
}

impl SPTAGIndex {
    pub fn new_kdt(k_neighbors: usize, num_trees: usize) -> Self {
        Self {
            data: Vec::new(),
            kdtree: Some(KDTree::new(num_trees, 1000, 5)),
            bktree: None,
            graph: NeighborhoodGraph::new(k_neighbors),
            index_type: IndexType::KDT,
        }
    }

    pub fn new_bkt(k_neighbors: usize, num_trees: usize, k_means: usize) -> Self {
        Self {
            data: Vec::new(),
            kdtree: None,
            bktree: Some(BKTree::new(num_trees, k_means, 1000)),
            graph: NeighborhoodGraph::new(k_neighbors),
            index_type: IndexType::BKT,
        }
    }

    pub fn build(&mut self, data: Vec<Vec<f32>>) {
        self.data = data;

        // Build tree index
        match self.index_type {
            IndexType::KDT => {
                if let Some(ref mut kdtree) = self.kdtree {
                    kdtree.build(&self.data);
                }
            }
            IndexType::BKT => {
                if let Some(ref mut bktree) = self.bktree {
                    bktree.build(&self.data);
                }
            }
        }

        // Build graph
        self.graph.build(&self.data);
        self.graph.refine(&self.data, 3);
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        // Find entry points using tree
        let entry_points = match self.index_type {
            IndexType::KDT => {
                if let Some(ref kdtree) = self.kdtree {
                    kdtree.search(query, &self.data, 10)
                } else {
                    vec![(0, 0.0)]
                }
            }
            IndexType::BKT => {
                if let Some(ref bktree) = self.bktree {
                    bktree.search(query, &self.data, 10)
                } else {
                    vec![(0, 0.0)]
                }
            }
        };

        if entry_points.is_empty() {
            return Vec::new();
        }

        // Graph search from best entry point
        let entry = entry_points[0].0;
        self.graph.search(query, &self.data, k, entry, 1000)
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}
