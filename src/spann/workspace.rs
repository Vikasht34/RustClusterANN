/// Workspace pooling for reusing buffers across queries
use crossbeam::queue::ArrayQueue;
use std::sync::Arc;

pub struct SearchWorkspace {
    pub decompress_buffer: Vec<u8>,
    pub decode_buffer: Vec<f32>,
    pub result_buffer: Vec<(usize, f32)>,
}

impl SearchWorkspace {
    pub fn new(max_posting_size: usize, dim: usize, k: usize) -> Self {
        Self {
            decompress_buffer: Vec::with_capacity(max_posting_size),
            decode_buffer: vec![0.0; dim],
            result_buffer: Vec::with_capacity(k * 10),
        }
    }
    
    pub fn clear(&mut self) {
        self.decompress_buffer.clear();
        self.result_buffer.clear();
    }
}

pub struct WorkspacePool {
    workspaces: Vec<SearchWorkspace>,
    free_ids: Arc<ArrayQueue<usize>>,
    max_posting_size: usize,
    dim: usize,
    k: usize,
}

impl WorkspacePool {
    pub fn new(num_threads: usize, max_posting_size: usize, dim: usize, k: usize) -> Self {
        let mut workspaces = Vec::with_capacity(num_threads);
        let free_ids = Arc::new(ArrayQueue::new(num_threads));
        
        for i in 0..num_threads {
            workspaces.push(SearchWorkspace::new(max_posting_size, dim, k));
            free_ids.push(i).ok();
        }
        
        Self {
            workspaces,
            free_ids,
            max_posting_size,
            dim,
            k,
        }
    }
    
    pub fn acquire(&mut self) -> (usize, &mut SearchWorkspace) {
        let id = self.free_ids.pop().unwrap_or_else(|| {
            // Allocate new workspace if pool exhausted
            let id = self.workspaces.len();
            self.workspaces.push(SearchWorkspace::new(self.max_posting_size, self.dim, self.k));
            id
        });
        
        let workspace = &mut self.workspaces[id];
        workspace.clear();
        (id, workspace)
    }
    
    pub fn release(&self, id: usize) {
        self.free_ids.push(id).ok();
    }
}
