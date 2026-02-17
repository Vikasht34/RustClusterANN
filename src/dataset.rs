/// Dataset with flat memory layout (SPTAG-style)
/// Stores vectors in contiguous memory for zero-copy operations
pub struct Dataset {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Dataset {
    /// Create from Vec<Vec<f32>> by flattening
    pub fn from_vectors(vectors: &[Vec<f32>]) -> Self {
        if vectors.is_empty() {
            return Self {
                data: Vec::new(),
                rows: 0,
                cols: 0,
            };
        }
        
        let rows = vectors.len();
        let cols = vectors[0].len();
        let mut data = Vec::with_capacity(rows * cols);
        
        for vec in vectors {
            data.extend_from_slice(vec);
        }
        
        Self { data, rows, cols }
    }
    
    /// Get reference to a row (vector) - O(1), no copying
    #[inline]
    pub fn at(&self, index: usize) -> &[f32] {
        let start = index * self.cols;
        &self.data[start..start + self.cols]
    }
    
    #[inline]
    pub fn rows(&self) -> usize {
        self.rows
    }
    
    #[inline]
    pub fn cols(&self) -> usize {
        self.cols
    }
}
