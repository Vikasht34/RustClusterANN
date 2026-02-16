/// Fast deduplication hash table (SPTAG-style)
/// Uses open addressing with linear probing
pub struct FastDedup {
    table: Vec<u32>,
    mask: usize,
    max_check: usize,
}

impl FastDedup {
    pub fn new(max_check: usize) -> Self {
        // Size = next power of 2 after max_check * 2
        let size = (max_check * 2).next_power_of_two();
        Self {
            table: vec![0; size],
            mask: size - 1,
            max_check,
        }
    }
    
    #[inline]
    pub fn clear(&mut self) {
        self.table.fill(0);
    }
    
    #[inline]
    pub fn check_and_set(&mut self, id: usize) -> bool {
        let id = (id + 1) as u32; // 0 means empty
        let mut idx = self.hash(id);
        
        // Linear probing (max 8 attempts)
        for _ in 0..8 {
            let slot = unsafe { self.table.get_unchecked_mut(idx) };
            if *slot == 0 {
                *slot = id;
                return false; // Not seen before
            }
            if *slot == id {
                return true; // Already seen
            }
            idx = (idx + 1) & self.mask;
        }
        
        // Fallback: assume not seen
        false
    }
    
    #[inline]
    fn hash(&self, id: u32) -> usize {
        (id.wrapping_mul(99991).wrapping_add(id.rotate_left(2)).wrapping_add(101) as usize) & self.mask
    }
}
