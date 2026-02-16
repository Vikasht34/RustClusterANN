/// Batch processing for large-scale index building
use std::fs::File;
use std::io::{self, Read, Write, Seek, SeekFrom};

pub struct BatchProcessor {
    tmp_file: String,
    batch_size: usize,
    current_batch: Vec<Vec<u8>>,
    total_items: usize,
}

impl BatchProcessor {
    pub fn new(tmp_dir: &str, batch_size: usize) -> Self {
        let tmp_file = format!("{}/batch_tmp_{}", tmp_dir, std::process::id());
        Self {
            tmp_file,
            batch_size,
            current_batch: Vec::new(),
            total_items: 0,
        }
    }
    
    /// Add item to batch
    pub fn add(&mut self, item: Vec<u8>) -> io::Result<()> {
        self.current_batch.push(item);
        self.total_items += 1;
        
        if self.current_batch.len() >= self.batch_size {
            self.flush()?;
        }
        
        Ok(())
    }
    
    /// Flush current batch to disk
    pub fn flush(&mut self) -> io::Result<()> {
        if self.current_batch.is_empty() {
            return Ok(());
        }
        
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.tmp_file)?;
        
        for item in &self.current_batch {
            file.write_all(&(item.len() as u32).to_le_bytes())?;
            file.write_all(item)?;
        }
        
        self.current_batch.clear();
        Ok(())
    }
    
    /// Load batch by index
    pub fn load_batch(&mut self, start: usize, end: usize) -> io::Result<Vec<Vec<u8>>> {
        let mut file = File::open(&self.tmp_file)?;
        let mut result = Vec::new();
        
        // Skip to start
        let mut current = 0;
        while current < start {
            let mut len_bytes = [0u8; 4];
            file.read_exact(&mut len_bytes)?;
            let len = u32::from_le_bytes(len_bytes) as usize;
            file.seek(SeekFrom::Current(len as i64))?;
            current += 1;
        }
        
        // Read items
        while current < end {
            let mut len_bytes = [0u8; 4];
            if file.read_exact(&mut len_bytes).is_err() {
                break;
            }
            let len = u32::from_le_bytes(len_bytes) as usize;
            
            let mut item = vec![0u8; len];
            file.read_exact(&mut item)?;
            result.push(item);
            current += 1;
        }
        
        Ok(result)
    }
    
    /// Get total number of items
    pub fn total(&self) -> usize {
        self.total_items
    }
    
    /// Clean up temporary file
    pub fn cleanup(&self) -> io::Result<()> {
        std::fs::remove_file(&self.tmp_file).ok();
        Ok(())
    }
}

impl Drop for BatchProcessor {
    fn drop(&mut self) {
        self.cleanup().ok();
    }
}
