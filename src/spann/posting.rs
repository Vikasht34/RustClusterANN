/// Posting List - stores vectors assigned to a cluster
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};

#[derive(Clone)]
pub struct PostingList {
    pub cluster_id: usize,
    pub vector_ids: Vec<usize>,
    pub quantized_data: Vec<u8>,  // Compressed/quantized vectors
}

impl PostingList {
    pub fn new(cluster_id: usize) -> Self {
        Self {
            cluster_id,
            vector_ids: Vec::new(),
            quantized_data: Vec::new(),
        }
    }

    pub fn add_vector(&mut self, vector_id: usize, quantized: &[u8]) {
        self.vector_ids.push(vector_id);
        self.quantized_data.extend_from_slice(quantized);
    }

    pub fn len(&self) -> usize {
        self.vector_ids.len()
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        
        // Write header
        writer.write_all(&(self.cluster_id as u32).to_le_bytes())?;
        writer.write_all(&(self.vector_ids.len() as u32).to_le_bytes())?;
        writer.write_all(&(self.quantized_data.len() as u32).to_le_bytes())?;
        
        // Write vector IDs
        for &id in &self.vector_ids {
            writer.write_all(&(id as u32).to_le_bytes())?;
        }
        
        // Write quantized data
        writer.write_all(&self.quantized_data)?;
        
        Ok(())
    }

    pub fn load(path: &str) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        
        // Read header
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        let cluster_id = u32::from_le_bytes(buf) as usize;
        
        reader.read_exact(&mut buf)?;
        let vec_count = u32::from_le_bytes(buf) as usize;
        
        reader.read_exact(&mut buf)?;
        let data_len = u32::from_le_bytes(buf) as usize;
        
        // Read vector IDs
        let mut vector_ids = Vec::with_capacity(vec_count);
        for _ in 0..vec_count {
            reader.read_exact(&mut buf)?;
            vector_ids.push(u32::from_le_bytes(buf) as usize);
        }
        
        // Read quantized data
        let mut quantized_data = vec![0u8; data_len];
        reader.read_exact(&mut quantized_data)?;
        
        Ok(Self {
            cluster_id,
            vector_ids,
            quantized_data,
        })
    }
}

pub struct PostingListManager {
    posting_lists: Vec<PostingList>,
    base_path: String,
}

impl PostingListManager {
    pub fn new(base_path: String) -> Self {
        std::fs::create_dir_all(&base_path).ok();
        Self {
            posting_lists: Vec::new(),
            base_path,
        }
    }

    pub fn add_posting_list(&mut self, posting: PostingList) {
        self.posting_lists.push(posting);
    }

    pub fn save_all(&self) -> std::io::Result<()> {
        for posting in &self.posting_lists {
            let path = format!("{}/posting_{}.bin", self.base_path, posting.cluster_id);
            posting.save(&path)?;
        }
        Ok(())
    }

    pub fn load_posting(&self, cluster_id: usize) -> std::io::Result<PostingList> {
        let path = format!("{}/posting_{}.bin", self.base_path, cluster_id);
        PostingList::load(&path)
    }

    pub fn get_posting(&self, cluster_id: usize) -> Option<&PostingList> {
        self.posting_lists.iter().find(|p| p.cluster_id == cluster_id)
    }
}
