/// Zstd dictionary training for better compression
use std::io;

pub struct ZstdCompressor {
    dict: Option<Vec<u8>>,
    level: i32,
}

impl ZstdCompressor {
    pub fn new(level: i32) -> Self {
        Self {
            dict: None,
            level,
        }
    }
    
    /// Train dictionary from sample posting lists
    pub fn train_dict(&mut self, samples: &[Vec<u8>], dict_size: usize) -> io::Result<()> {
        if samples.is_empty() {
            return Ok(());
        }
        
        // Concatenate samples
        let total_size: usize = samples.iter().map(|s| s.len()).sum();
        let mut buffer = Vec::with_capacity(total_size);
        let mut sizes = Vec::with_capacity(samples.len());
        
        for sample in samples {
            buffer.extend_from_slice(sample);
            sizes.push(sample.len());
        }
        
        // Train dictionary
        self.dict = Some(zstd::dict::from_continuous(&buffer, &sizes, dict_size)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?);
        
        Ok(())
    }
    
    /// Compress with dictionary if available
    pub fn compress(&self, data: &[u8]) -> io::Result<Vec<u8>> {
        if let Some(ref dict) = self.dict {
            let mut encoder = zstd::bulk::Compressor::with_dictionary(self.level, dict)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
            encoder.compress(data)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
        } else {
            zstd::encode_all(data, self.level)
        }
    }
    
    /// Decompress with dictionary if available
    pub fn decompress(&self, data: &[u8]) -> io::Result<Vec<u8>> {
        if let Some(ref dict) = self.dict {
            let mut decoder = zstd::bulk::Decompressor::with_dictionary(dict)
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
            decoder.decompress(data, 10 * 1024 * 1024) // 10MB max
                .map_err(|e| io::Error::new(io::ErrorKind::Other, e))
        } else {
            zstd::decode_all(data)
        }
    }
    
    /// Get dictionary for saving
    pub fn get_dict(&self) -> Option<&[u8]> {
        self.dict.as_deref()
    }
    
    /// Load dictionary
    pub fn set_dict(&mut self, dict: Vec<u8>) {
        self.dict = Some(dict);
    }
}
