use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Arc;
use tokio::task;

use super::storage::{ListInfo, PAGE_SIZE};

pub struct AsyncStorage {
    path: String,
    list_infos: Arc<Vec<ListInfo>>,
    enable_compression: bool,
    enable_delta: bool,
}

impl AsyncStorage {
    pub fn new(path: String, list_infos: Vec<ListInfo>, enable_compression: bool, enable_delta: bool) -> Self {
        Self {
            path,
            list_infos: Arc::new(list_infos),
            enable_compression,
            enable_delta,
        }
    }

    /// Load single posting list asynchronously
    pub async fn load_posting(&self, posting_id: usize) -> io::Result<Vec<u8>> {
        let path = self.path.clone();
        let info = self.list_infos[posting_id].clone();
        let enable_compression = self.enable_compression;

        task::spawn_blocking(move || {
            let mut file = File::open(&path)?;
            
            // Seek to offset
            file.seek(SeekFrom::Start(info.offset))?;
            
            // Read compressed data
            let mut buffer = vec![0u8; info.total_bytes as usize];
            file.read_exact(&mut buffer)?;
            
            // Decompress if needed
            if enable_compression {
                zstd::decode_all(&buffer[..])
            } else {
                Ok(buffer)
            }
        })
        .await
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?
    }

    /// Load multiple posting lists in parallel
    pub async fn load_postings_batch(&self, posting_ids: &[usize]) -> io::Result<Vec<Vec<u8>>> {
        let futures: Vec<_> = posting_ids
            .iter()
            .map(|&id| self.load_posting(id))
            .collect();

        futures::future::try_join_all(futures).await
    }

    /// Load posting list with Direct I/O
    pub async fn load_posting_direct(&self, posting_id: usize) -> io::Result<Vec<u8>> {
        let path = self.path.clone();
        let info = self.list_infos[posting_id].clone();
        let enable_compression = self.enable_compression;

        task::spawn_blocking(move || {
            #[cfg(target_os = "linux")]
            {
                use std::fs::OpenOptions;
                use std::os::unix::fs::OpenOptionsExt;
                
                let mut file = OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_DIRECT)
                    .open(&path)?;
                
                // Read page-aligned data
                let aligned_offset = (info.offset / PAGE_SIZE as u64) * PAGE_SIZE as u64;
                file.seek(SeekFrom::Start(aligned_offset))?;
                
                let read_size = info.page_count as usize * PAGE_SIZE;
                let mut buffer = vec![0u8; read_size];
                file.read_exact(&mut buffer)?;
                
                // Extract actual data
                let data = &buffer[info.page_offset as usize..info.page_offset as usize + info.total_bytes as usize];
                
                // Decompress if needed
                if enable_compression {
                    zstd::decode_all(data)
                } else {
                    Ok(data.to_vec())
                }
            }
            
            #[cfg(not(target_os = "linux"))]
            {
                // Fallback to regular I/O
                let mut file = File::open(&path)?;
                file.seek(SeekFrom::Start(info.offset))?;
                
                let mut buffer = vec![0u8; info.total_bytes as usize];
                file.read_exact(&mut buffer)?;
                
                if enable_compression {
                    zstd::decode_all(&buffer[..])
                } else {
                    Ok(buffer)
                }
            }
        })
        .await
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?
    }

    /// Load multiple posting lists with Direct I/O in parallel
    pub async fn load_postings_batch_direct(&self, posting_ids: &[usize]) -> io::Result<Vec<Vec<u8>>> {
        let futures: Vec<_> = posting_ids
            .iter()
            .map(|&id| self.load_posting_direct(id))
            .collect();

        futures::future::try_join_all(futures).await
    }
}
