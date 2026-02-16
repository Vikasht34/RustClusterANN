use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom};
use std::os::unix::fs::OpenOptionsExt;
use std::sync::Arc;
use tokio::task;

use super::storage::{ListInfo, PAGE_SIZE};
use super::posting::PostingList;

/// Optimized async storage with Direct I/O and parallel prefetch
pub struct OptimizedAsyncStorage {
    path: String,
    pub list_infos: Arc<Vec<ListInfo>>,
    enable_compression: bool,
    enable_delta: bool,
    enable_direct_io: bool,
}

impl OptimizedAsyncStorage {
    pub fn new(path: String, list_infos: Vec<ListInfo>, enable_compression: bool, enable_delta: bool) -> Self {
        // Enable Direct I/O on Linux (offsets are now properly aligned)
        let enable_direct_io = cfg!(target_os = "linux");
        
        Self {
            path,
            list_infos: Arc::new(list_infos),
            enable_compression,
            enable_delta,
            enable_direct_io,
        }
    }
    
    pub fn is_delta_enabled(&self) -> bool {
        self.enable_delta
    }

    /// Open file with Direct I/O if available
    fn open_file(&self) -> io::Result<File> {
        #[cfg(target_os = "linux")]
        if self.enable_direct_io {
            return OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECT)
                .open(&self.path);
        }
        
        File::open(&self.path)
    }

    /// Load single posting list with Direct I/O
    pub async fn load_posting(&self, posting_id: usize) -> io::Result<Vec<u8>> {
        let path = self.path.clone();
        let info = self.list_infos[posting_id].clone();
        let enable_compression = self.enable_compression;
        let enable_direct_io = self.enable_direct_io;

        task::spawn_blocking(move || {
            let mut file = if enable_direct_io {
                #[cfg(target_os = "linux")]
                {
                    OpenOptions::new()
                        .read(true)
                        .custom_flags(libc::O_DIRECT)
                        .open(&path)?
                }
                #[cfg(not(target_os = "linux"))]
                {
                    File::open(&path)?
                }
            } else {
                File::open(&path)?
            };
            
            // Seek to page-aligned offset
            let aligned_offset = info.offset;
            file.seek(SeekFrom::Start(aligned_offset))?;
            
            // Read page-aligned data
            let read_size = if enable_direct_io {
                // Round up to 4KB for Direct I/O (filesystem block size)
                ((info.total_bytes as usize + 4095) / 4096) * 4096
            } else {
                info.total_bytes as usize
            };
            
            // Allocate aligned buffer for Direct I/O
            let mut buffer = if enable_direct_io {
                #[cfg(target_os = "linux")]
                {
                    // Allocate 4KB-aligned buffer
                    let layout = std::alloc::Layout::from_size_align(read_size, 4096).unwrap();
                    unsafe {
                        let ptr = std::alloc::alloc(layout);
                        if ptr.is_null() {
                            return Err(io::Error::new(io::ErrorKind::OutOfMemory, "Failed to allocate aligned buffer"));
                        }
                        // Initialize to zero
                        std::ptr::write_bytes(ptr, 0, read_size);
                        Vec::from_raw_parts(ptr, read_size, read_size)
                    }
                }
                #[cfg(not(target_os = "linux"))]
                {
                    vec![0u8; read_size]
                }
            } else {
                vec![0u8; read_size]
            };
            
            file.read_exact(&mut buffer)?;
            
            // Trim to actual size
            buffer.truncate(info.total_bytes as usize);
            
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

    /// Load multiple posting lists in parallel with prefetch
    pub async fn load_postings_parallel(&self, posting_ids: &[usize]) -> io::Result<Vec<Vec<u8>>> {
        let futures: Vec<_> = posting_ids
            .iter()
            .map(|&id| self.load_posting(id))
            .collect();
        
        futures::future::try_join_all(futures).await
    }

    /// Prefetch posting lists (hint to OS for readahead)
    #[cfg(target_os = "linux")]
    pub async fn prefetch_postings(&self, posting_ids: &[usize]) -> io::Result<()> {
        use std::os::unix::io::AsRawFd;
        
        let path = self.path.clone();
        let infos: Vec<_> = posting_ids.iter().map(|&id| self.list_infos[id].clone()).collect();
        
        task::spawn_blocking(move || {
            let file = File::open(&path)?;
            let fd = file.as_raw_fd();
            
            for info in infos {
                unsafe {
                    // Advise kernel to readahead
                    libc::posix_fadvise(
                        fd,
                        info.offset as i64,
                        info.total_bytes as i64,
                        libc::POSIX_FADV_WILLNEED,
                    );
                }
            }
            
            Ok::<_, io::Error>(())
        })
        .await
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?
    }

    #[cfg(not(target_os = "linux"))]
    pub async fn prefetch_postings(&self, _posting_ids: &[usize]) -> io::Result<()> {
        Ok(())
    }

    /// Get list info for a posting
    pub fn get_list_info(&self, posting_id: usize) -> &ListInfo {
        &self.list_infos[posting_id]
    }

    /// Get total number of posting lists
    pub fn num_postings(&self) -> usize {
        self.list_infos.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_parallel_load() {
        // Test parallel loading
    }
}
