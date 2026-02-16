use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write, Seek, SeekFrom};
use std::path::Path;
use serde::{Serialize, Deserialize};

#[cfg(target_os = "linux")]
use std::os::unix::fs::OpenOptionsExt;

pub const PAGE_SIZE: usize = 8192;  // 8KB pages

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ListInfo {
    pub offset: u64,
    pub page_count: u16,
    pub ele_count: u16,
    pub total_bytes: u32,
    pub page_offset: u32,
}

pub struct SPANNStorage {
    list_infos: Vec<ListInfo>,
}

impl SPANNStorage {
    pub fn new() -> Self {
        Self {
            list_infos: Vec::new(),
        }
    }

    /// Save index to disk with compression and delta encoding
    pub fn save(
        &mut self,
        postings: &[crate::spann::PostingList],
        full_vectors: &[Vec<f32>],
        path: &str,
        enable_compression: bool,
        enable_delta: bool,
        save_quantized: bool,  // NEW: if true, save quantized data instead of full vectors
    ) -> io::Result<()> {
        let mut file = File::create(path)?;
        
        // Write header
        let num_postings = postings.len() as u64;
        let dim = if !full_vectors.is_empty() { full_vectors[0].len()} else { 0 };
        file.write_all(&num_postings.to_le_bytes())?;
        file.write_all(&(dim as u32).to_le_bytes())?;
        file.write_all(&[enable_compression as u8, enable_delta as u8, save_quantized as u8])?;
        
        // Reserve space for list infos
        let list_infos_offset = file.seek(SeekFrom::Current(0))?;
        let list_info_size = 22;
        for _ in 0..num_postings {
            file.write_all(&vec![0u8; list_info_size])?;
        }
        
        // Write posting lists
        self.list_infos.clear();
        for posting in postings {
            let start_offset = file.seek(SeekFrom::Current(0))?;
            
            let mut buffer = Vec::new();
            
            // Write vector IDs
            for &id in &posting.vector_ids {
                buffer.extend_from_slice(&(id as u32).to_le_bytes());
            }
            
            // Write vectors: quantized or full precision
            if save_quantized && posting.quantized_data.is_some() && posting.quantizer.is_some() {
                // Write quantized data
                let quantized = posting.quantized_data.as_ref().unwrap();
                let quantizer = posting.quantizer.as_ref().unwrap();
                
                // Write quantizer metadata
                let metadata = quantizer.serialize_metadata();
                buffer.extend_from_slice(&(metadata.len() as u32).to_le_bytes());
                buffer.extend_from_slice(&metadata);
                
                // Write quantized codes
                buffer.extend_from_slice(quantized);
            } else {
                // Write full precision vectors
                if enable_delta {
                    let head_vec = &full_vectors[posting.head_id];
                    let mut delta_buffer = vec![0.0f32; head_vec.len()];
                    
                    for &id in &posting.vector_ids {
                        let vec = &full_vectors[id];
                        // Use SIMD for delta encoding
                        crate::spann::simd_delta::encode_delta_simd(vec, head_vec, &mut delta_buffer);
                        
                        for &val in &delta_buffer {
                            buffer.extend_from_slice(&val.to_le_bytes());
                        }
                    }
                } else {
                    for &id in &posting.vector_ids {
                        let vec = &full_vectors[id];
                        for &val in vec {
                            buffer.extend_from_slice(&val.to_le_bytes());
                        }
                    }
                }
            }
            
            // Compress if enabled
            let data = if enable_compression {
                zstd::encode_all(&buffer[..], 3)?
            } else {
                buffer
            };
            
            // Align write offset to PAGE_SIZE boundary for Direct I/O
            let current_offset = file.seek(SeekFrom::Current(0))?;
            let aligned_offset = (current_offset + PAGE_SIZE as u64 - 1) & !(PAGE_SIZE as u64 - 1);
            let padding_before = (aligned_offset - current_offset) as usize;
            
            if padding_before > 0 {
                file.write_all(&vec![0u8; padding_before])?;
            }
            
            let start_offset = aligned_offset;
            let page_offset = 0; // Always 0 since we're aligned
            
            // Write data
            file.write_all(&data)?;
            
            // Pad to page boundary
            let aligned_size = ((data.len() + PAGE_SIZE - 1) / PAGE_SIZE) * PAGE_SIZE;
            let padding_after = aligned_size - data.len();
            if padding_after > 0 {
                file.write_all(&vec![0u8; padding_after])?;
            }
            
            let end_offset = file.seek(SeekFrom::Current(0))?;
            
            self.list_infos.push(ListInfo {
                offset: start_offset,
                page_count: ((end_offset - start_offset) / PAGE_SIZE as u64) as u16,
                ele_count: posting.vector_ids.len() as u16,
                total_bytes: data.len() as u32,
                page_offset,
            });
        }
        
        // Write list infos
        file.seek(SeekFrom::Start(list_infos_offset))?;
        for info in &self.list_infos {
            file.write_all(&info.offset.to_le_bytes())?;
            file.write_all(&info.page_count.to_le_bytes())?;
            file.write_all(&info.ele_count.to_le_bytes())?;
            file.write_all(&info.total_bytes.to_le_bytes())?;
            file.write_all(&info.page_offset.to_le_bytes())?;
        }
        
        Ok(())
    }

    /// Load index from disk
    pub fn load(path: &str) -> io::Result<(Vec<crate::spann::PostingList>, Vec<Vec<f32>>, Vec<ListInfo>)> {
        let mut file = File::open(path)?;
        
        // Read header
        let mut num_postings_bytes = [0u8; 8];
        file.read_exact(&mut num_postings_bytes)?;
        let num_postings = u64::from_le_bytes(num_postings_bytes) as usize;
        
        let mut dim_bytes = [0u8; 4];
        file.read_exact(&mut dim_bytes)?;
        let dim = u32::from_le_bytes(dim_bytes) as usize;
        
        let mut flags = [0u8; 3];
        file.read_exact(&mut flags)?;
        let enable_compression = flags[0] != 0;
        let enable_delta = flags[1] != 0;
        let has_quantization = flags[2] != 0;
        
        // Read list infos
        let mut list_infos = Vec::with_capacity(num_postings);
        for _ in 0..num_postings {
            let mut offset_bytes = [0u8; 8];
            file.read_exact(&mut offset_bytes)?;
            let offset = u64::from_le_bytes(offset_bytes);
            
            let mut page_count_bytes = [0u8; 2];
            file.read_exact(&mut page_count_bytes)?;
            let page_count = u16::from_le_bytes(page_count_bytes);
            
            let mut ele_count_bytes = [0u8; 2];
            file.read_exact(&mut ele_count_bytes)?;
            let ele_count = u16::from_le_bytes(ele_count_bytes);
            
            let mut total_bytes_bytes = [0u8; 4];
            file.read_exact(&mut total_bytes_bytes)?;
            let total_bytes = u32::from_le_bytes(total_bytes_bytes);
            
            let mut page_offset_bytes = [0u8; 4];
            file.read_exact(&mut page_offset_bytes)?;
            let page_offset = u32::from_le_bytes(page_offset_bytes);
            
            list_infos.push(ListInfo {
                offset,
                page_count,
                ele_count,
                total_bytes,
                page_offset,
            });
        }
        
        // Read posting lists and reconstruct full vector set
        let mut postings = Vec::with_capacity(num_postings);
        let mut vector_map = std::collections::HashMap::new();
        let mut max_vec_id = 0;
        
        // First pass: collect all unique vector IDs and find max
        for info in &list_infos {
            file.seek(SeekFrom::Start(info.offset))?;
            
            let mut compressed = vec![0u8; info.total_bytes as usize];
            file.read_exact(&mut compressed)?;
            
            let buffer = if enable_compression {
                zstd::decode_all(&compressed[..])?
            } else {
                compressed
            };
            
            // Parse vector IDs
            for i in 0..info.ele_count as usize {
                let id_bytes = &buffer[i * 4..(i + 1) * 4];
                let id = u32::from_le_bytes([id_bytes[0], id_bytes[1], id_bytes[2], id_bytes[3]]) as usize;
                if id > max_vec_id {
                    max_vec_id = id;
                }
            }
        }
        
        // Initialize full vectors array
        let mut all_vectors = vec![vec![0.0f32; dim]; max_vec_id + 1];
        
        // Second pass: read vectors and build postings
        for (head_idx, info) in list_infos.iter().enumerate() {
            file.seek(SeekFrom::Start(info.offset))?;
            
            let mut compressed = vec![0u8; info.total_bytes as usize];
            file.read_exact(&mut compressed)?;
            
            let buffer = if enable_compression {
                zstd::decode_all(&compressed[..])?
            } else {
                compressed
            };
            
            // Parse vector IDs
            let mut vector_ids = Vec::with_capacity(info.ele_count as usize);
            for i in 0..info.ele_count as usize {
                let id_bytes = &buffer[i * 4..(i + 1) * 4];
                let id = u32::from_le_bytes([id_bytes[0], id_bytes[1], id_bytes[2], id_bytes[3]]) as usize;
                vector_ids.push(id);
            }
            
            // Parse vectors
            let vec_data_offset = info.ele_count as usize * 4;
            
            let (quantized_data, quantizer) = if has_quantization && !vector_ids.is_empty() {
                // Read quantized data
                let meta_len = u32::from_le_bytes([
                    buffer[vec_data_offset],
                    buffer[vec_data_offset + 1],
                    buffer[vec_data_offset + 2],
                    buffer[vec_data_offset + 3],
                ]) as usize;
                
                if head_idx == 0 {
                    println!("Loading quantized data: meta_len={}, vec_count={}", meta_len, vector_ids.len());
                }
                
                let meta_start = vec_data_offset + 4;
                let (mut quantizer, _) = crate::multibit::MultiBitQuantizer::deserialize_metadata(
                    &buffer[meta_start..meta_start + meta_len]
                );
                
                let codes_start = meta_start + meta_len;
                let bits = quantizer.bits();
                let bytes_per_vec = (dim * bits + 7) / 8;
                let codes_len = info.ele_count as usize * bytes_per_vec;
                
                if head_idx == 0 {
                    println!("  bits={}, bytes_per_vec={}, codes_len={}, buffer_remaining={}", 
                             bits, bytes_per_vec, codes_len, buffer.len() - codes_start);
                }
                
                let quantized_data = buffer[codes_start..codes_start + codes_len].to_vec();
                
                // Populate binary_codes in quantizer
                quantizer.set_binary_codes(quantized_data.clone());
                
                (Some(quantized_data), Some(quantizer))
            } else {
                (None, None)
            };
            
            // Load full precision vectors only if NOT quantized
            if quantized_data.is_none() {
                let head_vec = if enable_delta && head_idx < all_vectors.len() {
                    Some(all_vectors[head_idx].clone())
                } else {
                    None
                };
                
                let mut decode_buffer = vec![0.0f32; dim];
                
                for (i, &vec_id) in vector_ids.iter().enumerate() {
                    if !vector_map.contains_key(&vec_id) {
                        let vec_offset = vec_data_offset + i * dim * 4;
                        let mut vec = Vec::with_capacity(dim);
                        for d in 0..dim {
                            let val_offset = vec_offset + d * 4;
                            let val_bytes = &buffer[val_offset..val_offset + 4];
                            let val = f32::from_le_bytes([val_bytes[0], val_bytes[1], val_bytes[2], val_bytes[3]]);
                            vec.push(val);
                        }
                        
                        // Apply delta decoding with SIMD if needed
                        if let Some(ref head) = head_vec {
                            crate::spann::simd_delta::decode_delta_simd(&vec, head, &mut decode_buffer);
                            all_vectors[vec_id] = decode_buffer.clone();
                        } else {
                            all_vectors[vec_id] = vec;
                        }
                        
                        vector_map.insert(vec_id, true);
                    }
                }
            }
            
            postings.push(crate::spann::PostingList {
                head_id: if !vector_ids.is_empty() { vector_ids[0] } else { head_idx },
                vector_ids,
                quantized_data,
                quantizer,
            });
        }
        
        Ok((postings, all_vectors, list_infos))
    }

    /// Open file with Direct I/O (O_DIRECT)
    #[cfg(target_os = "linux")]
    pub fn open_direct_io(path: &str) -> io::Result<File> {
        use std::os::unix::fs::OpenOptionsExt;
        OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)
    }

    #[cfg(not(target_os = "linux"))]
    pub fn open_direct_io(path: &str) -> io::Result<File> {
        // Fallback for non-Linux systems
        File::open(path)
    }
    
    /// Allocate aligned buffer for Direct I/O
    #[cfg(target_os = "linux")]
    pub fn alloc_aligned_buffer(size: usize) -> Vec<u8> {
        const ALIGNMENT: usize = 4096; // 4KB alignment for O_DIRECT
        let aligned_size = (size + ALIGNMENT - 1) & !(ALIGNMENT - 1);
        let layout = std::alloc::Layout::from_size_align(aligned_size, ALIGNMENT).unwrap();
        unsafe {
            let ptr = std::alloc::alloc(layout);
            Vec::from_raw_parts(ptr, size, aligned_size)
        }
    }
    
    #[cfg(not(target_os = "linux"))]
    pub fn alloc_aligned_buffer(size: usize) -> Vec<u8> {
        vec![0u8; size]
    }

    /// Read posting list with Direct I/O
    pub fn read_posting_direct(
        file: &mut File,
        info: &ListInfo,
        enable_compression: bool,
    ) -> io::Result<Vec<u8>> {
        // Seek to page-aligned offset
        let aligned_offset = (info.offset / PAGE_SIZE as u64) * PAGE_SIZE as u64;
        file.seek(SeekFrom::Start(aligned_offset))?;
        
        // Read page-aligned data
        let read_size = info.page_count as usize * PAGE_SIZE;
        let mut buffer = vec![0u8; read_size];
        file.read_exact(&mut buffer)?;
        
        // Extract actual data (skip page offset)
        let data = &buffer[info.page_offset as usize..info.page_offset as usize + info.total_bytes as usize];
        
        // Decompress if needed
        if enable_compression {
            zstd::decode_all(data)
        } else {
            Ok(data.to_vec())
        }
    }
}
