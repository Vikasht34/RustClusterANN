/// SPANN Index - Two-level index with head + posting lists
/// Matches SPTAG's SPANN implementation exactly
use crate::index::sptag_bkt::SPTAGBKTIndex;
use crate::spann::hbc::HBCSelector;
use crate::simd;
use crate::multibit::{MultiBitQuantizer, MetricType as QuantMetric};
use std::io::{Read, Write};

#[derive(Clone, Copy)]
pub enum DistanceMetric {
    L2,
    InnerProduct,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum QuantizationType {
    None,       // Full precision (32-bit float)
    OneBit,     // 1-bit binary
    TwoBit,     // 2-bit (1-bit + 1 ex_bit)
    FourBit,    // 4-bit (1-bit + 3 ex_bits)
}

#[derive(Default)]
pub struct SearchStats {
    pub bytes_read: usize,
    pub posting_lists_accessed: usize,
    pub head_search_time_us: u64,
    pub posting_search_time_us: u64,
}

pub struct PostingList {
    pub head_id: usize,
    pub vector_ids: Vec<usize>,
    // Quantized data (if quantization enabled)
    pub quantized_data: Option<Vec<u8>>,
    pub quantizer: Option<MultiBitQuantizer>,
}

impl PostingList {
    pub fn new(head_id: usize) -> Self {
        Self {
            head_id,
            vector_ids: Vec::new(),
            quantized_data: None,
            quantizer: None,
        }
    }
}

pub struct SPANNIndex {
    // Head index (in-memory SPTAG-BKT) - ALWAYS in memory
    head_index: SPTAGBKTIndex,
    head_id_map: Vec<usize>,  // Maps head_index position to original vector ID
    
    // Posting lists (empty in on-demand mode)
    postings: Vec<PostingList>,
    
    // Full vectors (empty in on-demand mode)
    full_vectors: Vec<Vec<f32>>,
    
    // Distance metric
    metric: DistanceMetric,
    
    // Quantization
    quantization: QuantizationType,
    
    // SPTAG defaults
    ratio: f32,                    // 0.01 (1% heads)
    replica_count: usize,          // 8 replicas per vector
    posting_vector_limit: usize,   // 118 max vectors per posting
    num_heads_to_search: usize,    // 64 heads to search
    internal_result_num: usize,    // 64 candidates during build
    
    // Optional: sample size for HBC (None = no sampling, SPTAG default)
    hbc_sample_size: Option<usize>,
    
    // Storage metadata (for on-demand loading)
    list_infos: Vec<crate::spann::storage::ListInfo>,
    index_path: Option<String>,
    
    // On-demand mode flag
    on_demand_mode: bool,
    
    // Dimension (needed for on-demand mode)
    dim: usize,
}


impl SPANNIndex {
    pub fn new() -> Self {
        Self {
            head_index: SPTAGBKTIndex::new(32, 2000, 32, 1.0, 500, 32),
            head_id_map: Vec::new(),
            postings: Vec::new(),
            full_vectors: Vec::new(),
            metric: DistanceMetric::L2,     // Default to L2
            quantization: QuantizationType::None,  // Default: full precision
            ratio: 0.01,                    // 1% heads (SPTAG typical)
            replica_count: 8,               // 8 replicas per vector
            posting_vector_limit: 118,      // Max 118 vectors per posting
            num_heads_to_search: 64,        // Search 64 heads
            internal_result_num: 64,        // 64 candidates during build
            hbc_sample_size: Some(100_000), // Sample 100K for HBC (PlanetScale approach)
            list_infos: Vec::new(),
            index_path: None,
            on_demand_mode: false,
            dim: 0,
        }
    }
    
    pub fn set_metric(&mut self, metric: DistanceMetric) {
        self.metric = metric;
    }
    
    pub fn set_quantization(&mut self, quantization: QuantizationType) {
        self.quantization = quantization;
    }
    
    pub fn set_hbc_sample_size(&mut self, sample_size: Option<usize>) {
        self.hbc_sample_size = sample_size;
    }
    
    #[inline]
    fn compute_distance(&self, x: &[f32], y: &[f32]) -> f32 {
        match self.metric {
            DistanceMetric::L2 => simd::l2_distance(x, y),
            DistanceMetric::InnerProduct => simd::inner_product(x, y),
        }
    }

    pub fn build(&mut self, vectors: Vec<Vec<f32>>) {
        let n = vectors.len();
        println!("Building SPANN index on {} vectors", n);
        println!("Using {:.0}% heads (SPTAG default)", self.ratio * 100.0);
        
        self.full_vectors = vectors;
        
        // Phase 1: Select heads using HBC
        println!("\nPhase 1: Selecting heads (HBC)...");
        let head_indices = self.select_heads_hbc();
        let num_heads = head_indices.len();
        println!("  Selected {} heads ({:.2}%)", 
                 num_heads, 
                 num_heads as f32 / n as f32 * 100.0);
        
        // Adjust posting limit based on dataset size
        // Target: replica_count * n / num_heads vectors per posting
        let ideal_posting_size = (self.replica_count * n) / num_heads;
        self.posting_vector_limit = ideal_posting_size.max(self.posting_vector_limit);
        println!("  Posting limit: {} vectors/head", self.posting_vector_limit);
        
        // Phase 2: Build head index (SPTAG-BKT)
        println!("\nPhase 2: Building head index (SPTAG-BKT)...");
        let head_vectors: Vec<_> = head_indices.iter()
            .map(|&i| self.full_vectors[i].clone())
            .collect();
        
        self.head_index.build(head_vectors, 2);
        self.head_id_map = head_indices;
        println!("  Head index built: {} vectors", self.head_index.len());
        
        // Phase 3: Assign vectors to posting lists
        println!("\nPhase 3: Assigning vectors to posting lists...");
        self.assign_to_postings();
        
        // Phase 4: Quantize posting lists (if enabled)
        if self.quantization != QuantizationType::None {
            println!("\nPhase 4: Quantizing posting lists ({:?})...", self.quantization);
            self.quantize_postings();
        }
        
        println!("\nSPANN index built successfully!");
    }
    
    fn select_heads_hbc(&self) -> Vec<usize> {
        let selector = HBCSelector::new();
        selector.select_heads_with_sampling(&self.full_vectors, self.ratio, self.hbc_sample_size)
    }
    
    fn assign_to_postings(&mut self) {
        let num_heads = self.head_index.len();
        
        // Phase 1: Collect all assignments with distances
        let mut assignments: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_heads];
        
        for (vec_id, vec) in self.full_vectors.iter().enumerate() {
            if vec_id % 10000 == 0 && vec_id > 0 {
                println!("  Assigned {}/{} vectors", vec_id, self.full_vectors.len());
            }
            
            let results = self.head_index.search(vec, self.replica_count, self.internal_result_num);
            
            for (head_idx, dist) in results {
                if head_idx < num_heads {
                    assignments[head_idx].push((vec_id, dist));
                }
            }
        }
        
        // Phase 2: Prune each posting list to keep closest vectors
        println!("  Pruning posting lists to keep closest vectors...");
        self.postings = Vec::with_capacity(num_heads);
        let mut total_replicas = 0;
        let mut pruned_count = 0;
        
        for (head_idx, mut candidates) in assignments.into_iter().enumerate() {
            // Sort by distance (closest first)
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            
            // Count how many we're pruning
            if candidates.len() > self.posting_vector_limit {
                pruned_count += candidates.len() - self.posting_vector_limit;
            }
            
            // Keep only closest posting_vector_limit vectors
            candidates.truncate(self.posting_vector_limit);
            
            total_replicas += candidates.len();
            
            let vector_ids: Vec<usize> = candidates.iter().map(|(vec_id, _)| *vec_id).collect();
            self.postings.push(PostingList {
                head_id: self.head_id_map[head_idx],
                vector_ids,
                quantized_data: None,
                quantizer: None,
            });
        }
        
        // Statistics
        let posting_sizes: Vec<usize> = self.postings.iter().map(|p| p.vector_ids.len()).collect();
        let avg_size = total_replicas as f32 / num_heads as f32;
        let max_size = posting_sizes.iter().copied().max().unwrap_or(0);
        let min_size = posting_sizes.iter().copied().min().unwrap_or(0);
        let avg_replicas = total_replicas as f32 / self.full_vectors.len() as f32;
        
        println!("  Posting list statistics:");
        println!("    Average: {:.1} vectors/head", avg_size);
        println!("    Min: {}, Max: {}", min_size, max_size);
        println!("    Average replicas per vector: {:.1}", avg_replicas);
        println!("    Pruned (kept closest): {}", pruned_count);
    }
    
    fn quantize_postings(&mut self) {
        let bits = match self.quantization {
            QuantizationType::OneBit => 1,
            QuantizationType::TwoBit => 2,
            QuantizationType::FourBit => 4,
            QuantizationType::None => return,
        };
        
        let quant_metric = match self.metric {
            DistanceMetric::L2 => QuantMetric::L2,
            DistanceMetric::InnerProduct => QuantMetric::IP,
        };
        
        let mut total_original_bytes = 0u64;
        let mut total_quantized_bytes = 0u64;
        
        for posting in &mut self.postings {
            if posting.vector_ids.is_empty() {
                continue;
            }
            
            // Gather vectors for this posting
            let vectors: Vec<Vec<f32>> = posting.vector_ids.iter()
                .map(|&id| self.full_vectors[id].clone())
                .collect();
            
            let dim = vectors[0].len();
            total_original_bytes += (vectors.len() * dim * 4) as u64;
            
            // Train quantizer
            let mut quantizer = MultiBitQuantizer::new(bits);
            quantizer.train(&vectors, quant_metric);
            
            // Extract quantized codes
            let quantized_data = quantizer.get_all_codes();
            
            // Estimate compressed size (bits per vector)
            let bytes_per_vector = match bits {
                1 => (dim + 7) / 8,  // 1 bit per dimension
                2 => (dim * 2 + 7) / 8,  // 2 bits per dimension
                4 => (dim * 4 + 7) / 8,  // 4 bits per dimension
                _ => dim * 4,
            };
            total_quantized_bytes += (vectors.len() * bytes_per_vector) as u64;
            
            posting.quantized_data = Some(quantized_data);
            posting.quantizer = Some(quantizer);
        }
        
        let compression_ratio = total_original_bytes as f32 / total_quantized_bytes as f32;
        println!("  Original size: {:.2} MB", total_original_bytes as f32 / 1024.0 / 1024.0);
        println!("  Quantized size: {:.2} MB", total_quantized_bytes as f32 / 1024.0 / 1024.0);
        println!("  Compression ratio: {:.2}x", compression_ratio);
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let (results, _) = self.search_with_stats(query, k);
        results
    }

    pub fn search_with_stats(&self, query: &[f32], k: usize) -> (Vec<(usize, f32)>, SearchStats) {
        use std::collections::HashSet;
        use std::time::Instant;
        
        let mut stats = SearchStats::default();
        
        // Step 1: Search head index
        let head_start = Instant::now();
        let head_results = self.head_index.search(query, self.num_heads_to_search, 500);
        stats.head_search_time_us = head_start.elapsed().as_micros() as u64;
        
        // Step 2 & 3: Collect candidates and compute distances
        let posting_start = Instant::now();
        let mut results: Vec<(usize, f32)> = Vec::new();
        let quant_metric = match self.metric {
            DistanceMetric::L2 => crate::multibit::MetricType::L2,
            DistanceMetric::InnerProduct => crate::multibit::MetricType::IP,
        };
        
        for (head_idx, _) in head_results {
            if head_idx >= self.postings.len() {
                continue;
            }
            
            stats.posting_lists_accessed += 1;
            let posting = &self.postings[head_idx];
            
            // Use quantized distances if available, otherwise full precision
            if let (Some(ref quantizer), Some(ref quant_data)) = (&posting.quantizer, &posting.quantized_data) {
                // Quantized distance computation
                stats.bytes_read += quant_data.len();
                stats.bytes_read += posting.vector_ids.len() * 4; // vec_ids
                
                let distances = quantizer.compute_distances(query, quant_metric);
                for (i, &vec_id) in posting.vector_ids.iter().enumerate() {
                    if i < distances.len() {
                        results.push((vec_id, distances[i]));
                    }
                }
            } else {
                // Full precision distance computation
                for &vec_id in &posting.vector_ids {
                    if vec_id < self.full_vectors.len() {
                        let vec_ref = &self.full_vectors[vec_id];
                        stats.bytes_read += vec_ref.len() * 4; // f32 vectors
                        let dist = self.compute_distance(query, vec_ref);
                        results.push((vec_id, dist));
                    }
                }
                stats.bytes_read += posting.vector_ids.len() * 4; // vec_ids
            }
        }
        
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        stats.posting_search_time_us = posting_start.elapsed().as_micros() as u64;
        
        (results, stats)
    }

    /// On-demand search: loads posting lists from disk as needed
    /// Only head index stays in RAM, posting lists loaded per query
    /// On-demand search: loads posting lists + vectors from disk per query
    /// Only head index stays in RAM
    pub async fn search_async(
        &self,
        query: &[f32],
        k: usize,
        storage: &crate::spann::OptimizedAsyncStorage,
    ) -> std::io::Result<Vec<(usize, f32)>> {
        use std::collections::HashSet;
        
        // Step 1: Search head index (in RAM)
        let head_results = self.head_index.search(query, self.num_heads_to_search, 500);
        let posting_ids: Vec<usize> = head_results.iter().map(|(idx, _)| *idx).collect();
        
        // Step 2: Prefetch posting lists
        storage.prefetch_postings(&posting_ids).await?;
        
        // Step 3: Load posting lists in parallel from disk
        let posting_data = storage.load_postings_parallel(&posting_ids).await?;
        
        // Step 4: Parse posting lists and collect candidates with vectors
        let mut seen = HashSet::new();
        let mut candidates = Vec::new();
        
        for (posting_idx, data) in posting_data.iter().enumerate() {
            let list_info = &storage.list_infos[posting_ids[posting_idx]];
            let count = list_info.ele_count as usize;
            
            // Parse: [vec_id (u32), vec_id (u32), ..., vector data (f32)...]
            let mut cursor = 0;
            
            // Read vector IDs (u32 each, no count prefix)
            let mut vec_ids = Vec::with_capacity(count);
            for _ in 0..count {
                if cursor + 4 > data.len() {
                    break;
                }
                let id = u32::from_le_bytes([
                    data[cursor],
                    data[cursor + 1],
                    data[cursor + 2],
                    data[cursor + 3],
                ]) as usize;
                vec_ids.push(id);
                cursor += 4;
            }
            
            // Read vectors (full precision f32)
            for &vec_id in &vec_ids {
                if !seen.insert(vec_id) {
                    cursor += self.dim * 4;
                    continue;
                }
                
                if cursor + self.dim * 4 > data.len() {
                    break;
                }
                
                // Extract vector
                let mut vec = Vec::with_capacity(self.dim);
                for _ in 0..self.dim {
                    let val = f32::from_le_bytes([
                        data[cursor],
                        data[cursor + 1],
                        data[cursor + 2],
                        data[cursor + 3],
                    ]);
                    vec.push(val);
                    cursor += 4;
                }
                
                // Compute distance
                let dist = self.compute_distance(query, &vec);
                candidates.push((vec_id, dist));
            }
        }
        
        // Step 5: Sort and return top-k
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates.truncate(k);
        Ok(candidates)
    }
    
    pub fn len(&self) -> usize {
        self.full_vectors.len()
    }
    
    /// Save index to disk
    pub fn save(&self, path: &str, enable_compression: bool, enable_delta: bool) -> std::io::Result<()> {
        // Determine if we should save quantized data
        let save_quantized = self.quantization != QuantizationType::None 
            && self.postings.iter().any(|p| p.quantized_data.is_some());
        
        let mut storage = crate::spann::SPANNStorage::new();
        storage.save(&self.postings, &self.full_vectors, path, enable_compression, enable_delta, save_quantized)?;
        
        // Save head vectors separately (always full precision for accurate routing)
        let head_path = format!("{}.heads", path);
        let mut head_file = std::fs::File::create(&head_path)?;
        let num_heads = self.postings.len();
        let dim = if !self.full_vectors.is_empty() { self.full_vectors[0].len() } else { 0 };
        
        head_file.write_all(&(num_heads as u32).to_le_bytes())?;
        head_file.write_all(&(dim as u32).to_le_bytes())?;
        
        for posting in &self.postings {
            let head_id = posting.head_id;
            if head_id < self.full_vectors.len() {
                for &val in &self.full_vectors[head_id] {
                    head_file.write_all(&val.to_le_bytes())?;
                }
            }
        }
        
        // Save metadata
        let meta_path = format!("{}.meta", path);
        let meta = serde_json::json!({
            "metric": match self.metric {
                DistanceMetric::L2 => "L2",
                DistanceMetric::InnerProduct => "InnerProduct",
            },
            "quantization": match self.quantization {
                QuantizationType::None => "None",
                QuantizationType::OneBit => "OneBit",
                QuantizationType::TwoBit => "TwoBit",
                QuantizationType::FourBit => "FourBit",
            },
            "ratio": self.ratio,
            "replica_count": self.replica_count,
            "posting_vector_limit": self.posting_vector_limit,
            "num_heads_to_search": self.num_heads_to_search,
            "enable_compression": enable_compression,
            "enable_delta": enable_delta,
        });
        std::fs::write(meta_path, serde_json::to_string_pretty(&meta)?)?;
        
        println!("Index saved to {}", path);
        Ok(())
    }
    
    /// Load index from disk
    /// Load index in in-memory mode (default)
    pub fn load(path: &str) -> std::io::Result<Self> {
        Self::load_with_mode(path, false)
    }
    
    /// Load index with optional on-demand mode
    /// on_demand=false: Load everything into memory (default)
    /// on_demand=true: Only load head index, posting lists + vectors loaded per query
    pub fn load_with_mode(path: &str, on_demand: bool) -> std::io::Result<Self> {
        let (postings, full_vectors, list_infos) = crate::spann::SPANNStorage::load(path)?;
        
        // Load metadata
        let meta_path = format!("{}.meta", path);
        let meta_str = std::fs::read_to_string(meta_path)?;
        let meta: serde_json::Value = serde_json::from_str(&meta_str)?;
        
        let metric = match meta["metric"].as_str().unwrap() {
            "L2" => DistanceMetric::L2,
            "InnerProduct" => DistanceMetric::InnerProduct,
            _ => DistanceMetric::L2,
        };
        
        let quantization = match meta["quantization"].as_str().unwrap() {
            "OneBit" => QuantizationType::OneBit,
            "TwoBit" => QuantizationType::TwoBit,
            "FourBit" => QuantizationType::FourBit,
            _ => QuantizationType::None,
        };
        
        // Load head vectors from separate file
        let head_path = format!("{}.heads", path);
        let mut head_file = std::fs::File::open(&head_path)?;
        
        let mut num_heads_bytes = [0u8; 4];
        head_file.read_exact(&mut num_heads_bytes)?;
        let num_heads = u32::from_le_bytes(num_heads_bytes) as usize;
        
        let mut dim_bytes = [0u8; 4];
        head_file.read_exact(&mut dim_bytes)?;
        let dim = u32::from_le_bytes(dim_bytes) as usize;
        
        let mut head_vectors = Vec::with_capacity(num_heads);
        for _ in 0..num_heads {
            let mut vec = vec![0.0f32; dim];
            for val in vec.iter_mut() {
                let mut bytes = [0u8; 4];
                head_file.read_exact(&mut bytes)?;
                *val = f32::from_le_bytes(bytes);
            }
            head_vectors.push(vec);
        }
        
        // Rebuild head index from head vectors
        let mut head_id_map = Vec::new();
        for posting in &postings {
            head_id_map.push(posting.head_id);
        }
        
        let num_heads = head_vectors.len();
        let mut head_index = SPTAGBKTIndex::new(32, 2000, 32, 1.0, 500, 32);
        head_index.build(head_vectors, 2);
        
        // In on-demand mode, clear posting lists and vectors to save memory
        let (postings, full_vectors) = if on_demand {
            println!("Index loaded in ON-DEMAND mode from {}", path);
            println!("  Heads: {} (in memory)", num_heads);
            println!("  Posting lists: {} (on disk)", list_infos.len());
            println!("  Vectors: {} (on disk)", full_vectors.len());
            (Vec::new(), Vec::new())
        } else {
            println!("Index loaded in IN-MEMORY mode from {}", path);
            println!("  Vectors: {}", full_vectors.len());
            println!("  Heads: {}", num_heads);
            println!("  Postings: {}", postings.len());
            (postings, full_vectors)
        };
        
        Ok(Self {
            head_index,
            head_id_map,
            postings,
            full_vectors,
            metric,
            quantization,
            ratio: meta["ratio"].as_f64().unwrap() as f32,
            replica_count: meta["replica_count"].as_u64().unwrap() as usize,
            posting_vector_limit: meta["posting_vector_limit"].as_u64().unwrap() as usize,
            num_heads_to_search: meta["num_heads_to_search"].as_u64().unwrap() as usize,
            internal_result_num: 64,
            hbc_sample_size: Some(100_000),
            list_infos,
            index_path: Some(path.to_string()),
            on_demand_mode: on_demand,
            dim,
        })
    }
    
    /// Create optimized async storage for on-demand loading
    pub fn create_async_storage(&self) -> Option<crate::spann::OptimizedAsyncStorage> {
        if let Some(ref path) = self.index_path {
            Some(crate::spann::OptimizedAsyncStorage::new(
                path.clone(),
                self.list_infos.clone(),
                true, // enable compression
            ))
        } else {
            None
        }
    }
}
