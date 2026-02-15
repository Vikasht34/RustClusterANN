use rustsptag::spann::{SPANNIndex, DistanceMetric};
use std::time::Instant;

fn main() {
    println!("=== SPANN Storage Test ===\n");
    
    // Generate test data
    println!("Generating test data...");
    let n = 10_000;
    let dim = 128;
    let mut vectors = Vec::new();
    for i in 0..n {
        let vec: Vec<f32> = (0..dim).map(|j| ((i * dim + j) as f32).sin()).collect();
        vectors.push(vec);
    }
    
    // Build index
    println!("Building SPANN index...");
    let start = Instant::now();
    let mut index = SPANNIndex::new();
    index.set_metric(DistanceMetric::L2);
    index.build(vectors.clone());
    println!("Build time: {:.2}s\n", start.elapsed().as_secs_f32());
    
    // Test 1: Save without compression
    println!("Test 1: Save without compression");
    let start = Instant::now();
    index.save("test_index_nocomp.bin", false, false).unwrap();
    println!("  Save time: {:.2}s", start.elapsed().as_secs_f32());
    let size = std::fs::metadata("test_index_nocomp.bin").unwrap().len();
    println!("  File size: {:.2} MB\n", size as f32 / 1024.0 / 1024.0);
    
    // Test 2: Save with compression
    println!("Test 2: Save with compression");
    let start = Instant::now();
    index.save("test_index_comp.bin", true, false).unwrap();
    println!("  Save time: {:.2}s", start.elapsed().as_secs_f32());
    let size_comp = std::fs::metadata("test_index_comp.bin").unwrap().len();
    println!("  File size: {:.2} MB", size_comp as f32 / 1024.0 / 1024.0);
    println!("  Compression ratio: {:.2}x\n", size as f32 / size_comp as f32);
    
    // Test 3: Save with compression + delta encoding
    println!("Test 3: Save with compression + delta encoding");
    let start = Instant::now();
    index.save("test_index_delta.bin", true, true).unwrap();
    println!("  Save time: {:.2}s", start.elapsed().as_secs_f32());
    let size_delta = std::fs::metadata("test_index_delta.bin").unwrap().len();
    println!("  File size: {:.2} MB", size_delta as f32 / 1024.0 / 1024.0);
    println!("  Compression ratio: {:.2}x\n", size as f32 / size_delta as f32);
    
    // Test 4: Load and verify
    println!("Test 4: Load and verify");
    let start = Instant::now();
    let loaded_index = SPANNIndex::load("test_index_comp.bin").unwrap();
    println!("  Load time: {:.2}s", start.elapsed().as_secs_f32());
    
    // Verify search results match
    let query = &vectors[0];
    let results_original = index.search(query, 10);
    let results_loaded = loaded_index.search(query, 10);
    
    println!("  Original results: {:?}", &results_original[..3]);
    println!("  Loaded results:   {:?}", &results_loaded[..3]);
    
    let matches = results_original.iter()
        .zip(results_loaded.iter())
        .filter(|(a, b)| a.0 == b.0)
        .count();
    println!("  Matching results: {}/10\n", matches);
    
    // Cleanup
    std::fs::remove_file("test_index_nocomp.bin").ok();
    std::fs::remove_file("test_index_nocomp.bin.meta").ok();
    std::fs::remove_file("test_index_comp.bin").ok();
    std::fs::remove_file("test_index_comp.bin.meta").ok();
    std::fs::remove_file("test_index_delta.bin").ok();
    std::fs::remove_file("test_index_delta.bin.meta").ok();
    
    println!("=== Test Complete ===");
}
