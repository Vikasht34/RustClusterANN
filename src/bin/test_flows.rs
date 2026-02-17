use rustsptag::spann::{SPANNIndex, DistanceMetric, QuantizationType};

fn main() {
    println!("=== Testing All Flows ===\n");
    
    // Create small test data
    let vectors: Vec<Vec<f32>> = (0..100).map(|i| {
        vec![i as f32; 10]
    }).collect();
    
    let query = vec![50.0; 10];
    
    // Test 1: Regular index (no quantization)
    println!("1. Testing regular index (no quantization)...");
    let mut index1 = SPANNIndex::new();
    index1.set_metric(DistanceMetric::L2);
    index1.build(vectors.clone());
    let results1 = index1.search(&query, 5);
    println!("   Results: {} vectors found", results1.len());
    assert_eq!(results1.len(), 5);
    println!("   ✓ Regular search works\n");
    
    // Test 2: Quantized index with reranking
    println!("2. Testing quantized index with reranking...");
    let mut index2 = SPANNIndex::new();
    index2.set_metric(DistanceMetric::L2);
    index2.set_quantization(QuantizationType::OneBit);
    index2.build(vectors.clone());
    
    // Regular search on quantized
    let results2a = index2.search(&query, 5);
    println!("   Quantized search: {} vectors", results2a.len());
    assert_eq!(results2a.len(), 5);
    
    // Reranking search
    let results2b = index2.search_with_rerank(&query, 5, 3);
    println!("   Reranked search: {} vectors", results2b.len());
    assert_eq!(results2b.len(), 5);
    println!("   ✓ Quantized + reranking works\n");
    
    // Test 3: Save and load
    println!("3. Testing save/load...");
    index1.save("/tmp/test_index.bin", false, false).unwrap();
    let loaded = SPANNIndex::load("/tmp/test_index.bin").unwrap();
    let results3 = loaded.search(&query, 5);
    println!("   Loaded index search: {} vectors", results3.len());
    assert_eq!(results3.len(), 5);
    println!("   ✓ Save/load works\n");
    
    // Test 4: Save quantized with vectors file
    println!("4. Testing quantized save with .vectors file...");
    index2.save("/tmp/test_quant.bin", false, false).unwrap();
    
    // Check if .vectors file was created
    let vectors_exists = std::path::Path::new("/tmp/test_quant.bin.vectors").exists();
    println!("   .vectors file created: {}", vectors_exists);
    assert!(vectors_exists, "Vectors file should be created for quantized index");
    println!("   ✓ Quantized save works\n");
    
    // Test 5: Load quantized and rerank
    println!("5. Testing load quantized and rerank...");
    let loaded_quant = SPANNIndex::load("/tmp/test_quant.bin").unwrap();
    let results5 = loaded_quant.search_with_rerank(&query, 5, 3);
    println!("   Reranked after load: {} vectors", results5.len());
    assert_eq!(results5.len(), 5);
    println!("   ✓ Load + rerank works\n");
    
    println!("=== All Tests Passed! ===");
}
