/// Simple k-means test to debug performance
use rustsptag::index::BalancedKMeans;
use std::time::Instant;

fn main() {
    println!("Testing k-means performance...\n");

    // Test 1: Small data
    println!("Test 1: 1000 vectors, K=32");
    let data: Vec<Vec<f32>> = (0..1000)
        .map(|_| (0..128).map(|_| rand::random::<f32>()).collect())
        .collect();
    
    let start = Instant::now();
    let mut kmeans = BalancedKMeans::new(32, 128);
    
    // Manual fit without lambda selection
    kmeans.fit_with_lambda(&data, 1.0, 20);
    println!("Time: {:.2?}\n", start.elapsed());

    // Test 2: Lambda selection on sample
    println!("Test 2: Lambda selection on 1000 samples");
    let start = Instant::now();
    let mut kmeans2 = BalancedKMeans::new(32, 128);
    kmeans2.fit(&data, 20);
    println!("Time: {:.2?}\n", start.elapsed());

    // Test 3: Larger data
    println!("Test 3: 10000 vectors, K=100");
    let data2: Vec<Vec<f32>> = (0..10000)
        .map(|_| (0..128).map(|_| rand::random::<f32>()).collect())
        .collect();
    
    let start = Instant::now();
    let mut kmeans3 = BalancedKMeans::new(100, 128);
    kmeans3.fit_with_lambda(&data2, 1.0, 20);
    println!("Time: {:.2?}", start.elapsed());
}
