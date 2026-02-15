use rustsptag::lib::MultiBitQuantizer;

fn main() {
    // Simple test case
    let dim = 4;
    let data = vec![
        vec![1.0, 2.0, -1.0, 0.5],
        vec![0.5, -1.0, 2.0, 1.0],
    ];
    let centroid = vec![0.75, 0.5, 0.5, 0.75];
    
    let mut quantizer = MultiBitQuantizer::new(dim, 2);
    quantizer.fit(&data, &centroid);
    
    println!("Vector 0: {:?}", data[0]);
    println!("Centroid: {:?}", centroid);
    println!("Residual: {:?}", data[0].iter().zip(centroid.iter()).map(|(d, c)| d - c).collect::<Vec<_>>());
}
