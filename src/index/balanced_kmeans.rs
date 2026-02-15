/// Balanced K-means with lambda factor (SPTAG implementation)
use crate::simd;
use rand::seq::SliceRandom;
use rand::thread_rng;

pub struct BalancedKMeans {
    pub centers: Vec<Vec<f32>>,
    pub assignments: Vec<usize>,
    pub counts: Vec<usize>,
}

impl BalancedKMeans {
    pub fn new(k: usize, dim: usize) -> Self {
        Self {
            centers: vec![vec![0.0; dim]; k],
            assignments: Vec::new(),
            counts: vec![0; k],
        }
    }

    /// Cluster with automatic lambda selection (SPTAG: FIXED 1000 samples)
    pub fn fit(&mut self, data: &[Vec<f32>], max_iters: usize) -> f32 {
        use rand::seq::SliceRandom;
        let mut rng = thread_rng();
        
        println!("  Selecting lambda from {} vectors...", data.len());
        
        // SPTAG uses FIXED 1000 samples, not scaled
        let samples = 1000.min(data.len());
        
        // Sample for lambda selection
        let mut sample_indices: Vec<usize> = (0..data.len()).collect();
        sample_indices.shuffle(&mut rng);
        sample_indices.truncate(samples);
        
        let sample_data: Vec<Vec<f32>> = sample_indices.iter()
            .map(|&i| data[i].clone())
            .collect();
        
        // Try 7 lambda values: 0.001 to 1000
        let mut best_lambda = 1.0;
        let mut best_std = f32::MAX;
        
        println!("  Trying 7 lambda values on {} samples...", samples);
        for exp in -3..=3 {
            let lambda = 10.0_f32.powi(exp);
            print!("    lambda={:.3}...", lambda);
            let std = self.try_clustering_sptag(&sample_data, data, lambda, 10);
            println!(" std/avg={:.3}", std);
            
            if std < best_std {
                best_std = std;
                best_lambda = lambda;
            }
        }
        
        println!("  Selected lambda: {:.3}", best_lambda);
        println!("  Final clustering on {} vectors...", data.len());
        
        // Final clustering with selected lambda
        self.fit_with_lambda_sptag(data, best_lambda, max_iters)
    }

    /// SPTAG algorithm: train on samples ONLY (don't assign all data!)
    fn try_clustering_sptag(&self, samples: &[Vec<f32>], _all_data: &[Vec<f32>], lambda: f32, max_iters: usize) -> f32 {
        let mut temp = Self::new(self.centers.len(), samples[0].len());
        temp.centers = self.centers.clone();
        
        // Train on samples only
        temp.fit_with_lambda(samples, lambda, max_iters);
        
        // Return balance metric FROM SAMPLES (not all data!)
        let avg = samples.len() as f32 / temp.counts.len() as f32;
        let variance: f32 = temp.counts.iter()
            .map(|&c| (c as f32 - avg).powi(2))
            .sum::<f32>() / temp.counts.len() as f32;
        
        variance.sqrt() / avg
    }

    /// SPTAG fit: train on samples, assign all
    fn fit_with_lambda_sptag(&mut self, data: &[Vec<f32>], lambda: f32, max_iters: usize) -> f32 {
        use rand::seq::SliceRandom;
        let mut rng = thread_rng();
        
        // Train on samples
        let samples = 1000.min(data.len());
        let mut sample_indices: Vec<usize> = (0..data.len()).collect();
        sample_indices.shuffle(&mut rng);
        sample_indices.truncate(samples);
        
        let sample_data: Vec<Vec<f32>> = sample_indices.iter()
            .map(|&i| data[i].clone())
            .collect();
        
        self.fit_with_lambda(&sample_data, lambda, max_iters);
        
        // Assign all data to learned centers
        self.assignments = vec![0; data.len()];
        self.counts.fill(0);
        
        for (i, vec) in data.iter().enumerate() {
            let mut best_cluster = 0;
            let mut best_dist = f32::MAX;
            
            for j in 0..self.centers.len() {
                let dist = simd::l2_distance(vec, &self.centers[j]);
                if dist < best_dist {
                    best_dist = dist;
                    best_cluster = j;
                }
            }
            
            self.assignments[i] = best_cluster;
            self.counts[best_cluster] += 1;
        }
        
        // Return objective
        let mut total_dist = 0.0;
        for (i, vec) in data.iter().enumerate() {
            total_dist += simd::l2_distance(vec, &self.centers[self.assignments[i]]);
        }
        total_dist
    }

    /// Try clustering with given lambda and return cluster size std/avg
    fn try_clustering(&self, data: &[Vec<f32>], lambda: f32, max_iters: usize) -> f32 {
        let mut temp = Self::new(self.centers.len(), data[0].len());
        temp.centers = self.centers.clone();
        temp.fit_with_lambda(data, lambda, max_iters);
        
        // Compute std/avg of cluster sizes
        let avg = data.len() as f32 / temp.counts.len() as f32;
        let variance: f32 = temp.counts.iter()
            .map(|&c| (c as f32 - avg).powi(2))
            .sum::<f32>() / temp.counts.len() as f32;
        
        variance.sqrt() / avg
    }

    /// Fit with specific lambda factor
    pub fn fit_with_lambda(&mut self, data: &[Vec<f32>], lambda: f32, max_iters: usize) -> f32 {
        let k = self.centers.len();
        let dim = data[0].len();
        
        // Initialize centers with k-means++
        self.initialize_centers_kmeans_pp(data);
        
        self.assignments = vec![0; data.len()];
        self.counts = vec![0; k];
        
        for iter in 0..max_iters {
            // Assignment step with lambda penalty
            let mut changed = false;
            self.counts.fill(0);
            
            for (i, vec) in data.iter().enumerate() {
                let mut best_cluster = 0;
                let mut best_dist = f32::MAX;
                
                for j in 0..k {
                    // Distance with lambda penalty for cluster size
                    let dist = simd::l2_distance(vec, &self.centers[j]) 
                             + lambda * self.counts[j] as f32;
                    
                    if dist < best_dist {
                        best_dist = dist;
                        best_cluster = j;
                    }
                }
                
                if self.assignments[i] != best_cluster {
                    self.assignments[i] = best_cluster;
                    changed = true;
                }
                self.counts[best_cluster] += 1;
            }
            
            if !changed {
                break;
            }
            
            // Update centers
            let mut new_centers = vec![vec![0.0; dim]; k];
            
            for (i, vec) in data.iter().enumerate() {
                let cluster = self.assignments[i];
                for d in 0..dim {
                    new_centers[cluster][d] += vec[d];
                }
            }
            
            for j in 0..k {
                if self.counts[j] > 0 {
                    for d in 0..dim {
                        self.centers[j][d] = new_centers[j][d] / self.counts[j] as f32;
                    }
                }
            }
        }
        
        // Return final objective
        let mut total_dist = 0.0;
        for (i, vec) in data.iter().enumerate() {
            total_dist += simd::l2_distance(vec, &self.centers[self.assignments[i]]);
        }
        total_dist
    }

    /// K-means++ initialization
    fn initialize_centers_kmeans_pp(&mut self, data: &[Vec<f32>]) {
        use rand::Rng;
        let mut rng = thread_rng();
        let k = self.centers.len();
        
        // First center: random
        let first_idx = rng.gen_range(0..data.len());
        self.centers[0] = data[first_idx].clone();
        
        // Remaining centers: weighted by distance
        for i in 1..k {
            let mut distances: Vec<f32> = data.iter()
                .map(|vec| {
                    (0..i).map(|j| simd::l2_distance(vec, &self.centers[j]))
                        .min_by(|a, b| a.partial_cmp(b).unwrap())
                        .unwrap()
                })
                .collect();
            
            // Weighted random selection
            let total: f32 = distances.iter().sum();
            let mut target = rng.gen::<f32>() * total;
            
            for (idx, &dist) in distances.iter().enumerate() {
                target -= dist;
                if target <= 0.0 {
                    self.centers[i] = data[idx].clone();
                    break;
                }
            }
        }
    }

    /// Find nearest center to a vector
    pub fn find_nearest(&self, vec: &[f32]) -> usize {
        self.centers.iter()
            .enumerate()
            .map(|(i, c)| (i, simd::l2_distance(vec, c)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap().0
    }

    /// Get cluster statistics
    pub fn stats(&self) -> (usize, usize, f32, f32) {
        let max = *self.counts.iter().max().unwrap_or(&0);
        let min = *self.counts.iter().filter(|&&c| c > 0).min().unwrap_or(&0);
        let avg = self.assignments.len() as f32 / self.counts.len() as f32;
        let variance: f32 = self.counts.iter()
            .map(|&c| (c as f32 - avg).powi(2))
            .sum::<f32>() / self.counts.len() as f32;
        let std_ratio = variance.sqrt() / avg;
        
        (max, min, avg, std_ratio)
    }
}
