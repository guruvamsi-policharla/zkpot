use ndarray::{Array, Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::time::Instant;

const D: usize = 1 << 8;
const N: usize = 1 << 20;
const B: usize = 1 << 10;

fn main() {
    let x = Array::random((B, D), Uniform::new(-1.0, 1.0));
    let y = Array::random(B, Uniform::new(-1.0, 1.0));

    // Initialize the weights and learning rate
    let mut weights = Array::random(D, Uniform::new(-1.0, 1.0));

    let learning_rate = 0.1;

    // Perform mini-batch gradient descent
    let num_batches = N / B;

    let before = Instant::now();
    for i in 0..num_batches {
        println!("trianing batch: {}", i);
        // Compute the gradient for the current mini-batch
        let gradient = learning_rate * compute_gradient(&weights, &x, &y);

        // Update the weights using the gradient and learning rate
        weights -= gradient[0];
    }
    println!("Elapsed time: {:.2?}", before.elapsed());
}

fn compute_gradient(weights: &Array1<f64>, x: &Array2<f64>, y: &Array1<f64>) -> Array1<f64> {
    let predictions = x.dot(weights);
    let error = &predictions - y;
    let mut gradient = x.t().dot(&error) / B as f64;

    for i in 0..D {
        if gradient[i] < -0.5 {
            gradient[i] = 0.0;
        } else if gradient[i] > 0.5 {
            gradient[i] = 1.0;
        } else {
            gradient[i] += gradient[i] + 0.5
        }
    }

    gradient
}
