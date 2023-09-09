use ark_std::{end_timer, start_timer};
use fri::utils::transpose;
use math::StarkField;

const D: usize = 1 << 8;
const N: usize = 1 << 20;
const B: usize = 1 << 10;

fn train<B: StarkField>() {
    let mut x: Vec<Vec<B>> = Vec::new();
    for _ in 0..B {
        x.push(rand_vector_range(D, 16));
    }

    let y = rand_vector_range(B, 16);

    // Initialize the weights and learning rate
    let mut weights = rand_vector_range(D, 16);

    // Perform mini-batch gradient descent
    let num_batches = N / B;

    let training_timer = start_timer!(|| "training");
    for i in 0..num_batches {
        println!("trianing batch: {}", i);
        // Compute the gradient for the current mini-batch
        let gradient = compute_gradient(&weights, &x, &y);

        // Update the weights using the gradient and learning rate
        for j in 0..D {
            weights[j] -= gradient[j];
        }
    }
    end_timer!(training_timer);
}

fn compute_gradient<B: StarkField>(weights: &Vec<B>, x: &Vec<Vec<B>>, y: &Vec<B>) -> Vec<B> {
    let mut predictions = vec![B::ZERO; B];
    for i in 0..B {
        for j in 0..D {
            predictions[i] += weights[j] * x[i][j];
        }
    }

    let mut error = vec![B::ZERO; B];
    for i in 0..B {
        error[i] = predictions[i] - y[i];
    }

    let xt = transpose(x.clone());

    let mut gradient = vec![B::ZERO; D];
    for i in 0..D {
        for j in 0..B {
            gradient[i] += xt[i][j] * error[j];
        }
    }

    gradient
}

use math::fields::f128::BaseElement;
use rand_utils::rand_vector_range;

fn main() {
    train::<BaseElement>();
}
