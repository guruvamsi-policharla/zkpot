use math::{fft, StarkField};
use rand_utils::rand_vector;

/// returns (coeff, evals) a randomly sampled polynomial of degree `degree`
/// will panic if degree+1 is not a power of 2
pub fn random_poly<B: StarkField>(degree: usize) -> (Vec<B>, Vec<B>) {
    let coeff_p = rand_vector::<B>(degree + 1);
    let mut evals_p = coeff_p.clone();

    let twiddles = fft::get_twiddles::<B>(degree + 1);
    fft::evaluate_poly(&mut evals_p, &twiddles);

    (coeff_p, evals_p)
}
