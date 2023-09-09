use fri::{batch_proof::BatchOpening, BatchProver};

use crypto::{ElementHasher, RandomCoin};
use math::{fft, polynom::syn_div_with_rem, FieldElement, StarkField};

use crate::polyutils::random_poly;

// Struct for sumcheck proofs
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SCProof<B>
where
    B: StarkField,
{
    // A sumcheck proof for the polynomial p
    // q is a random masking polynomial
    pub sc_p: String, //labels of the polynomials
    pub sc_q: String,
    pub sc_q1: String,
    pub sc_q2: String,
    pub alpha: B, //sum on p
    pub beta: B,  // sum on q
}

// SUMCHECK
// ================================================================================================
/// Creates a sumcheck proof where the sum check domain is a subset of points in the domain
/// this subset is defined as 1, omega, omega^2 ... where omega = domain_generator^lde_blowup_factor
/// evals_p does not have any error correction i.e. it is of length p_degree+1
// todo: pass a label_p and use that
pub fn sc_prove<B, E, H, R>(
    sum_domain_size: usize,
    evals_p: &Vec<B>,
    label_p: String,
    batch_prover: &mut BatchProver<B, E, H, R>,
    id: String,
) -> SCProof<B>
where
    B: StarkField,
    E: FieldElement<BaseField = B> + math::ExtensionOf<B>,
    H: ElementHasher<BaseField = B>,
    R: RandomCoin<BaseField = B, Hasher = H>,
{
    // assert that sum_domain_size is at most evals_p.len()
    assert!(
        sum_domain_size <= evals_p.len(),
        "sum_domain_size must be at most p_degree+1"
    );

    let p_degree = evals_p.len() - 1;
    // todo: this can be cached to save a little bit of work
    let inv_twiddles = fft::get_inv_twiddles(p_degree + 1);

    // sample a random polynomial q of degree p_degree
    let (coeff_q, evals_q) = random_poly(p_degree);

    // evaluate p and q on sum_domain to get alpha and beta respectively
    let mut alpha = B::ZERO;
    let mut beta = B::ZERO;

    let step_size = (p_degree + 1) / sum_domain_size;
    for i in 0..sum_domain_size {
        alpha += evals_p[i * step_size];
        beta += evals_q[i * step_size];
    }

    // sample random challenge c using the channel
    // todo: do actual Fiat-Shamir here
    let c = B::from(7u32); //dummy challenge

    // gamma = c.alpha+beta
    let gamma = c * alpha + beta;
    let gammabys = gamma / B::from(sum_domain_size as u64);

    // set f(x) = c.p(x) + q(x) and
    let mut evals_f = vec![B::ZERO; evals_p.len()];
    for i in 0..(p_degree + 1) {
        evals_f[i] = c * evals_p[i] + evals_q[i];
    }

    // compute quotient polynomial Q1 and Q2 such that f(x) = gamma/s + x.Q1 + Z.Q2,
    // get quotient and remainder of f(x) / Z where Z = x^sum_domain_size - 1. they come padded to sum_domain_size
    // Q2 = quotient
    // Q1 = (remainder-gamma/sum_domain_size)/x
    let mut coeff_f = evals_f.clone();
    fft::interpolate_poly(&mut coeff_f, &inv_twiddles);

    // comes padded to sum_domain_size
    let (coeff_q2, mut coeff_x_q1) = syn_div_with_rem(&coeff_f, sum_domain_size, B::ONE);

    debug_assert_eq!(
        gammabys, coeff_x_q1[0],
        "gammabys must be equal to coeff_x_q1[0]."
    );
    coeff_x_q1[0] = B::ZERO; //drop the constant term. this is x.q1

    // create labels for the polynomials
    let sc_q = id.to_owned() + &"sc_q".to_owned();
    let sc_q1 = id.to_owned() + &"sc_q1".to_owned();
    let sc_q2 = id.to_owned() + &"sc_q2".to_owned();

    // add the polynomials to the batch_prover
    batch_prover.insert(sc_q.clone(), coeff_q);
    batch_prover.insert(sc_q1.clone(), coeff_x_q1);
    batch_prover.insert(sc_q2.clone(), coeff_q2);

    SCProof {
        sc_p: label_p.clone(),
        sc_q,
        sc_q1,
        sc_q2,
        alpha,
        beta,
    }
}

/// Verifies a sumcheck proof
pub fn sc_verify<B, E, H, R>(
    sc_proof: &SCProof<B>,
    sum_domain_size: usize,
    batch_opening: &BatchOpening<B, E, H, R>,
) where
    B: StarkField,
    E: FieldElement<BaseField = B> + math::ExtensionOf<B>,
    H: ElementHasher<BaseField = B>,
    R: RandomCoin<BaseField = B, Hasher = H>,
{
    // retrive openings
    let &p_open = batch_opening.poly_openings.get(&sc_proof.sc_p).unwrap();
    let &q_open = batch_opening.poly_openings.get(&sc_proof.sc_q).unwrap();
    let &q1_open = batch_opening.poly_openings.get(&sc_proof.sc_q1).unwrap();
    let &q2_open = batch_opening.poly_openings.get(&sc_proof.sc_q2).unwrap();

    // check the polynomial identity testing
    // c.p(r) + q(r) = gammabys +r.q1(r) + q2(r).z(r)
    let c = B::from(7u32);
    let gammabys = (c * sc_proof.alpha + sc_proof.beta) / B::from(sum_domain_size as u64);

    let lhs = q1_open
        + E::from(gammabys)
        + q2_open * (E::from(7u32).exp_vartime((sum_domain_size as u64).into()) - E::ONE);

    let rhs = p_open * E::from(c) + q_open;

    assert_eq!(lhs, rhs, "PIT failed");
}

// PROVE/VERIFY TEST
// ================================================================================================

#[cfg(test)]
mod tests {
    use crate::polyutils::random_poly;

    use super::*;
    use crypto::{hashers::Blake3_256, DefaultRandomCoin};
    use fri::{DefaultProverChannel, FriOptions};
    use math::fields::f128::BaseElement;

    type Blake3 = Blake3_256<BaseElement>;

    pub const LDE_BLOWUP: usize = 1 << 2;
    pub const FOLDING_FACTOR: usize = 1 << 2;
    pub const MAX_REMAINDER_DEGREE: usize = 1;

    #[test]
    fn sc_prove_verify() {
        let options = FriOptions::new(LDE_BLOWUP, FOLDING_FACTOR, MAX_REMAINDER_DEGREE);
        let p_degree = (1 << 12) - 1;
        let sum_domain_size = 1 << 12;

        let (coeff_p, evals_p) = random_poly(p_degree);

        let mut batch_prover = BatchProver::<
            BaseElement,
            BaseElement,
            Blake3,
            DefaultRandomCoin<Blake3_256<BaseElement>>,
        >::new(options.clone(), p_degree);

        let id = "test";

        // add p(x) to the batch_prover
        let sc_p = id.to_owned() + &"sc_p".to_owned();
        batch_prover.insert(sc_p.clone(), coeff_p.clone());

        let sc_proof = sc_prove(
            sum_domain_size,
            &evals_p,
            sc_p,
            &mut batch_prover,
            id.to_owned(),
        );

        let mut channel = DefaultProverChannel::new((p_degree + 1) * options.blowup_factor(), 32);
        let batch_opening =
            batch_prover.batch_prove(&mut channel, BaseElement::from(7u32), p_degree);

        batch_opening.verify(p_degree, options.clone());

        sc_verify(&sc_proof, sum_domain_size, &batch_opening);
    }
}
