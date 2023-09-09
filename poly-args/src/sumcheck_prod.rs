use fri::{batch_proof::BatchOpening, utils::low_degree_extension, BatchProver};

use crypto::{ElementHasher, RandomCoin};
use math::{
    fft,
    polynom::{eval, syn_div_with_rem},
    FieldElement, StarkField,
};

use super::polyutils::random_poly;

// Struct for sumcheck proofs
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SCPProof<B>
where
    B: StarkField,
{
    // A sumcheck proof for the polynomial f = (k1.p1(x)+k2.p2(x)+...).w(x)
    // k is publicly known vector of coefficients
    pub sc_p: Vec<String>, //labels of the polynomials
    pub sc_q: String,
    pub sc_q1: String,
    pub sc_q2: String,
    pub alpha: B, //sum on p
    pub beta: B,  // sum on q
}

// SUMCHECK
// ================================================================================================
/// Creates a sumcheck proof where the sum check domain is a subset of points in the domain
/// this subset is defined as the entire p_evals domain [1, omega, omega^2 ... omega^{p_degree-1}]
/// evals_p and evals_w do not have any error correction i.e. they both have length p_degree+1
/// NOTE: this code only works when the sum_domain_size = p_degree+1.
/// for other sizes, the degree of Q2 is not p_degree+1 and will need a separate polynomial commitment
pub fn scp_prove<B, E, H, R>(
    evals_p: &Vec<Vec<B>>,
    labels_p: &Vec<String>,
    evals_w: &Vec<B>,
    k: &Vec<B>, // defines the linear combination of the polynomials
    batch_prover: &mut BatchProver<B, E, H, R>,
    id: String,
) -> SCPProof<B>
where
    B: StarkField,
    E: FieldElement<BaseField = B> + math::ExtensionOf<B>,
    H: ElementHasher<BaseField = B>,
    R: RandomCoin<BaseField = B, Hasher = H>,
{
    debug_assert_eq!(evals_p.len(), k.len());
    let num_poly = evals_p.len();
    let p_degree = evals_p[0].len() - 1; //implicitly assumes that all polynomials have same degree
    let sum_domain_size = p_degree + 1;

    // sample a random polynomial q of degree p_degree
    let (coeff_q, evals_q) = random_poly(p_degree);

    // evaluate p and q on sum_domain to get alpha and beta respectively
    let mut alpha = B::ZERO;
    let mut beta = B::ZERO;

    let step_size = (p_degree + 1) / sum_domain_size;
    for i in 0..sum_domain_size {
        let mut alphai = B::ZERO;
        for j in 0..num_poly {
            alphai += k[j] * evals_p[j][i * step_size];
        }
        alpha += alphai * evals_w[i * step_size];
        beta += evals_q[i * step_size];
    }

    // sample random challenge c using the channel
    // todo: do actual Fiat-Shamir here
    let c = B::from(7u32); //dummy challenge

    // set f(x) = c.(k1.p1(x)+k2.p2(x)+...).w(x) + q(x)
    // note: here the degree of f(x) increases (doubles) and hence we need to extend the evals_f
    // we want to minimize the number of ffts and hence we do the following
    // first compute evals of p(x) = k1.p1(x)+k2.p2(x)+...
    // then extend evals of p(x), w(x) and q(x) to 2*(p_degree+1) using low_degree_extension
    // then compute evals of f(x) = c.p(x).w(x) + q(x)
    let mut evals_sump = vec![B::ZERO; evals_p[0].len()];
    for j in 0..num_poly {
        for i in 0..(p_degree + 1) {
            evals_sump[i] += k[j] * evals_p[j][i];
        }
    }

    // extend evals of p(x), w(x) and q(x) to 2*(p_degree+1) using low_degree_extension
    let mut evals_sump_lde = evals_sump.clone();
    low_degree_extension(&mut evals_sump_lde, 2 * (p_degree + 1));

    let mut evals_w_lde = evals_w.clone();
    low_degree_extension(&mut evals_w_lde, 2 * (p_degree + 1));

    let mut evals_q_lde = evals_q.clone();
    low_degree_extension(&mut evals_q_lde, 2 * (p_degree + 1));

    let mut evals_f = vec![B::ZERO; evals_p[0].len() * 2];
    for i in 0..2 * (p_degree + 1) {
        evals_f[i] = c * evals_w_lde[i] * evals_sump_lde[i] + evals_q_lde[i];
    }

    // compute quotient polynomial Q1 and Q2 such that f(x) = gamma/s + x.Q1 + Z.Q2,
    // get quotient and remainder of f(x) / Z where Z = x^sum_domain_size - 1. they come padded to sum_domain_size
    // Q2 = quotient
    // Q1 = (remainder-gamma/sum_domain_size)/x
    let mut coeff_f = evals_f.clone();
    let inv_twiddles = fft::get_inv_twiddles(2 * (p_degree + 1));
    fft::interpolate_poly(&mut coeff_f, &inv_twiddles);

    // comes padded to sum_domain_size
    let (coeff_q2, mut coeff_x_q1) = syn_div_with_rem(&coeff_f, sum_domain_size, B::ONE);

    coeff_x_q1[0] = B::ZERO; //drop the constant term. this is x.q1

    // create labels for the polynomials
    let sc_q = format!("{}_sc_q", id);
    let sc_q1 = format!("{}_sc_q1", id);
    let sc_q2 = format!("{}_sc_q2", id);

    batch_prover.insert(sc_q.clone(), coeff_q);
    batch_prover.insert(sc_q1.clone(), coeff_x_q1);
    batch_prover.insert(sc_q2.clone(), coeff_q2);

    SCPProof {
        sc_p: labels_p.clone(),
        sc_q,
        sc_q1,
        sc_q2,
        alpha,
        beta,
    }
}

/// Verifies a sumcheck_prod proof
pub fn scp_verify<B, E, H, R>(
    scp_proof: &SCPProof<B>,
    p_degree: usize,
    evals_w: &Vec<B>,
    k: &Vec<B>,
    batch_opening: &BatchOpening<B, E, H, R>,
    num_poly: usize,
) where
    B: StarkField,
    E: FieldElement<BaseField = B> + math::ExtensionOf<B>,
    H: ElementHasher<BaseField = B>,
    R: RandomCoin<BaseField = B, Hasher = H>,
{
    let open_pos = E::from(8u32);

    // polyonomial labels
    let mut p_open = Vec::new();
    for i in 0..num_poly {
        p_open.push(*batch_opening.poly_openings.get(&scp_proof.sc_p[i]).unwrap());
    }
    let &q_open = batch_opening.poly_openings.get(&scp_proof.sc_q).unwrap();
    let &q1_open = batch_opening.poly_openings.get(&scp_proof.sc_q1).unwrap();
    let &q2_open = batch_opening.poly_openings.get(&scp_proof.sc_q2).unwrap();

    let inv_twiddles = fft::get_inv_twiddles(p_degree + 1);
    let mut coeff_w = evals_w.clone();
    fft::interpolate_poly(&mut coeff_w, &inv_twiddles);
    let w_open = eval(coeff_w.as_slice(), open_pos);

    let sum_domain_size = p_degree + 1;
    // check the polynomial identity testing
    // c.p(r) + q(r) = gammabys +r.q1(r) + q2(r).z(r)
    let c = B::from(7u32);
    let gammabys = (c * scp_proof.alpha + scp_proof.beta) / B::from(sum_domain_size as u64);

    let lhs = q1_open
        + E::from(gammabys)
        + q2_open * (open_pos.exp_vartime((sum_domain_size as u64).into()) - E::ONE);

    let mut rhs = E::ZERO;
    for i in 0..num_poly {
        rhs += E::from(k[i]) * p_open[i];
    }
    rhs *= w_open * E::from(c);
    rhs += q_open;

    assert_eq!(lhs, rhs, "PIT failed");
}

// PROVE/VERIFY TEST
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crypto::{hashers::Blake3_256, DefaultRandomCoin};
    use fri::{DefaultProverChannel, FriOptions};
    use math::fields::f128::BaseElement;
    use rand_utils::rand_vector;

    type Blake3 = Blake3_256<BaseElement>;

    pub const LDE_BLOWUP: usize = 1 << 2;
    pub const FOLDING_FACTOR: usize = 1 << 2;
    pub const MAX_REMAINDER_DEGREE: usize = 1;

    #[test]
    fn scp_prove_verify() {
        let options = FriOptions::new(LDE_BLOWUP, FOLDING_FACTOR, MAX_REMAINDER_DEGREE);
        let size = 12;
        let p_degree = (1 << size) - 1;

        let mut evals_p: Vec<Vec<BaseElement>> = Vec::new();
        let mut coeff_p: Vec<Vec<BaseElement>> = Vec::new();
        let num_poly = 10;

        for _ in 0..num_poly {
            let (coeffs, evals) = random_poly(p_degree);
            evals_p.push(evals);
            coeff_p.push(coeffs);
        }

        // let (_, evals_w) = random_poly(p_degree);
        let evals_w = vec![BaseElement::ONE; p_degree + 1];

        let k = rand_vector(num_poly);

        let mut batch_prover = BatchProver::<
            BaseElement,
            BaseElement,
            Blake3,
            DefaultRandomCoin<Blake3_256<BaseElement>>,
        >::new(options.clone(), p_degree);

        let id = "test";

        // add p polynomials in the batch prover
        let mut sc_p = Vec::new();
        for i in 0..num_poly {
            sc_p.push(format!("{}_sc_p_{}", id, i));
        }

        for i in 0..num_poly {
            batch_prover.insert(sc_p[i].clone(), coeff_p[i].clone());
        }

        let scp_proof = scp_prove::<
            BaseElement,
            BaseElement,
            Blake3,
            DefaultRandomCoin<Blake3_256<BaseElement>>,
        >(
            &evals_p,
            &sc_p,
            &evals_w,
            &k,
            &mut batch_prover,
            id.to_owned(),
        );

        let open_pos = BaseElement::from(8u32);
        let mut channel = DefaultProverChannel::new((p_degree + 1) * options.blowup_factor(), 32);
        let batch_opening = batch_prover.batch_prove(&mut channel, open_pos, p_degree);
        batch_opening.verify(p_degree, options.clone());
        println!("p_degree: {}", p_degree);
        scp_verify(&scp_proof, p_degree, &evals_w, &k, &batch_opening, num_poly);
    }
}
