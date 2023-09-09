// Cheking consistency of packed secret shares as used by pol

use crypto::{ElementHasher, RandomCoin};
use fri::{batch_proof::BatchOpening, BatchProver};
use math::{batch_inversion, get_power_series, log2, StarkField};
use poly_args::sumcheck_prod::{scp_prove, scp_verify, SCPProof};
use secret_sharing::pss::PackedSharingParams;

// Struct for PSS Consistency proofs
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PSSConsistencyProof<B: StarkField> {
    labels_gi: Vec<String>,
    labels_gv: Vec<String>,
    qvg: Vec<String>,
    scp_proof: SCPProof<B>,
}

/// Proves the consistency of packed secret shares
/// see test case for generation procedure.
/// todo: handle zero knowledge. this is when the polynomials are masked
// todo: make this an impl of PSSConsistencyProof?
pub fn prove_pss_consistency<B, H, R>(
    evals_gi: &Vec<Vec<B>>, // without lde -- shares
    evals_gv: &Vec<Vec<B>>, // without lde -- secrets
    labels_gi: &Vec<String>,
    labels_gv: &Vec<String>,
    batch_prover: &mut BatchProver<B, B, H, R>,
    n: usize, // number of parties
    id: &str,
) -> PSSConsistencyProof<B>
where
    B: StarkField,
    H: ElementHasher<BaseField = B>,
    R: RandomCoin<BaseField = B, Hasher = H>,
{
    let l = evals_gv.len();
    debug_assert_eq!(l, n / 8);

    let s = evals_gi[0].len();

    // generate the dual vector
    let omega = B::get_root_of_unity(log2(n));
    let domain = get_power_series(omega, n);
    let mut dual_vec: Vec<B> = vec![B::ONE; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                dual_vec[i] *= domain[i] - domain[j];
            }
        }
    }
    dual_vec = batch_inversion(&dual_vec);

    let scp_proof = scp_prove(
        &evals_gi,
        &labels_gi,
        &vec![B::ONE; s],
        &dual_vec,
        batch_prover,
        id.to_owned(),
    );

    // todo: handle zero knowledge by using masking polynomials.
    // Use the pack_from_public method to obtain qvg
    PSSConsistencyProof {
        labels_gi: labels_gi.clone(),
        labels_gv: labels_gv.clone(),
        qvg: Vec::new(),
        scp_proof,
    }
}

// this should support the extension field but we would not be able to support optimized FFT based check
// this is okay because in our instantiation we do not use an extension field
pub fn verify_pss_consistency<B, H, R>(
    proof: &PSSConsistencyProof<B>,
    batch_opening: &BatchOpening<B, B, H, R>,
    n: usize, // number of parties
    sum_domain_size: usize,
) where
    B: StarkField,
    H: ElementHasher<BaseField = B>,
    R: RandomCoin<BaseField = B, Hasher = H>,
{
    let l = n / 8;

    // generate the dual vector
    let omega = B::get_root_of_unity(log2(n));
    let domain = get_power_series(omega, n);
    let mut dual_vec: Vec<B> = vec![B::ONE; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                dual_vec[i] *= domain[i] - domain[j];
            }
        }
    }
    dual_vec = batch_inversion(&dual_vec);

    scp_verify(
        &proof.scp_proof,
        sum_domain_size - 1,
        &vec![B::ONE; sum_domain_size],
        &dual_vec,
        &batch_opening,
        n,
    );
    assert_eq!(proof.scp_proof.alpha, B::ZERO);

    // retrieve openings
    let mut gv_open = Vec::new();
    for i in 0..l {
        gv_open.push(
            *batch_opening
                .poly_openings
                .get(&proof.labels_gv[i])
                .unwrap(),
        );
    }

    let mut gi_open = Vec::new();
    for i in 0..n {
        gi_open.push(
            *batch_opening
                .poly_openings
                .get(&proof.labels_gi[i])
                .unwrap(),
        );
    }

    // check that the evaluations are consistent
    // todo: avoid regenerating the PackedSharingParams
    let pp = PackedSharingParams::<B>::new(l);
    let should_be_gi_open = pp.pack_from_public(&gv_open);
    assert_eq!(should_be_gi_open, gi_open);

    // todo: add the check for zero knowledge above will fail when using masked polynomials
}

#[cfg(test)]
mod tests {
    use crypto::{hashers::Blake3_256, DefaultRandomCoin, ElementHasher};
    use fri::{utils::transpose, DefaultProverChannel, FriOptions};
    use math::{fft, fields::f128::BaseElement, StarkField};
    use rand_utils::rand_vector;

    use crate::FRI_QUERIES;

    use super::*;

    type Blake3 = Blake3_256<BaseElement>;

    pub const LDE_BLOWUP: usize = 1 << 3;
    pub const FOLDING_FACTOR: usize = 1 << 2;
    pub const MAX_REMAINDER_DEGREE: usize = 1 << 1;

    #[test]
    fn pss_consistency() {
        let n: usize = 512;
        let s: usize = 16;
        let options = FriOptions::new(LDE_BLOWUP, FOLDING_FACTOR, MAX_REMAINDER_DEGREE);

        let (evals_gi, evals_gv) = pss_consistency_test_case::<BaseElement, Blake3>(n, s);

        // interpolate the polynomials
        let inv_twiddles = fft::get_inv_twiddles(s);
        let mut coeffs_gi = evals_gi.clone();
        coeffs_gi
            .iter_mut()
            .for_each(|evals| fft::interpolate_poly(evals, &inv_twiddles));

        let mut coeffs_gv = evals_gv.clone();
        coeffs_gv
            .iter_mut()
            .for_each(|evals| fft::interpolate_poly(evals, &inv_twiddles));

        let mut batch_prover = BatchProver::<
            BaseElement,
            BaseElement,
            Blake3,
            DefaultRandomCoin<Blake3_256<BaseElement>>,
        >::new(options.clone(), s - 1);

        let id = "test";

        // create labels for the polynomials
        let mut labels_gi = Vec::new();
        for i in 0..n {
            labels_gi.push(format!("{}_gi_{}", id, i));
        }

        let mut labels_gv = Vec::new();
        for i in 0..n / 8 {
            labels_gv.push(format!("{}_gv_{}", id, i));
        }

        // add the polynomials to the batch_prover
        for i in 0..n {
            batch_prover.insert(labels_gi[i].clone(), coeffs_gi[i].clone());
        }

        for i in 0..n / 8 {
            batch_prover.insert(labels_gv[i].clone(), coeffs_gv[i].clone());
        }

        let proof = prove_pss_consistency::<
            BaseElement,
            Blake3,
            DefaultRandomCoin<Blake3_256<BaseElement>>,
        >(
            &evals_gi,
            &evals_gv,
            &labels_gi,
            &labels_gv,
            &mut batch_prover,
            n,
            "test",
        );

        let open_pos = BaseElement::from(8u32);
        let mut channel = DefaultProverChannel::new((s) * options.blowup_factor(), FRI_QUERIES);
        let batch_opening = batch_prover.batch_prove(&mut channel, open_pos, s - 1);
        batch_opening.verify(s - 1, options.clone());

        verify_pss_consistency::<BaseElement, Blake3, DefaultRandomCoin<Blake3_256<BaseElement>>>(
            &proof,
            &batch_opening,
            n,
            s,
        );
    }

    // Utils for testing ========================================================
    /// Generates a test case for consistency of packed shares proof
    fn pss_consistency_test_case<B, H>(n: usize, s: usize) -> (Vec<Vec<B>>, Vec<Vec<B>>)
    where
        B: StarkField,
        H: ElementHasher<BaseField = B::BaseField>,
    {
        // check that n is a power of 2
        assert_eq!(n & (n - 1), 0);

        // l is packing_factor
        let l = n / 8;

        // sample a matrix of s x l random values (the secrets)
        let mut secrets: Vec<Vec<B>> = Vec::new();
        for _ in 0..s {
            secrets.push(rand_vector(l));
        }

        let pp = PackedSharingParams::<B>::new(l);

        // pack the secrets
        let mut shares: Vec<Vec<B>> = Vec::new();

        // n x s matrix of packed shares
        for i in 0..s {
            shares.push(pp.pack_from_public(&secrets[i]));
        }

        // perform low_degree_extension and commit ================================
        let evals_gi = transpose(shares);
        let evals_gv = transpose(secrets);

        #[cfg(debug_assertions)]
        {
            use math::polynom::eval;

            let omega = B::get_root_of_unity(log2(n));
            let domain = get_power_series(omega, n);

            // test that inner product of evals_gi with a vector from dual space is zero on the roots of unity

            // generate the dual vector
            let mut dual_vec: Vec<B> = vec![B::ONE; n];
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        dual_vec[i] *= domain[i] - domain[j];
                    }
                }
            }
            dual_vec = batch_inversion(&dual_vec);

            // check that the inner product is zero
            for i in 0..s {
                let mut inner_prod = B::ZERO;
                for j in 0..n {
                    inner_prod += evals_gi[j][i] * dual_vec[j];
                }

                assert_eq!(inner_prod, B::ZERO);
            }

            let eps = B::from(3u32);

            // obtain coefficients after interpolating gi, gv across s
            let mut gv_coeffs: Vec<Vec<B>> = evals_gv.clone();
            let inv_twiddles = fft::get_inv_twiddles(s);
            for i in 0..l {
                debug_assert_eq!(gv_coeffs[i].len(), s);
                fft::interpolate_poly(&mut gv_coeffs[i], &inv_twiddles);
            }

            let mut gi_coeffs: Vec<Vec<B>> = evals_gi.clone();
            for i in 0..n {
                debug_assert_eq!(gi_coeffs[i].len(), s);
                fft::interpolate_poly(&mut gi_coeffs[i], &inv_twiddles);
            }

            // evaluate gi_coeffs and gv_coeffs at eps
            let mut gv_evals: Vec<B> = Vec::new();
            for i in 0..l {
                gv_evals.push(eval(&gv_coeffs[i], eps));
            }

            let mut gi_evals: Vec<B> = Vec::new();
            for i in 0..n {
                gi_evals.push(eval(&gi_coeffs[i], eps));
            }

            // check that the evaluations are consistent
            let should_be_gi_evals = pp.pack_from_public(&gv_evals);
            debug_assert_eq!(should_be_gi_evals, gi_evals);
        }

        (evals_gi, evals_gv)
    }
}
