use core::marker::PhantomData;
use crypto::{ElementHasher, MerkleTree, RandomCoin};
use math::{
    get_power_series, log2,
    polynom::{eval, remove_leading_zeros},
    FieldElement, StarkField,
};
use std::collections::BTreeMap;

use crate::{
    batch_proof::BatchOpening,
    utils::{get_quotient, hash_vec_values, low_degree_evaluation, transpose},
    DefaultProverChannel, FriOptions, FriProver,
};

use ark_std::{end_timer, start_timer};

#[derive(Clone)]
pub struct BatchProver<B, E, H, R>
where
    B: StarkField,
    E: FieldElement<BaseField = B>,
    H: ElementHasher<BaseField = B>,
    R: RandomCoin<BaseField = E::BaseField, Hasher = H>,
{
    options: FriOptions,
    polynomials: BTreeMap<String, Vec<B>>, // In coefficient form.
    // Can create a BatchProver and pass it to all functions and append to polynomials
    // Then create a final proof
    degree: usize,
    _public_coin: R,
    _extension: PhantomData<E>,
    _hasher: PhantomData<H>,
}

impl<B, E, H, R> BatchProver<B, E, H, R>
where
    B: StarkField,
    E: FieldElement<BaseField = B> + math::ExtensionOf<B>,
    H: ElementHasher<BaseField = B>,
    R: RandomCoin<BaseField = E::BaseField, Hasher = H>,
{
    // CONSTRUCTOR
    // --------------------------------------------------------------------------------------------
    #[allow(unused)]
    pub fn new(options: FriOptions, degree: usize) -> Self {
        BatchProver {
            options,
            polynomials: BTreeMap::new(),
            degree: degree,
            _public_coin: R::new(&[]),
            _extension: PhantomData,
            _hasher: PhantomData,
        }
    }

    // INSERT POLYNOMIALS
    // --------------------------------------------------------------------------------------------
    // Insert polynomials in the BTreeMap
    // Remove leadings zeros and check that degree of all polynomials is the same
    pub fn insert(&mut self, key: String, mut poly: Vec<B>) {
        // check that poly_degree is correct
        let sanitized_poly = remove_leading_zeros(&poly);
        assert!(sanitized_poly.len() <= self.degree + 1);

        poly.resize(self.degree + 1, B::ZERO);

        self.polynomials.try_insert(key, poly).unwrap();
    }

    // COMMIT POLYNOMIALS
    // --------------------------------------------------------------------------------------------
    // Create a Merkle tree from the provided polynomials and commit its root to the channel.
    // provide a FRI proof for a folded polynomial and a batch merkle proof for the polynomials
    // todo: return a list of evaluations of all the polynomials which can be indexed by a label
    // todo: return the proof
    #[allow(unused)]
    pub fn batch_prove(
        &mut self,
        channel: &mut DefaultProverChannel<E, H, R>,
        open_pos: E,
        max_degree: usize,
    ) -> BatchOpening<B, E, H, R> {
        println!(
            "Creating batch opening of {} polynomials of degree {}",
            self.polynomials.len(),
            self.degree
        );
        // Create LDEs of the polynomials
        let domain_size = self.options.blowup_factor() * (max_degree + 1);

        let omega = B::get_root_of_unity(log2(domain_size));
        let domain = get_power_series(omega, domain_size);
        let domain = domain.iter().map(|x| E::from(*x)).collect::<Vec<_>>();

        let lde_timer = start_timer!(|| "lde");
        let mut lde_polynomials = self.polynomials.values().cloned().collect::<Vec<_>>();

        for i in 0..lde_polynomials.len() {
            // if i % 50 == 0 {
            //     println!("LDE of polynomial {}", i);
            // }
            low_degree_evaluation(&mut lde_polynomials[i], domain_size);
        }
        end_timer!(lde_timer);

        // transpose the lde_polynomials, hash_values and commit via merkle tree
        let transpose_timer = start_timer!(|| "transpose");
        let lde_polynomials = transpose(lde_polynomials);
        end_timer!(transpose_timer);

        // lde_polynomials if a Vec<Vec<B>> where each inner vector is a list of evaluations of
        // all the polynomials at the corresponding root of unity
        let merkle_leaves = hash_vec_values::<H, B>(&lde_polynomials);

        // build a merkle tree for the merkle_leaves
        let poly_mt: MerkleTree<H> = MerkleTree::new(merkle_leaves).unwrap();

        // fiat-shamir
        let mut public_coin: R = RandomCoin::new(&[]);
        public_coin.reseed(poly_mt.root().clone());
        let alpha: E = public_coin.draw().expect("fiat-shamir failed");

        let folding_timer = start_timer!(|| "folding");
        // Compute the folded polynomial (coeffs and lde) using alpha as f = f0 + alpha*f1 + alpha^2*f2 + ...
        let mut folded_poly_lde = vec![E::ZERO; domain_size];
        let mut alpha_pow = E::ONE;
        let alpha_powers = get_power_series(alpha, self.polynomials.len());
        for i in 0..domain_size {
            for j in 0..self.polynomials.len() {
                folded_poly_lde[i] += E::from(lde_polynomials[i][j]) * alpha_powers[j];
            }
        }

        let mut folded_poly = vec![E::ZERO; max_degree + 1];
        let mut alpha_pow = E::ONE;
        for (_, poly) in self.polynomials.iter() {
            for i in 0..(max_degree + 1) {
                folded_poly[i] += E::from(poly[i]) * alpha_pow;
            }
            alpha_pow *= alpha;
        }
        end_timer!(folding_timer);

        let opening_timer = start_timer!(|| "opening");
        // Compute evaluations of all the polynomials at x
        let mut poly_openings: BTreeMap<String, E> = BTreeMap::new();
        for (label, poly) in self.polynomials.iter() {
            poly_openings.insert(label.clone(), eval(&poly, open_pos));
        }

        let folded_opening = eval(&folded_poly, open_pos);
        end_timer!(opening_timer);

        let quotient_timer = start_timer!(|| "quotient");
        // Commit the folded quotient polynomial xq
        let xq_lde = get_quotient(&folded_poly_lde, open_pos, folded_opening, &domain);
        end_timer!(quotient_timer);

        let layers_timer = start_timer!(|| "layers and commitments");
        // instantiate the prover and generate the proof
        let mut prover = FriProver::new(self.options.clone());
        prover.build_layers(channel, xq_lde.clone());
        let positions = channel.draw_query_positions();
        let proof = prover.build_proof(&positions);

        let commitments = channel.layer_commitments().to_vec();
        let xq_queries = positions.iter().map(|&p| xq_lde[p]).collect::<Vec<_>>();
        end_timer!(layers_timer);

        #[cfg(debug_assertions)]
        {
            use crate::{DefaultVerifierChannel, FriVerifier};
            use crypto::DefaultRandomCoin;
            // check the FRI proof
            // verify the proof
            let mut channel = DefaultVerifierChannel::<E, H>::new(
                proof.clone(),
                commitments.clone(),
                domain_size,
                self.options.folding_factor(),
            )
            .unwrap();

            let mut coin = DefaultRandomCoin::<H>::new(&[]);
            let verifier: FriVerifier<E, DefaultVerifierChannel<E, H>, H, DefaultRandomCoin<H>> =
                FriVerifier::new(&mut channel, &mut coin, self.options.clone(), max_degree)
                    .unwrap();
            verifier
                .verify(&mut channel, &xq_queries, &positions)
                .unwrap();
        }

        // Additionally, for all openings generated in the first layer (contained in positions),
        // give batch merkle openings to poly_mt, and the verifier checks that the folding was done correctly
        let poly_mt_open = poly_mt.prove_batch(&positions).unwrap();

        // gather the values from lde_polynomials and folded_poly_lde
        let poly_queries = positions
            .iter()
            .map(|&i| lde_polynomials[i].clone())
            .collect::<Vec<_>>();

        // to prove a polynomial opening we start use a low degree test on the quotient polynomial
        // since q(x) = (f(x) - y*)/(x-x*), proving that x.q(x) has
        // 1. low degree
        // 2. the "folding" was done correctly, i.e q(w) = (f(w) - y*)/(w-x*), at all points opened in the FRI proof
        // is sufficient to convince a verifier that f(x*) = y*

        let batch_opening = BatchOpening::new(
            commitments,
            poly_mt.root().clone(),
            proof.clone(),
            positions.clone(),
            poly_mt_open,
            poly_queries.clone(),
            xq_queries.clone(),
            open_pos,
            folded_opening,
            poly_openings,
        );

        batch_opening
    }
}

#[cfg(test)]
mod tests {
    use crypto::{hashers::Blake3_256, DefaultRandomCoin};
    use math::fields::f128::BaseElement;
    use rand_utils::rand_vector;

    use crate::prover::channel;

    use super::*;

    type B = BaseElement;
    type E = BaseElement;
    type H = Blake3_256<BaseElement>;
    type R = DefaultRandomCoin<Blake3_256<BaseElement>>;

    #[test]
    fn test_batch_prover() {
        let max_degree = (1 << 8) - 1;
        let num_poly = 1000;

        // create a batch prover
        let options = FriOptions::new(4, 4, 4);

        // create a prover
        let mut prover: BatchProver<B, E, H, R> = BatchProver::new(options.clone(), max_degree);

        // create a set of polynomials in coefficient form and insert them in the prover
        for i in 0..num_poly {
            prover
                .polynomials
                .insert(i.to_string(), rand_vector(max_degree + 1));
        }

        let mut channel = channel::DefaultProverChannel::<E, H, R>::new(
            (max_degree + 1) * options.blowup_factor(),
            32,
        );

        let batch_opening = prover.batch_prove(&mut channel, E::from(19u32), max_degree);

        batch_opening.verify(max_degree, options.clone());
        batch_opening.vsize();
    }
}
