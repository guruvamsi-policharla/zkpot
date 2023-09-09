// Cheking consistency of X and X^T (a weird transpose) as used by zkPoT

use crypto::{ElementHasher, RandomCoin};
use fri::{batch_proof::BatchOpening, BatchProver, FriOptions};
use math::{fft, StarkField};
use poly_args::sumcheck_prod::{scp_prove, scp_verify, SCPProof};
use secret_sharing::pss::PackedSharingParams;

use crate::{consistency::pss::prove_pss_consistency, sample::data::MPCData, MLParams};

use super::pss::{verify_pss_consistency, PSSConsistencyProof};

// Struct for Data PSS proofs
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataPSSProof<B: StarkField> {
    y_pss_proof: PSSConsistencyProof<B>,
    x_pss_proof: PSSConsistencyProof<B>,
    xt_pss_proof: PSSConsistencyProof<B>,
}

// Struct for Data Transpose proofs
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransposeProof<B: StarkField> {
    labels_rowuv: Vec<String>,
    labels_coluv: Vec<String>,
    row_proof: SCPProof<B>,
    col_proof: SCPProof<B>,
}

// todo: test with randomness
/// Proves the PSS correctness of data and shares
pub fn prove_datapss<B, H, R>(
    data: &MPCData<B>,
    batch_prover_nbyl: &mut BatchProver<B, B, H, R>,
    batch_prover_ndbyl: &mut BatchProver<B, B, H, R>,
    mlp: &MLParams,
    pp: &PackedSharingParams<B>,
) -> DataPSSProof<B>
where
    B: StarkField,
    H: ElementHasher<BaseField = B>,
    R: RandomCoin<BaseField = B, Hasher = H>,
{
    //////////////////////////////////////////////////////////////////////////
    // carry out pss check on y
    // do it for all batches at once
    // gather all secrets of y
    let mut y_gv: Vec<Vec<B>> = vec![vec![B::ZERO; mlp.data_size / pp.l]; pp.l];
    let mut y_gi: Vec<Vec<B>> = vec![vec![B::ZERO; mlp.data_size / pp.l]; pp.n];

    for i in 0..mlp.batches() {
        for j in 0..mlp.batch_size / pp.l {
            for k in 0..pp.l {
                y_gv[k][j + i * mlp.batch_size / pp.l] = data.y[i][j][k];
            }
        }
    }

    // gather all shares of y
    for i in 0..mlp.batches() {
        for j in 0..mlp.batch_size / pp.l {
            for k in 0..pp.n {
                y_gi[k][j + i * mlp.batch_size / pp.l] = data.ypack[k][i][j];
            }
        }
    }

    let id = "y_pss";

    // create labels for the polynomials
    let mut y_gi_labels = Vec::new();
    for i in 0..pp.n {
        y_gi_labels.push(format!("{}_gi_{}", id, i));
    }

    let mut y_gv_labels = Vec::new();
    for i in 0..pp.l {
        y_gv_labels.push(format!("{}_gv_{}", id, i));
    }

    // add the polynomials to the batch_prover_dbyl
    for i in 0..pp.n {
        batch_prover_nbyl.insert(y_gi_labels[i].clone(), y_gi[i].clone());
    }

    for i in 0..pp.l {
        batch_prover_nbyl.insert(y_gv_labels[i].clone(), y_gv[i].clone());
    }

    let y_pss_proof = prove_pss_consistency(
        &y_gi,
        &y_gv,
        &y_gi_labels,
        &y_gv_labels,
        batch_prover_nbyl,
        pp.n,
        id,
    );

    ////////////////////////////////////////////////////////////////////////////////////////
    // carry out pss checks for x
    // gather all secrets of x
    let mut x_gv: Vec<Vec<B>> = vec![vec![B::ZERO; mlp.data_size * mlp.dim / pp.l]; pp.l];
    let mut x_gi: Vec<Vec<B>> = vec![vec![B::ZERO; mlp.data_size * mlp.dim / pp.l]; pp.n];

    for i in 0..mlp.batches() {
        for j in 0..mlp.batch_size {
            for k in 0..mlp.dim / pp.l {
                for ii in 0..pp.l {
                    x_gv[ii][k + i * mlp.batch_size * (mlp.dim / pp.l) + j * (mlp.dim / pp.l)] =
                        data.x[i][j][k][ii];
                }
            }
        }
    }

    // gather all shares of x
    for i in 0..mlp.batches() {
        for j in 0..mlp.batch_size {
            for k in 0..mlp.dim / pp.l {
                for ii in 0..pp.n {
                    x_gi[ii][k + i * mlp.batch_size * (mlp.dim / pp.l) + j * (mlp.dim / pp.l)] =
                        data.xpack[ii][i][k][j];
                }
            }
        }
    }

    let id = "x_pss";
    // create labels of the polynomials
    let mut x_gi_labels = Vec::new();
    for i in 0..pp.n {
        x_gi_labels.push(format!("{}_gi_{}", id, i));
    }

    let mut x_gv_labels = Vec::new();
    for i in 0..pp.l {
        x_gv_labels.push(format!("{}_gv_{}", id, i));
    }

    // add the polynomials to the batch_prover_ndbyl
    for i in 0..pp.n {
        batch_prover_ndbyl.insert(x_gi_labels[i].clone(), x_gi[i].clone());
    }

    for i in 0..pp.l {
        batch_prover_ndbyl.insert(x_gv_labels[i].clone(), x_gv[i].clone());
    }

    let x_pss_proof = prove_pss_consistency(
        &x_gi,
        &x_gv,
        &x_gi_labels,
        &x_gv_labels,
        batch_prover_ndbyl,
        pp.n,
        id,
    );

    ////////////////////////////////////////////////////////////////////////////
    // carry out pss checks for xt
    // gather all secrets of xt
    let mut xt_gv: Vec<Vec<B>> = vec![vec![B::ZERO; mlp.data_size * mlp.dim / pp.l]; pp.l];
    let mut xt_gi: Vec<Vec<B>> = vec![vec![B::ZERO; mlp.data_size * mlp.dim / pp.l]; pp.n];

    for i in 0..mlp.batches() {
        for j in 0..mlp.dim {
            for k in 0..mlp.batch_size / pp.l {
                for ii in 0..pp.l {
                    xt_gv[ii]
                        [k + i * mlp.dim * (mlp.batch_size / pp.l) + j * (mlp.batch_size / pp.l)] =
                        data.xt[i][j][k][ii];
                }
            }
        }
    }

    // gather all shares of xt
    for i in 0..mlp.batches() {
        for j in 0..mlp.dim {
            for k in 0..mlp.batch_size / pp.l {
                for ii in 0..pp.n {
                    xt_gi[ii]
                        [k + i * mlp.dim * (mlp.batch_size / pp.l) + j * (mlp.batch_size / pp.l)] =
                        data.xtpack[ii][i][k][j];
                }
            }
        }
    }

    let id = "xt_pss";
    // create labels of the polynomials
    let mut xt_gi_labels = Vec::new();
    for i in 0..pp.n {
        xt_gi_labels.push(format!("{}_gi_{}", id, i));
    }

    let mut xt_gv_labels = Vec::new();
    for i in 0..pp.l {
        xt_gv_labels.push(format!("{}_gv_{}", id, i));
    }

    // add the polynomials to the batch_prover_ndbyl
    for i in 0..pp.n {
        batch_prover_ndbyl.insert(xt_gi_labels[i].clone(), xt_gi[i].clone());
    }

    for i in 0..pp.l {
        batch_prover_ndbyl.insert(xt_gv_labels[i].clone(), xt_gv[i].clone());
    }

    let xt_pss_proof = prove_pss_consistency(
        &xt_gi,
        &xt_gv,
        &xt_gi_labels,
        &xt_gv_labels,
        batch_prover_ndbyl,
        pp.n,
        id,
    );

    DataPSSProof {
        y_pss_proof,
        x_pss_proof,
        xt_pss_proof,
    }
}

impl<B> DataPSSProof<B>
where
    B: StarkField,
{
    pub fn verify<H, R>(
        &self,
        batch_opening_nbyl: &BatchOpening<B, B, H, R>,
        batch_opening_ndbyl: &BatchOpening<B, B, H, R>,
        mlp: &MLParams,
        pp: &PackedSharingParams<B>,
        options: &FriOptions,
    ) where
        H: ElementHasher<BaseField = B>,
        R: RandomCoin<BaseField = B, Hasher = H>,
    {
        batch_opening_nbyl.verify(mlp.data_size / pp.l - 1, options.clone());

        verify_pss_consistency(
            &self.y_pss_proof,
            &batch_opening_nbyl,
            pp.n,
            mlp.data_size / pp.l,
        );

        batch_opening_ndbyl.verify(mlp.data_size * mlp.dim / pp.l - 1, options.clone());

        verify_pss_consistency(
            &self.x_pss_proof,
            &batch_opening_ndbyl,
            pp.n,
            mlp.data_size * mlp.dim / pp.l,
        );

        verify_pss_consistency(
            &self.xt_pss_proof,
            &batch_opening_ndbyl,
            pp.n,
            mlp.data_size * mlp.dim / pp.l,
        );
    }
}

/// Proves the transpose consistency of data and shares
/// transpose check
/// see test case for generation procedure.
/// todo: handle zero knowledge. this is when the polynomials are masked
pub fn prove_data_consistency<B, H, R>(
    data: &MPCData<B>,
    mlp: &MLParams,
    batch_prover_n: &mut BatchProver<B, B, H, R>,
    batch_prover_ndbyb: &mut BatchProver<B, B, H, R>,
    pp: &PackedSharingParams<B>,
) -> TransposeProof<B>
where
    B: StarkField,
    H: ElementHasher<BaseField = B>,
    R: RandomCoin<BaseField = B, Hasher = H>,
{
    // compute rouv and coluv
    let mut evals_rowuv: Vec<Vec<B>> = vec![vec![B::ZERO; mlp.data_size]; mlp.dim];

    for i in 0..mlp.batches() {
        for j in 0..mlp.batch_size {
            for k in 0..mlp.dim / pp.l {
                for ii in 0..pp.l {
                    evals_rowuv[ii + k * pp.l][j + i * mlp.batch_size] = data.x[i][j][k][ii];
                }
            }
        }
    }
    debug_assert_eq!(evals_rowuv[0].len(), mlp.data_size);

    let mut evals_coluv: Vec<Vec<B>> =
        vec![vec![B::ZERO; mlp.data_size * mlp.dim / mlp.batch_size]; mlp.batch_size];

    for i in 0..mlp.batches() {
        for j in 0..mlp.dim {
            for k in 0..mlp.batch_size / pp.l {
                for ii in 0..pp.l {
                    evals_coluv[ii + k * pp.l][j + i * mlp.dim] = data.xt[i][j][k][ii];
                }
            }
        }
    }
    debug_assert_eq!(
        evals_coluv[0].len(),
        mlp.data_size * mlp.dim / mlp.batch_size
    );

    // provide sumcheck proof on rowuv and coluv
    let id = "rowuv";

    // todo: change these to mlp parameters
    // interpolate the polynomials
    let row_twiddles = fft::get_inv_twiddles(mlp.data_size);
    let col_twiddles = fft::get_inv_twiddles(mlp.data_size * mlp.dim / mlp.batch_size);

    let mut coeffs_rowuv = evals_rowuv.clone();
    let mut coeffs_coluv = evals_coluv.clone();

    coeffs_rowuv
        .iter_mut()
        .for_each(|evals| fft::interpolate_poly(evals, &row_twiddles));
    coeffs_coluv
        .iter_mut()
        .for_each(|evals| fft::interpolate_poly(evals, &col_twiddles));

    // add p polynomials in the batch prover
    let mut labels_rowuv = Vec::new();
    for i in 0..mlp.dim {
        labels_rowuv.push(format!("{}_sc_p_{}", id, i));
        batch_prover_n.insert(labels_rowuv[i].clone(), coeffs_rowuv[i].clone());
    }
    let row_proof = scp_prove(
        &evals_rowuv,
        &labels_rowuv,
        &vec![B::ONE; mlp.data_size],
        &vec![B::ONE; mlp.dim],
        batch_prover_n,
        id.to_owned(),
    );

    let id = "coluv";
    let mut labels_coluv = Vec::new();
    for i in 0..mlp.batch_size {
        labels_coluv.push(format!("{}_sc_p_{}", id, i));
        batch_prover_ndbyb.insert(labels_coluv[i].clone(), coeffs_coluv[i].clone());
    }

    let col_proof = scp_prove(
        &evals_coluv,
        &labels_coluv,
        &vec![B::ONE; mlp.data_size * mlp.dim / mlp.batch_size],
        &vec![B::ONE; mlp.batch_size],
        batch_prover_ndbyb,
        id.to_owned(),
    );

    TransposeProof {
        labels_rowuv,
        labels_coluv,
        row_proof,
        col_proof,
    }
}

impl<B> TransposeProof<B>
where
    B: StarkField,
{
    pub fn verify<H, R>(
        &self,
        batch_opening_n: &BatchOpening<B, B, H, R>,
        batch_opening_ndbyb: &BatchOpening<B, B, H, R>,
        mlp: &MLParams,
    ) where
        H: ElementHasher<BaseField = B>,
        R: RandomCoin<BaseField = B, Hasher = H>,
    {
        scp_verify(
            &self.row_proof,
            mlp.data_size - 1,
            &vec![B::ONE; mlp.data_size],
            &vec![B::ONE; mlp.dim],
            &batch_opening_n,
            mlp.dim,
        );

        scp_verify(
            &self.col_proof,
            mlp.data_size * mlp.dim / mlp.batch_size - 1,
            &vec![B::ONE; mlp.data_size * mlp.dim / mlp.batch_size],
            &vec![B::ONE; mlp.batch_size],
            &batch_opening_ndbyb,
            mlp.batch_size,
        );

        assert_eq!(self.row_proof.alpha, self.col_proof.alpha);
    }
}

#[cfg(test)]
mod tests {
    use ark_std::{end_timer, start_timer};
    use crypto::{hashers::Blake3_256, DefaultRandomCoin};
    use fri::{BatchProver, DefaultProverChannel, FriOptions};
    use math::fields::f128::BaseElement;

    use secret_sharing::pss::PackedSharingParams;

    use crate::{
        sample::data::sample_data_weights, MLParams, FOLDING_FACTOR, FRI_QUERIES, LDE_BLOWUP,
        MAX_REMAINDER_DEGREE, N_PARTIES,
    };

    use super::{prove_data_consistency, prove_datapss};
    use human_bytes::human_bytes;

    type Blake3 = Blake3_256<BaseElement>;

    #[test]
    pub fn bench_data() {
        for data_size in [12, 14, 16, 18] {
            for dim in [128, 256, 512, 1024] {
                println!("======================================================");
                let mlp = MLParams {
                    batch_size: 256,
                    data_size: 1 << data_size,
                    dim,
                    precision: 16,
                    fixp_size: 64,
                };

                println!("mlp: {:?}", mlp);

                let pp = PackedSharingParams::<BaseElement>::new(N_PARTIES / 8);

                let options = FriOptions::new(LDE_BLOWUP, FOLDING_FACTOR, MAX_REMAINDER_DEGREE);
                let mut batch_prover_nbyl =
                    BatchProver::<
                        BaseElement,
                        BaseElement,
                        Blake3,
                        DefaultRandomCoin<Blake3_256<BaseElement>>,
                    >::new(options.clone(), mlp.data_size / pp.l - 1);

                let mut batch_prover_ndbyl =
                    BatchProver::<
                        BaseElement,
                        BaseElement,
                        Blake3,
                        DefaultRandomCoin<Blake3_256<BaseElement>>,
                    >::new(options.clone(), mlp.data_size * mlp.dim / pp.l - 1);

                let mut batch_prover_n = BatchProver::<
                    BaseElement,
                    BaseElement,
                    Blake3,
                    DefaultRandomCoin<Blake3_256<BaseElement>>,
                >::new(options.clone(), mlp.data_size - 1);

                let mut batch_prover_ndbyb = BatchProver::<
                    BaseElement,
                    BaseElement,
                    Blake3,
                    DefaultRandomCoin<Blake3_256<BaseElement>>,
                >::new(
                    options.clone(),
                    mlp.data_size * mlp.dim / mlp.batch_size - 1,
                );

                let data_timer = start_timer!(|| "Data generation");
                let (data, _w) = sample_data_weights(&mlp, &pp);
                end_timer!(data_timer);

                let datapss_timer = start_timer!(|| "Data PSS");
                let datapss_proof = prove_datapss(
                    &data,
                    &mut batch_prover_nbyl,
                    &mut batch_prover_ndbyl,
                    &mlp,
                    &pp,
                );
                end_timer!(datapss_timer);

                let transpose_timer = start_timer!(|| "Data Transpose");
                let transpose_proof = prove_data_consistency(
                    &data,
                    &mlp,
                    &mut batch_prover_n,
                    &mut batch_prover_ndbyb,
                    &pp,
                );
                end_timer!(transpose_timer);

                let fri_timer = start_timer!(|| "FRI Opening");
                let open_pos = BaseElement::from(8u32);
                let mut channel = DefaultProverChannel::new(
                    (mlp.data_size / pp.l) * options.blowup_factor(),
                    FRI_QUERIES,
                );

                let batch_opening_nbyl =
                    batch_prover_nbyl.batch_prove(&mut channel, open_pos, mlp.data_size / pp.l - 1);

                let mut channel = DefaultProverChannel::new(
                    (mlp.data_size * mlp.dim / pp.l) * options.blowup_factor(),
                    FRI_QUERIES,
                );

                let batch_opening_ndbyl = batch_prover_ndbyl.batch_prove(
                    &mut channel,
                    open_pos,
                    mlp.data_size * mlp.dim / pp.l - 1,
                );

                let mut channel =
                    DefaultProverChannel::new(mlp.data_size * options.blowup_factor(), FRI_QUERIES);

                let batch_opening_n =
                    batch_prover_n.batch_prove(&mut channel, open_pos, mlp.data_size - 1);

                let mut channel = DefaultProverChannel::new(
                    (mlp.data_size * mlp.dim / mlp.batch_size) * options.blowup_factor(),
                    FRI_QUERIES,
                );

                let batch_opening_ndbyb = batch_prover_ndbyb.batch_prove(
                    &mut channel,
                    open_pos,
                    mlp.data_size * mlp.dim / mlp.batch_size - 1,
                );
                end_timer!(fri_timer);

                // print size of all batch opening proofs
                println!(
                    "Batch opening nbyl size: {}",
                    human_bytes(batch_opening_nbyl.size() as f64)
                );
                println!(
                    "Batch opening ndbyl size: {}",
                    human_bytes(batch_opening_ndbyl.size() as f64)
                );
                println!(
                    "Batch opening n size: {}",
                    human_bytes(batch_opening_n.size() as f64)
                );
                println!(
                    "Batch opening ndbyb size: {}",
                    human_bytes(batch_opening_ndbyb.size() as f64)
                );

                // print total size
                println!(
                    "Total size: {}",
                    human_bytes(
                        (batch_opening_nbyl.size()
                            + batch_opening_ndbyl.size()
                            + batch_opening_n.size()
                            + batch_opening_ndbyb.size()) as f64
                    )
                );

                let verify_timer = start_timer!(|| "Verify");
                batch_opening_nbyl.verify(mlp.data_size / pp.l - 1, options.clone());
                batch_opening_ndbyl.verify(mlp.data_size * mlp.dim / pp.l - 1, options.clone());
                batch_opening_n.verify(mlp.data_size - 1, options.clone());
                batch_opening_ndbyb.verify(
                    mlp.data_size * mlp.dim / mlp.batch_size - 1,
                    options.clone(),
                );

                datapss_proof.verify(
                    &batch_opening_nbyl,
                    &batch_opening_ndbyl,
                    &mlp,
                    &pp,
                    &options,
                );

                transpose_proof.verify(&batch_opening_n, &batch_opening_ndbyb, &mlp);
                end_timer!(verify_timer);
            }
        }
    }
}
