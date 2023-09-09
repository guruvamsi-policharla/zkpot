use crypto::{ElementHasher, RandomCoin};
use fri::BatchProver;
use math::{fft, FieldElement, StarkField};
use poly_args::sumcheck_prod::{scp_prove, SCPProof};
use secret_sharing::pss::PackedSharingParams;

use crate::{online::epoch::EpochProof, sample::data::MPCData, MLParams};

use super::MatVecProof;

pub fn mat_vec_prove<B, E, H, R>(
    data: &MPCData<B>,
    epoch_proofs: &Vec<EpochProof<B>>,
    mlp: &MLParams,
    pp: &PackedSharingParams<B>,
    batch_prover_n: &mut BatchProver<B, E, H, R>, // degree N batch_prover
    batch_prover_ndb: &mut BatchProver<B, E, H, R>, // degree ND/B batch_prover
) -> MatVecProof<B>
where
    B: StarkField,
    E: FieldElement<BaseField = B> + math::ExtensionOf<B>,
    H: ElementHasher<BaseField = B>,
    R: RandomCoin<BaseField = B, Hasher = H>,
{
    // extract zpack, wpack, spack, and rpack from all the epoch proofs.
    let mut w: Vec<Vec<B>> = vec![vec![B::ZERO; mlp.dim * mlp.batches()]; pp.l];
    for i in 0..pp.l {
        for j in 0..mlp.batches() {
            for k in 0..mlp.dim {
                w[i][k + j * mlp.dim] = epoch_proofs[j].wpack[i][k];
            }
        }
    }

    let mut z: Vec<Vec<Vec<B>>> =
        vec![vec![vec![B::ZERO; mlp.batches()]; mlp.batch_size / pp.l]; pp.l];
    for j in 0..mlp.batches() {
        for i in 0..pp.l {
            for u in 0..mlp.batch_size / pp.l {
                z[i][u][j] += epoch_proofs[j].zpack[u][i];
            }
        }
    }

    // contains B/l polynomials of degree ND/B-1 for each or the l parties being opened (should be l-1)
    let mut xtpack_enc: Vec<Vec<Vec<B>>> =
        vec![vec![vec![B::ZERO; mlp.dim * mlp.batches()]; mlp.batch_size / pp.l]; pp.l];
    for i in 0..pp.l {
        for j in 0..mlp.batches() {
            for u in 0..mlp.batch_size / pp.l {
                for k in 0..mlp.dim {
                    xtpack_enc[i][u][k + j * mlp.dim] = data.xtpack[i][j][u][k];
                }
            }
        }
    }

    // perform low degree extension on xtpack_enc and get the coefficients
    let inv_twiddles = fft::get_inv_twiddles::<B>(mlp.dim * mlp.batches());
    let mut xtsumproof: Vec<SCPProof<B>> = Vec::new();

    for i in 0..pp.l {
        let id = format!("xtpack_{}", i);

        // create labels and interpolate
        // add xtpack_coeff to batch prover
        let mut xtpack_labels: Vec<String> = Vec::new();
        let mut xtpack_coeff = xtpack_enc[i].clone();
        for u in 0..mlp.batch_size / pp.l {
            xtpack_labels.push(format!("xtpack_{}_{}", i, u)); //xtpack_{party}_{poly}
            fft::interpolate_poly(&mut xtpack_coeff[u], &inv_twiddles);
            batch_prover_ndb.insert(xtpack_labels[u].clone(), xtpack_coeff[u].clone());
        }

        // sum_check creation
        xtsumproof.push(scp_prove(
            &xtpack_enc[i],
            &xtpack_labels,
            &w[i],
            &vec![B::ONE; mlp.batch_size / pp.l],
            batch_prover_ndb,
            id,
        ));
    }

    #[cfg(debug_assertions)]
    {
        // sanity check on player 0
        let mut z_sum: Vec<B> = vec![B::ZERO; pp.l];
        for i in 0..pp.l {
            for j in 0..mlp.batches() {
                for u in 0..mlp.batch_size / pp.l {
                    z_sum[i] += z[i][u][j];
                }
            }
        }

        let mut xtw_sum = vec![B::ZERO; pp.l];
        for i in 0..pp.l {
            for j in 0..mlp.batches() {
                for u in 0..mlp.batch_size / pp.l {
                    for k in 0..mlp.dim {
                        xtw_sum[i] += xtpack_enc[i][u][k + j * mlp.dim] * w[i][k + j * mlp.dim];
                    }
                }
            }
        }

        // create a sumcheck proof for the above
        assert_eq!(z_sum, xtw_sum);
        // assert_eq!(xtw_sum[1], sumproof[1].alpha);

        for i in 0..pp.l {
            assert_eq!(z_sum[i], xtsumproof[i].alpha);
        }
    }

    ///////////////////////////////////////////////////////////////////
    // repeat above for xpack
    // contains D/l polynomials of degree N-1 for each or the l parties being opened (should be l-1)
    let mut xpack_enc: Vec<Vec<Vec<B>>> =
        vec![vec![vec![B::ZERO; mlp.data_size]; mlp.dim / pp.l]; pp.l];
    for i in 0..pp.l {
        for j in 0..mlp.batches() {
            for u in 0..mlp.dim / pp.l {
                for k in 0..mlp.batch_size {
                    xpack_enc[i][u][k + j * mlp.batch_size] = data.xpack[i][j][u][k];
                }
            }
        }
    }

    let pp = PackedSharingParams::<B>::new(pp.l);

    let mut r: Vec<Vec<B>> = vec![vec![B::ZERO; mlp.data_size]; pp.l];
    for j in 0..mlp.batches() {
        let mut rd = epoch_proofs[j].rd2shares.clone();

        for u in 0..mlp.batch_size / pp.l {
            pp.unpack2_in_place(&mut rd[u]);
        }

        let rd = rd.into_iter().flatten().collect::<Vec<B>>();

        // rpack is of size BxN_PARTIES
        let mut rpack = Vec::new();
        for k in 0..mlp.batch_size {
            rpack.push(pp.pack_from_public(&mut vec![rd[k]; pp.l]))
        }
        for i in 0..pp.l {
            for k in 0..mlp.batch_size {
                r[i][k + j * mlp.batch_size] = rpack[k][i];
            }
        }
    }

    let mut s: Vec<Vec<Vec<B>>> = vec![vec![vec![B::ZERO; mlp.batches()]; mlp.dim / pp.l]; pp.l];
    for j in 0..mlp.batches() {
        for i in 0..pp.l {
            for u in 0..mlp.dim / pp.l {
                s[i][u][j] += epoch_proofs[j].spack[u][i];
            }
        }
    }

    // perform low degree extension on xtpack_enc and get the coefficients
    let inv_twiddles = fft::get_inv_twiddles::<B>(mlp.data_size);
    let mut xsumproof: Vec<SCPProof<B>> = Vec::new();

    for i in 0..pp.l {
        let id = format!("xpack_{}", i);

        let mut xpack_labels: Vec<String> = Vec::new();
        let mut xpack_coeff = xpack_enc[i].clone();
        for u in 0..mlp.dim / pp.l {
            xpack_labels.push(format!("xpack_{}_{}", i, u)); //xpack_{party}_{poly}
            fft::interpolate_poly(&mut xpack_coeff[u], &inv_twiddles);
            batch_prover_n.insert(xpack_labels[u].clone(), xpack_coeff[u].clone());
        }

        // sum_check creation
        xsumproof.push(scp_prove(
            &xpack_enc[i],
            &xpack_labels,
            &r[i],
            &vec![B::ONE; mlp.dim / pp.l],
            batch_prover_n,
            id,
        ));
    }

    #[cfg(debug_assertions)]
    {
        // sanity check on player 0
        let mut s_sum: Vec<B> = vec![B::ZERO; pp.l];
        for i in 0..pp.l {
            for j in 0..mlp.batches() {
                for u in 0..mlp.dim / pp.l {
                    s_sum[i] += s[i][u][j];
                }
            }
        }

        let mut xr_sum = vec![B::ZERO; pp.l];
        for i in 0..pp.l {
            for j in 0..mlp.batches() {
                for u in 0..mlp.dim / pp.l {
                    for k in 0..mlp.batch_size {
                        xr_sum[i] +=
                            xpack_enc[i][u][k + j * mlp.batch_size] * r[i][k + j * mlp.batch_size];
                    }
                }
            }
        }

        // create a sumcheck proof for the above
        assert_eq!(s_sum, xr_sum);
    }

    MatVecProof {
        xtsumproof,
        xsumproof,
    }
}

#[cfg(test)]
mod tests {
    use ark_std::{end_timer, start_timer};
    use crypto::{hashers::Blake3_256, DefaultRandomCoin};
    use fri::{BatchProver, DefaultProverChannel, FriOptions};
    use human_bytes::human_bytes;
    use math::fields::f128::BaseElement;
    use secret_sharing::pss::PackedSharingParams;

    use crate::{
        online::{
            epoch::prove::prove_epoch,
            matvec::{prove::mat_vec_prove, verify::mat_vec_verify},
        },
        sample::data::sample_dummy,
        MLParams, FOLDING_FACTOR, FRI_QUERIES, LDE_BLOWUP, MAX_REMAINDER_DEGREE, N_PARTIES,
    };

    type Blake3 = Blake3_256<BaseElement>;

    #[test]
    fn mat_vec_kernel() {
        for data_size in [10, 12, 14, 16, 18] {
            for dim in [128, 256, 512, 1024] {
                println!("==============================================");

                let mlp = MLParams {
                    batch_size: 256,
                    data_size: 1 << data_size,
                    dim,
                    precision: 16,
                    fixp_size: 64,
                };
                println!("mlp:{:?}", mlp);
                let pp = PackedSharingParams::<BaseElement>::new(N_PARTIES / 8);
                let options = FriOptions::new(LDE_BLOWUP, FOLDING_FACTOR, MAX_REMAINDER_DEGREE);
                println!("LDE_BLOWUP:{}", LDE_BLOWUP);

                let datapack_section = start_timer!(|| "Data packing");
                let (data, mut w) = sample_dummy::<BaseElement>(&mlp, &pp);
                end_timer!(datapack_section);

                let mut batch_prover_n = BatchProver::<
                    BaseElement,
                    BaseElement,
                    Blake3,
                    DefaultRandomCoin<Blake3_256<BaseElement>>,
                >::new(options.clone(), mlp.data_size - 1);

                let mut batch_prover_ndb =
                    BatchProver::<
                        BaseElement,
                        BaseElement,
                        Blake3,
                        DefaultRandomCoin<Blake3_256<BaseElement>>,
                    >::new(options.clone(), mlp.dim * mlp.batches() - 1);

                let train_section = start_timer!(|| "Training");
                let mut epoch_proofs = Vec::new();
                for i in 0..1 {
                    epoch_proofs.push(prove_epoch(&data, &mut w, i, &mlp, &pp));
                }
                for _ in 1..mlp.batches() {
                    epoch_proofs.push(epoch_proofs[0].clone());
                }
                end_timer!(train_section);

                let matvec_prove = start_timer!(|| "MatVec Proof");
                let matvecproof = mat_vec_prove(
                    &data,
                    &epoch_proofs,
                    &mlp,
                    &pp,
                    &mut batch_prover_n,
                    &mut batch_prover_ndb,
                );
                end_timer!(matvec_prove);

                let fri_proof = start_timer!(|| "FRI Proof");
                let open_pos = BaseElement::from(8u32);
                let mut channel_n =
                    DefaultProverChannel::new(mlp.data_size * options.blowup_factor(), FRI_QUERIES);
                let batch_opening_n =
                    batch_prover_n.batch_prove(&mut channel_n, open_pos, mlp.data_size - 1);

                let mut channel_ndb = DefaultProverChannel::new(
                    mlp.dim * mlp.batches() * options.blowup_factor(),
                    FRI_QUERIES,
                );
                let batch_opening_ndb = batch_prover_ndb.batch_prove(
                    &mut channel_ndb,
                    open_pos,
                    mlp.dim * mlp.batches() - 1,
                );
                end_timer!(fri_proof);

                println!(
                    "BatchOpening size: {}",
                    human_bytes((batch_opening_n.size() + batch_opening_ndb.size()) as f64)
                );

                let matvec_verify = start_timer!(|| "MatVec Verify");
                batch_opening_n.verify(mlp.data_size - 1, options.clone());
                batch_opening_ndb.verify(mlp.dim * mlp.batches() - 1, options.clone());
                mat_vec_verify::<
                    BaseElement,
                    BaseElement,
                    Blake3_256<BaseElement>,
                    DefaultRandomCoin<Blake3_256<BaseElement>>,
                >(
                    &matvecproof,
                    &epoch_proofs,
                    &batch_opening_n,
                    &batch_opening_ndb,
                    &mlp,
                    &pp,
                );
                end_timer!(matvec_verify);
            }
        }
    }
}
