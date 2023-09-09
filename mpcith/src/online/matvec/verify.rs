use crypto::{ElementHasher, RandomCoin};
use fri::batch_proof::BatchOpening;
use math::{FieldElement, StarkField};
use poly_args::sumcheck_prod::scp_verify;
use secret_sharing::pss::PackedSharingParams;

use super::MatVecProof;
use crate::{online::epoch::EpochProof, MLParams};

pub fn mat_vec_verify<B, E, H, R>(
    proof: &MatVecProof<B>,
    epoch_proofs: &Vec<EpochProof<B>>,
    batch_opening_n_: &BatchOpening<B, B, H, R>,
    batch_opening_ndb: &BatchOpening<B, B, H, R>,
    mlp: &MLParams,
    pp: &PackedSharingParams<B>,
) where
    B: StarkField,
    E: FieldElement<BaseField = B> + math::ExtensionOf<B>,
    H: ElementHasher<BaseField = B>,
    R: RandomCoin<BaseField = B, Hasher = H>,
{
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

    // compute z_sum
    let mut z_sum: Vec<B> = vec![B::ZERO; pp.l];
    for i in 0..pp.l {
        for j in 0..mlp.batches() {
            for u in 0..mlp.batch_size / pp.l {
                z_sum[i] += z[i][u][j];
            }
        }
    }

    // verify the prover's claim of xt_sum
    for i in 0..pp.l {
        scp_verify(
            &proof.xtsumproof[i],
            mlp.dim * mlp.batches() - 1,
            &w[i],
            &vec![B::ONE; mlp.batch_size / pp.l],
            batch_opening_ndb,
            mlp.batch_size / pp.l,
        );
    }

    for i in 0..pp.l {
        assert_eq!(z_sum[i], proof.xtsumproof[i].alpha);
    }

    ///////////////////////////////////////////////////////////////////
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
            rpack.push(pp.pack_from_public(&vec![rd[k]; pp.l]))
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

    let mut s_sum: Vec<B> = vec![B::ZERO; pp.l];
    for i in 0..pp.l {
        for j in 0..mlp.batches() {
            for u in 0..mlp.dim / pp.l {
                s_sum[i] += s[i][u][j];
            }
        }
    }

    // verify the prover's claim of x_sum
    for i in 0..pp.l {
        scp_verify(
            &proof.xsumproof[i],
            mlp.data_size - 1,
            &r[i],
            &vec![B::ONE; mlp.dim / pp.l],
            batch_opening_n_,
            mlp.dim / pp.l,
        );
    }

    for i in 0..pp.l {
        assert_eq!(s_sum[i], proof.xsumproof[i].alpha);
    }
}
