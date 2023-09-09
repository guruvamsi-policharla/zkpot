use math::StarkField;
use secret_sharing::pss::PackedSharingParams;

use crate::{
    online::utils::field_mod,
    sample::data::{MPCData, MPCWeights},
    MLParams,
};

use crate::online::epoch::EpochProof;

/// Takes as input training data, batch number (epoch) and updates the weights
/// todo: give as input a channel that the prover write to.
/// This is then given to the verifier who checks the transcript
pub fn prove_epoch<B>(
    data: &MPCData<B>,
    w: &mut MPCWeights<B>,
    epoch: usize,
    mlp: &MLParams,
    pp: &PackedSharingParams<B>,
) -> EpochProof<B>
where
    B: StarkField,
{
    // Data processing
    // extract out the epoch and half the parties
    let mut epoch_ypack: Vec<Vec<B>> = vec![vec![B::ZERO; mlp.batch_size / pp.l]; pp.l];
    for i in 0..pp.l {
        for u in 0..mlp.batch_size / pp.l {
            epoch_ypack[i][u] = data.ypack[i][epoch][u];
        }
    }

    // STEP I ================================================
    // 1. Local Computation
    // Multiplying weights by data in a distributed way
    // todo: add randomness
    let mut zpack: Vec<Vec<B>> = vec![vec![B::ZERO; pp.n]; mlp.batch_size / pp.l];
    for u in 0..mlp.batch_size / pp.l {
        for i in 0..pp.n {
            for k in 0..mlp.dim {
                zpack[u][i] += data.xtpack[i][epoch][u][k] * w.wpack[i][k];
            }
        }
    }

    // 2. Degree Reduction
    // All parties send their shares to king who reconstructs z
    // todo: avoid fft and directly use secrets
    let mut zshares: Vec<Vec<B>> = zpack.clone();
    for u in 0..mlp.batch_size / pp.l {
        pp.unpack2_in_place(&mut zshares[u]);
        pp.pack_from_public_in_place(&mut zshares[u]);
    }

    // 3. Truncation
    // mask the shares of z with randomness in order to do truncation
    // broadcast to all parties who engage in reconstruction and truncation
    // reconstruct a. compute a mod 2^m and share locally
    let ashares = zshares.clone(); //subtract randomness and add masks here

    let mut asharesmod = ashares.clone();
    let m: u128 = 1 << mlp.precision;
    for u in 0..mlp.batch_size / pp.l {
        pp.unpack_in_place(&mut asharesmod[u]);
        for j in 0..pp.l {
            asharesmod[u][j] = field_mod(asharesmod[u][j], m);
        }
        pp.pack_from_public_in_place(&mut asharesmod[u]);
    }

    let minv = B::ONE / B::from(m);
    // compute bshares
    let mut bshares = zshares.clone();
    for u in 0..mlp.batch_size / pp.l {
        for i in 0..pp.n {
            bshares[u][i] -= asharesmod[u][i];
            bshares[u][i] *= minv; //multiply by 2^-m
        }
    }

    // linear regression
    // ROUND 4 ================================================
    let fix_size: u128 = 1 << mlp.fixp_size; //2^M

    let mut bshift1 = bshares.clone();
    let shift1 = B::from(fix_size) + B::ONE / B::from(2 as u32);
    for u in 0..mlp.batch_size / pp.l {
        for i in 0..pp.n {
            bshift1[u][i] += shift1;
        }
    }

    let mut bshift2 = bshares.clone();
    let shift2 = B::from(fix_size) - B::ONE / B::from(2 as u32);
    for u in 0..mlp.batch_size / pp.l {
        for i in 0..pp.n {
            bshift2[u][i] += shift2;
        }
    }

    // todo: add randomness and send bshares1 and bshares2 to the king(verifier)
    let bshares1 = bshift1.clone();
    let bshares2 = bshift2.clone();

    // todo: pick masks appropriately to make lower M bits 1

    let fix_sizem1: u128 = 1 << (mlp.fixp_size - 1); //2^{M-1}
    let inv_fix_sizem1 = B::ONE / B::from(fix_sizem1); //1/2^{M-1}

    let mut ind1shares = bshift1.clone();
    let mut ind2shares = bshift2.clone();

    let mut bshares1mod = bshares1.clone();
    let mut bshares2mod = bshares2.clone();
    // todo: add randomness to bshares1/2 here and send to verifier

    for u in 0..mlp.batch_size / pp.l {
        pp.unpack_in_place(&mut bshares1mod[u]);
        pp.unpack_in_place(&mut bshares2mod[u]);
        for i in 0..pp.l {
            bshares1mod[u][i] = field_mod(bshares1mod[u][i], fix_sizem1);
            bshares2mod[u][i] = field_mod(bshares2mod[u][i], fix_sizem1);
        }
        pp.pack_from_public_in_place(&mut bshares1mod[u]);
        pp.pack_from_public_in_place(&mut bshares2mod[u]);
    }

    for u in 0..mlp.batch_size / pp.l {
        for i in 0..pp.n {
            ind1shares[u][i] -= bshares1mod[u][i];
            ind2shares[u][i] -= bshares2mod[u][i];

            ind1shares[u][i] *= inv_fix_sizem1;
            ind2shares[u][i] *= inv_fix_sizem1;
        }
    }

    let mut pind1shares = vec![vec![B::ZERO; pp.n / 8]; mlp.batch_size / pp.l];
    let mut pind2shares = vec![vec![B::ZERO; pp.n / 8]; mlp.batch_size / pp.l];

    for u in 0..mlp.batch_size / pp.l {
        for i in 0..pp.n / 8 {
            pind1shares[u][i] = ind1shares[u][i];
            pind2shares[u][i] = ind2shares[u][i];
        }
    }

    // ROUND 5 ================================================
    let mut f2shares = ind1shares.clone();
    for u in 0..mlp.batch_size / pp.l {
        for j in 0..pp.n {
            f2shares[u][j] *= B::ONE - ind2shares[u][j];
            // todo: add randomness to f2shares
        }
    }

    let mut fshares = f2shares.clone();
    for u in 0..mlp.batch_size / pp.l {
        // degree reduction
        pp.unpack2_in_place(&mut fshares[u]);
        pp.pack_from_public_in_place(&mut fshares[u]);
    }

    let ind12shares = fshares.clone(); //sub randomness here
    let mut pind12shares = vec![vec![B::ZERO; pp.n / 8]; mlp.batch_size / pp.l];
    for u in 0..mlp.batch_size / pp.l {
        for i in 0..pp.n / 8 {
            pind12shares[u][i] = ind12shares[u][i];
        }
    }

    let twoinv = B::ONE / B::from(2 as u32);
    let mut ed2shares = ind12shares.clone();
    for u in 0..mlp.batch_size / pp.l {
        for i in 0..pp.n {
            ed2shares[u][i] *= bshares[u][i] + twoinv;
            ed2shares[u][i] += ind2shares[u][i];
        }
    }

    let mut edshares = ed2shares.clone();
    for u in 0..mlp.batch_size / pp.l {
        // degree reduction of ehsares
        pp.unpack2_in_place(&mut edshares[u]);
        pp.pack_from_public_in_place(&mut edshares[u]);
    }

    let eshares = edshares.clone(); //sub randomness here

    // ROUND 6 ================================================
    let mut rd2shares = eshares;
    for u in 0..mlp.batch_size / pp.l {
        for i in 0..pp.n {
            rd2shares[u][i] -= data.ypack[i][epoch][u]; //add mask here
        }
    }

    // todo: send rdshares to verifier
    // king reconstruct rd
    let mut rd = rd2shares.clone();
    for u in 0..mlp.batch_size / pp.l {
        pp.unpack2_in_place(&mut rd[u]);
    }

    let rd = rd.into_iter().flatten().collect::<Vec<B>>();

    // rpack is of size Bxpp.n
    let mut rpack = Vec::new();
    for k in 0..mlp.batch_size {
        rpack.push(pp.pack_from_public(&mut vec![rd[k]; pp.l]))
        // todo: subtract randomness
    }

    // Multiplying weights by data in a distributed way
    // todo: add randomness
    let mut spack: Vec<Vec<B>> = vec![vec![B::ZERO; pp.n]; mlp.dim / pp.l];

    for u in 0..mlp.dim / pp.l {
        for k in 0..mlp.batch_size {
            for i in 0..pp.n {
                spack[u][i] += data.xpack[i][epoch][u][k] * rpack[k][i];
            }
        }
    }

    // ROUND 7 ================================================
    let mut sshares: Vec<Vec<B>> = spack.clone();
    for u in 0..mlp.dim / pp.l {
        pp.unpack2_in_place(&mut sshares[u]);
        pp.pack_from_public_in_place(&mut sshares[u]);
    }

    // ROUND 8 ================================================
    // mask the shares of c with randomness in order to do truncation
    // broadcast to all parties who engage in reconstruction and truncation
    // reconstruct c. compute c mod 2^m and share locally
    let cshares = sshares.clone();

    let mut csharemod = cshares.clone();
    for u in 0..mlp.dim / pp.l {
        pp.unpack_in_place(&mut csharemod[u]);
        for j in 0..pp.l {
            csharemod[u][j] = field_mod(csharemod[u][j], m);
        }
        pp.pack_from_public_in_place(&mut csharemod[u]);
    }

    // compute dshares
    let mut dshares = sshares.clone();
    for u in 0..mlp.dim / pp.l {
        for i in 0..pp.n {
            dshares[u][i] -= csharemod[u][i];
            dshares[u][i] *= minv; //multiply by 2^-m
        }
    }

    // open up dshares
    let mut d = dshares;
    for u in 0..mlp.dim / pp.l {
        pp.unpack2_in_place(&mut d[u]);
    }

    let d = d.into_iter().flatten().collect::<Vec<B>>();

    // dpack is of size DxN_PARTIES
    let mut dpack = Vec::new();
    for k in 0..mlp.dim {
        dpack.push(pp.pack_from_public(&mut vec![d[k]; pp.l]));
        // todo: subtract randomness
    }

    let wpack_old = w.wpack[0..pp.n / 8].to_vec();

    // LOCAL COMPUTATION -- UPDATING THE WEIGHTS
    let ml_rate: B = B::ONE / B::from(mlp.batch_size as u32);
    for u in 0..mlp.dim {
        for i in 0..pp.n {
            w.wpack[i][u] -= ml_rate * dpack[u][i];
        }
    }

    // move the below thing lower and lower as you verify the proof
    EpochProof::<B> {
        epoch,
        wpack: wpack_old,
        epoch_ypack,
        zpack: zpack.clone(),
        ashares: ashares.clone(),
        bshares1: bshares1.clone(),
        bshares2: bshares2.clone(),
        f2shares: f2shares.clone(),
        ed2shares: ed2shares.clone(),
        rd2shares: rd2shares.clone(),
        spack: spack.clone(),
        cshares: cshares.clone(),
    }
}

// Tests
#[cfg(test)]
mod tests {
    use crate::{
        online::epoch::prove::prove_epoch, sample::data::sample_data_weights, MLParams, N_PARTIES,
    };
    use ark_std::{end_timer, start_timer};

    use human_bytes::human_bytes;
    use math::fields::f128::BaseElement;
    use secret_sharing::pss::PackedSharingParams;

    #[test]
    fn mpcith_kernel() {
        for batch_size in [128, 256, 512, 1024] {
            for dim in [128, 256, 512, 1024] {
                let mlp = MLParams {
                    batch_size,
                    data_size: batch_size,
                    dim,
                    precision: 16,
                    fixp_size: 64,
                };

                let pp = PackedSharingParams::<BaseElement>::new(N_PARTIES / 8);

                let datapack_section = start_timer!(|| "Data packing");
                let (data, mut w) = sample_data_weights::<BaseElement>(&mlp, &pp);
                end_timer!(datapack_section);

                // KERNEL BENCHMARKS
                let epoch_train = start_timer!(|| "Epoch Training");
                let epoch_proof = prove_epoch(&data, &mut w, 0, &mlp, &pp);
                end_timer!(epoch_train);

                println!(
                    "EpochProof size: {} for batch_size: {}, dim: {}",
                    human_bytes(epoch_proof.size(&pp) as f64),
                    batch_size,
                    dim
                );

                let verify_section = start_timer!(|| "Epoch Verification");
                epoch_proof.verify(&mlp, &pp);
                end_timer!(verify_section);
            }
        }
    }
}
