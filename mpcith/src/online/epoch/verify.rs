/// Defines an struct which contains a proof
/// Has two main interfaces. Prove and Verify.
/// Will contain shares, FRI commitments and openings
// The data structure is a proof for each epoch.
// The verifier will iterate through each epochs
// and check consistnecy of views and FRI Openings
use math::StarkField;
use secret_sharing::pss::PackedSharingParams;

use crate::{online::utils::field_mod, MLParams};

use super::EpochProof;

impl<B> EpochProof<B>
where
    B: StarkField,
{
    pub fn verify(&self, mlp: &MLParams, pp: &PackedSharingParams<B>) {
        // ROUND 1 ================================================
        // Happens across all epochs at once

        // ROUND 2 ================================================
        // Copy the behaviour of king
        let mut zshares: Vec<Vec<B>> = self.zpack.clone();
        for u in 0..mlp.batch_size / pp.l {
            pp.unpack2_in_place(&mut zshares[u]);
            pp.pack_from_public_in_place(&mut zshares[u]);
        }

        // ROUND 3 ================================================
        // For n/8, check that ashares = zshares - beta + 2^m.masku + maskd
        for i in 0..pp.l {
            for u in 0..mlp.batch_size / pp.l {
                assert_eq!(self.ashares[u][i], zshares[u][i] + B::ONE - B::ONE);
            }
        }

        let mut asharesmod = self.ashares.clone();
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
        let mut bshares = vec![vec![B::ZERO; pp.l]; mlp.batch_size / pp.l];
        for i in 0..pp.l {
            for u in 0..mlp.batch_size / pp.l {
                bshares[u][i] = zshares[u][i] - asharesmod[u][i];
                bshares[u][i] *= minv; //multiply by 2^-m
            }
        }

        drop(zshares);
        drop(asharesmod);

        // linear regression
        // ROUND 4 ================================================
        // todo: For l-1, check that bshares1 and bshares2 were computed corrctly from bshares
        let fix_size: u128 = 1 << mlp.fixp_size; //2^M

        let mut bshift1 = bshares.clone();
        let shift1 = B::from(fix_size) + B::ONE / B::from(2 as u32);
        for u in 0..mlp.batch_size / pp.l {
            for i in 0..pp.l {
                bshift1[u][i] += shift1;
            }
        }

        let mut bshift2 = bshares.clone();
        let shift2 = B::from(fix_size) - B::ONE / B::from(2 as u32);
        for u in 0..mlp.batch_size / pp.l {
            for i in 0..pp.l {
                bshift2[u][i] += shift2;
            }
        }

        // assert that bshares1 = bshift1+mask and bshares2 = bshift2+mask
        for u in 0..mlp.batch_size / pp.l {
            for i in 0..pp.l {
                assert_eq!(self.bshares1[u][i], bshift1[u][i] + B::ZERO);
                assert_eq!(self.bshares2[u][i], bshift2[u][i] + B::ZERO);
            }
        }

        let fix_sizem1: u128 = 1 << (mlp.fixp_size - 1); //2^{M-1}
        let inv_fix_sizem1 = B::ONE / B::from(fix_sizem1); //1/2^{M-1}

        let mut bshares1mod = self.bshares1.clone();
        let mut bshares2mod = self.bshares2.clone();

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

        let mut ind1shares = bshift1.clone();
        let mut ind2shares = bshift2.clone();

        for u in 0..mlp.batch_size / pp.l {
            for i in 0..pp.l {
                ind1shares[u][i] -= bshares1mod[u][i];
                ind2shares[u][i] -= bshares2mod[u][i];

                ind1shares[u][i] *= inv_fix_sizem1;
                ind2shares[u][i] *= inv_fix_sizem1;
            }
        }

        drop(bshift1);
        drop(bshift2);
        drop(bshares1mod);
        drop(bshares2mod);

        // ROUND 5 ================================================
        // assert that f2shares = ind1shares*(1-ind2shares)
        for u in 0..mlp.batch_size / pp.l {
            for i in 0..pp.l {
                assert_eq!(
                    self.f2shares[u][i],
                    ind1shares[u][i] * (B::ONE - ind2shares[u][i])
                );
                // todo: add randomness to the rhs
            }
        }

        let mut fshares = self.f2shares.clone();
        for u in 0..mlp.batch_size / pp.l {
            // degree reduction
            pp.unpack2_in_place(&mut fshares[u]);
            pp.pack_from_public_in_place(&mut fshares[u]);
        }

        // set ind12shares = fshares - mask
        let ind12shares = fshares.clone();

        let twoinv = B::ONE / B::from(2 as u32);
        for u in 0..mlp.batch_size / pp.l {
            for i in 0..pp.l {
                assert_eq!(
                    self.ed2shares[u][i],
                    ind12shares[u][i] * (bshares[u][i] + twoinv) + ind2shares[u][i]
                );
            }
        }

        let mut edshares = self.ed2shares.clone();
        for u in 0..mlp.batch_size / pp.l {
            // degree reduction of ehsares
            pp.unpack2_in_place(&mut edshares[u]);
            pp.pack_from_public_in_place(&mut edshares[u]);
        }

        let eshares = edshares.clone(); //sub randomness here

        drop(fshares);

        // ROUND 6 ================================================
        // compute rshares = eshares - epoch_ypack
        let mut rd2shares = vec![vec![B::ZERO; pp.l]; mlp.batch_size / pp.l];
        for u in 0..mlp.batch_size / pp.l {
            for i in 0..pp.l {
                rd2shares[u][i] = eshares[u][i] - self.epoch_ypack[i][u]; //add mask here
            }
        }

        // assert rd2shares and self.rd2shares are equal
        for u in 0..mlp.batch_size / pp.l {
            for i in 0..pp.l {
                assert_eq!(self.rd2shares[u][i], rd2shares[u][i]);
            }
        }

        let mut rd = self.rd2shares.clone();
        for u in 0..mlp.batch_size / pp.l {
            pp.unpack2_in_place(&mut rd[u]);
        }

        let rd = rd.into_iter().flatten().collect::<Vec<B>>();
        // rpack is of size Bxn
        let mut rpack = Vec::new();
        for k in 0..mlp.batch_size {
            rpack.push(pp.pack_from_public(&mut vec![rd[k]; pp.l]))
            // todo: subtract randomness
        }

        // Data multiplied by weight check happens outside

        // ROUND 7 ================================================
        // Copy the behaviour of king
        let mut sshares: Vec<Vec<B>> = self.spack.clone();
        for u in 0..mlp.dim / pp.l {
            pp.unpack2_in_place(&mut sshares[u]);
            pp.pack_from_public_in_place(&mut sshares[u]);
        }

        // ROUND 8 ================================================
        // For l-1, check that cshares = sshares - delta + 2^m.masku + maskd
        for i in 0..pp.n {
            for u in 0..mlp.dim / pp.l {
                assert_eq!(self.cshares[u][i], sshares[u][i] + B::ONE - B::ONE);
            }
        }

        let mut csharesmod = self.cshares.clone();
        for u in 0..mlp.dim / pp.l {
            pp.unpack_in_place(&mut csharesmod[u]);
            for j in 0..pp.l {
                csharesmod[u][j] = field_mod(csharesmod[u][j], m);
            }
            pp.pack_from_public_in_place(&mut csharesmod[u]);
        }

        // compute dshares
        let mut dshares = vec![vec![B::ZERO; pp.l]; mlp.dim / pp.l];
        for i in 0..pp.l {
            for u in 0..mlp.dim / pp.l {
                dshares[u][i] = sshares[u][i] - csharesmod[u][i];
                dshares[u][i] *= minv; //multiply by 2^-m
            }
        }

        drop(sshares);
        drop(csharesmod);
        drop(dshares);

        // ROUND 9 ================================================
        // todo: check that the new weights and old weights are correctly related
    }
}
