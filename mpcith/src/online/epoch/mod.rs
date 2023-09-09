use math::StarkField;
use secret_sharing::pss::PackedSharingParams;
use std::mem::size_of;

pub mod prove;
pub mod verify;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpochProof<B>
where
    B: StarkField,
{
    // parameters
    pub epoch: usize,

    // Data
    pub wpack: Vec<Vec<B>>,       //view of n/8
    pub epoch_ypack: Vec<Vec<B>>, //view of n/8
    // ypack contains the current epochs data only

    // Round 1
    pub zpack: Vec<Vec<B>>, //All the shares

    // ROUND 2
    // No new things needed

    // ROUND 3
    pub ashares: Vec<Vec<B>>, //All the shares

    // ROUND 4
    pub bshares1: Vec<Vec<B>>, //All the shares
    pub bshares2: Vec<Vec<B>>, //All the shares

    // ROUND 5
    pub f2shares: Vec<Vec<B>>,  //All the shares
    pub ed2shares: Vec<Vec<B>>, //All the shares

    // an intermediate round with new protocol
    pub rd2shares: Vec<Vec<B>>, //All the shares

    // ROUND 6
    pub spack: Vec<Vec<B>>, //All the shares

    // ROUND 7
    // No new things needed

    // ROUND 8
    pub cshares: Vec<Vec<B>>, //All the shares
}

impl<B> EpochProof<B>
where
    B: StarkField,
{
    pub fn size(&self, pp: &PackedSharingParams<B>) -> usize {
        let poly_degree = pp.l + pp.t;

        // for each "all shares" without degree doubling we send poly_degree+1 elements
        // if there is degree doubling we send 2*(poly_degree+1) elements
        // rand terms of size 2*(pp.l-1) for each of the "all shares"
        let single_view = poly_degree + 1 + 2 * (pp.l - 1);
        let double_view = 2 * (poly_degree + 1) + 2 * (pp.l - 1);

        let mut size = 0;
        size += self.epoch_ypack.len() * self.epoch_ypack[0].len() * size_of::<B>();
        size += self.zpack.len() * size_of::<B>() * double_view; //account for rand alpha2 + alpha
        size += self.ashares.len() * size_of::<B>() * single_view; //account for rand masku + maskd
        size += self.bshares1.len() * size_of::<B>() * single_view; //account for rand ltu + ltd
        size += self.bshares2.len() * size_of::<B>() * single_view; //account for rand ltbu + ltbd
        size += self.f2shares.len() * size_of::<B>() * double_view; // account for rand theta2 + theta
        size += self.ed2shares.len() * size_of::<B>() * double_view; // account for rand mu2 + mu
        size += self.rd2shares.len() * size_of::<B>() * double_view; //account for rand tau + nu
        size += self.cshares.len() * size_of::<B>() * double_view; // account for rand gamma2 + gamma
        size += self.spack.len() * size_of::<B>() * single_view; // account for rand maskbu + maskbd

        size
    }
}
