use math::StarkField;
use poly_args::sumcheck_prod::SCPProof;

pub mod prove;
pub mod verify;

#[derive(Clone)]
pub struct MatVecProof<B: StarkField> {
    pub xtsumproof: Vec<SCPProof<B>>,
    pub xsumproof: Vec<SCPProof<B>>,
}
