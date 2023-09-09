use math::{
    fft::{
        self, evaluate_poly, evaluate_poly_with_offset, interpolate_poly,
        interpolate_poly_with_offset,
    },
    StarkField,
};

/// Currently the packed secret sharing is deterministic, but it can easily be extended to add random values when packing
/// done for ease of debugging. slightly underestimating work
#[derive(Debug, Clone, PartialEq)]
pub struct PackedSharingParams<F> {
    pub t: usize,
    pub l: usize,
    pub n: usize,
    pub share_twiddles: Vec<F>,
    pub secret_twiddles: Vec<F>,
    pub secret2_twiddles: Vec<F>,
    pub inv_share_twiddles: Vec<F>,
    pub inv_secret_twiddles: Vec<F>,
    pub inv_secret2_twiddles: Vec<F>,
}

impl<F: StarkField> PackedSharingParams<F> {
    #[allow(unused)]
    pub fn new(l: usize) -> Self {
        let n = l * 8;
        let t = l - 1;
        debug_assert_eq!(n, 4 * (t + l + 1));

        let share_twiddles = fft::get_twiddles::<F>(n);
        let secret_twiddles = fft::get_twiddles::<F>(l + t + 1);
        let secret2_twiddles = fft::get_twiddles::<F>(2 * (l + t + 1));

        let inv_share_twiddles = fft::get_inv_twiddles::<F>(n);
        let inv_secret_twiddles = fft::get_inv_twiddles::<F>(l + t + 1);
        let inv_secret2_twiddles = fft::get_inv_twiddles::<F>(2 * (l + t + 1));

        debug_assert_eq!(share_twiddles.len() * 2, n);
        debug_assert_eq!(secret_twiddles.len() * 2, l + t + 1);
        debug_assert_eq!(secret2_twiddles.len() * 2, 2 * (l + t + 1));

        return PackedSharingParams {
            t,
            l,
            n,
            share_twiddles,
            secret_twiddles,
            secret2_twiddles,
            inv_share_twiddles,
            inv_secret_twiddles,
            inv_secret2_twiddles,
        };
    }

    #[allow(unused)]
    pub fn pack_from_public(&self, secrets: &Vec<F>) -> Vec<F> {
        let mut result = secrets.clone();
        self.pack_from_public_in_place(&mut result);

        result
    }

    #[allow(unused)]
    pub fn pack_from_public_in_place(&self, mut secrets: &mut Vec<F>) {
        debug_assert_eq!(secrets.len(), self.inv_secret_twiddles.len());

        // interpolating on secrets domain
        // extend to l+t+1
        secrets.resize(self.l + self.t + 1, F::ZERO);
        interpolate_poly_with_offset(&mut secrets, &self.inv_secret_twiddles, F::GENERATOR);

        // resizing to share domain
        secrets.resize(self.n, F::ZERO);

        evaluate_poly(&mut secrets, &self.share_twiddles);
    }

    #[allow(unused)]
    pub fn unpack_in_place(&self, mut shares: &mut Vec<F>) {
        debug_assert_eq!(shares.len(), self.share_twiddles.len() * 2);

        // interpolating on secrets domain
        interpolate_poly(&mut shares, &self.inv_share_twiddles);

        // resizing to share domain
        shares.resize(self.secret_twiddles.len() * 2, F::ZERO);

        *shares = evaluate_poly_with_offset(&mut shares, &self.secret_twiddles, F::GENERATOR, 1);
        shares.truncate(self.l)
    }

    #[allow(unused)]
    pub fn unpack2_in_place(&self, mut shares: &mut Vec<F>) {
        debug_assert_eq!(shares.len(), self.share_twiddles.len() * 2);

        // interpolating on secrets domain
        interpolate_poly(&mut shares, &self.inv_share_twiddles);

        // resizing to share domain
        shares.resize(self.secret2_twiddles.len() * 2, F::ZERO);

        *shares = evaluate_poly_with_offset(&mut shares, &self.secret2_twiddles, F::GENERATOR, 1);

        // drop alternate elements from shares array
        shares.truncate(2 * self.l);
        *shares = shares.iter().step_by(2).map(|x| *x).collect();
    }
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use math::fields::f128::BaseElement;
    use rand_utils::rand_vector;
    use PackedSharingParams;
    #[test]
    fn test_initialize() {
        let l: usize = 2;

        let pp = PackedSharingParams::<BaseElement>::new(l);
        debug_assert_eq!(pp.t, l - 1);
        debug_assert_eq!(pp.l, l);
        debug_assert_eq!(pp.n, 8 * l);
        debug_assert_eq!(pp.share_twiddles.len() * 2, 8 * l);
        debug_assert_eq!(pp.secret_twiddles.len() * 2, pp.l + pp.t + 1);
        debug_assert_eq!(pp.secret2_twiddles.len() * 2, 2 * (pp.l + pp.t + 1));
    }

    #[test]
    fn test_pack_from_public() {
        let l: usize = 2;

        let pp = PackedSharingParams::<BaseElement>::new(l);

        let mut secrets: Vec<BaseElement> = rand_vector(l);
        let expected = secrets.clone();

        pp.pack_from_public_in_place(&mut secrets);
        pp.unpack_in_place(&mut secrets);

        debug_assert_eq!(expected, secrets);
    }

    #[test]
    fn test_multiplication() {
        let l: usize = 2;

        let pp = PackedSharingParams::<BaseElement>::new(l);

        let mut secrets: Vec<BaseElement> = rand_vector(l);
        let expected: Vec<BaseElement> = secrets.iter().map(|x| (*x) * (*x)).collect();

        pp.pack_from_public_in_place(&mut secrets);

        let mut shares: Vec<BaseElement> = secrets.iter().map(|x| (*x) * (*x)).collect();

        pp.unpack2_in_place(&mut shares);

        debug_assert_eq!(expected, shares);
    }
}
