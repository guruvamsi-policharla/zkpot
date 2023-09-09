// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

use crypto::ElementHasher;
use math::{batch_inversion, fft, FieldElement, StarkField};
use utils::{collections::Vec, iter_mut, uninit_vector};

/// Maps positions in the evaluation domain to indexes of commitment Merkle tree.
pub fn map_positions_to_indexes(
    positions: &[usize],
    source_domain_size: usize,
    folding_factor: usize,
    num_partitions: usize,
) -> Vec<usize> {
    // if there was only 1 partition, order of elements in the commitment tree
    // is the same as the order of elements in the evaluation domain
    if num_partitions == 1 {
        return positions.to_vec();
    }

    let target_domain_size = source_domain_size / folding_factor;
    let partition_size = target_domain_size / num_partitions;

    let mut result = Vec::new();
    for position in positions {
        let partition_idx = position % num_partitions;
        let local_idx = (position - partition_idx) / num_partitions;
        let position = partition_idx * partition_size + local_idx;
        result.push(position);
    }

    result
}

/// Hashes each of the arrays in the provided slice and returns a vector of resulting hashes.
pub fn hash_values<H, E, const N: usize>(values: &[[E; N]]) -> Vec<H::Digest>
where
    E: FieldElement,
    H: ElementHasher<BaseField = E::BaseField>,
{
    let mut result: Vec<H::Digest> = unsafe { uninit_vector(values.len()) };
    iter_mut!(result, 1024).zip(values).for_each(|(r, v)| {
        *r = H::hash_elements(v);
    });
    result
}

/// Hashes each of the arrays in the provided slice and returns a vector of resulting hashes.
pub fn hash_vec_values<H, E>(values: &Vec<Vec<E>>) -> Vec<H::Digest>
where
    E: FieldElement,
    H: ElementHasher<BaseField = E::BaseField>,
{
    let mut result: Vec<H::Digest> = unsafe { uninit_vector(values.len()) };
    iter_mut!(result, 1024).zip(values).for_each(|(r, v)| {
        *r = H::hash_elements(v.as_slice());
    });
    result
}

/// given evals of a polynomial, extends it to domain_size evaluations
pub fn low_degree_extension<B>(evals: &mut Vec<B>, domain_size: usize)
where
    B: StarkField,
{
    // interpolate the polynomial
    let inv_twiddles = fft::get_inv_twiddles::<B>(evals.len());
    fft::interpolate_poly(evals, &inv_twiddles);

    // resize to domain_size and evaluate
    evals.resize(domain_size, B::ZERO);
    let twiddles = fft::get_twiddles::<B>(domain_size);
    fft::evaluate_poly(evals, &twiddles);
}

/// given coeffs of a polynomial, extends it to domain_size evaluations
pub fn low_degree_evaluation<B>(coeffs: &mut Vec<B>, domain_size: usize)
where
    B: StarkField,
{
    // resize to domain_size and evaluate
    coeffs.resize(domain_size, B::ZERO);
    let twiddles = fft::get_twiddles::<B>(domain_size);
    fft::evaluate_poly(coeffs, &twiddles);
}

pub fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>>
where
    T: Clone,
{
    assert!(!v.is_empty());
    (0..v[0].len())
        .map(|i| v.iter().map(|inner| inner[i].clone()).collect::<Vec<T>>())
        .collect()
}

/// given the evaluations of a polynomial and an opening position, compute the x.q(x)
/// where q(x) = (f(x)-f(open_pos))/(x-open_pos)
pub fn get_quotient<B: FieldElement>(
    evals: &Vec<B>,
    open_pos: B,
    opening: B,
    domain: &Vec<B>,
) -> Vec<B> {
    // get all the roots of unity and compute inverse of the denominator using batch_inverse
    let mut denoms = domain.clone();
    denoms.iter_mut().for_each(|x| *x -= open_pos);
    let denoms_inv = batch_inversion(&denoms);

    // Compute the evals of quotient folded polynomial as q(x) = (f(x)-f(open_pos))/(x-open_pos)
    let mut q = evals.clone();
    q.iter_mut().for_each(|y| *y -= opening);

    // multiply q by denoms_inv
    q.iter_mut()
        .zip(denoms_inv.iter())
        .for_each(|(y, &den)| *y *= den);

    // multiply quotient polynomial by x
    let mut xq = q;

    // multiply q by domain to get xq
    xq.iter_mut().zip(domain.iter()).for_each(|(y, &x)| *y *= x);

    xq
}
