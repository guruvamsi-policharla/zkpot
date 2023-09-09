use ark_std::{end_timer, start_timer};
use fri::utils::transpose;
use math::StarkField;
use rand_utils::rand_vector_range;
use secret_sharing::pss::PackedSharingParams;

use crate::MLParams;

// Sampling the data for the MPC protocol which will be converted into an MPCITH

/// Struct containing the data and packed shares of the data
pub struct MPCData<B: StarkField> {
    pub y: Vec<Vec<Vec<B>>>,
    pub ypack: Vec<Vec<Vec<B>>>,
    pub x: Vec<Vec<Vec<Vec<B>>>>,
    pub xpack: Vec<Vec<Vec<Vec<B>>>>,
    pub xt: Vec<Vec<Vec<Vec<B>>>>,
    pub xtpack: Vec<Vec<Vec<Vec<B>>>>,
}

/// Struct containing the weights and packed shares of the weights
pub struct MPCWeights<B: StarkField> {
    pub w: Vec<B>,
    pub wpack: Vec<Vec<B>>,
}

/// Sampling and packing data and initial weights into shares
pub fn sample_data_weights<B: StarkField>(
    mlp: &MLParams,
    pp: &PackedSharingParams<B>,
) -> (MPCData<B>, MPCWeights<B>) {
    let mut y: Vec<Vec<Vec<B>>> = Vec::new();
    let mut tempypack: Vec<Vec<Vec<B>>> = Vec::new();
    let mut ypack: Vec<Vec<Vec<B>>>;

    let mut x: Vec<Vec<Vec<Vec<B>>>> = Vec::new();
    let mut tempxpack: Vec<Vec<Vec<Vec<B>>>> = Vec::new();
    let mut xpack: Vec<Vec<Vec<Vec<B>>>>;

    let mut xt: Vec<Vec<Vec<Vec<B>>>> = Vec::new();
    let mut tempxtpack: Vec<Vec<Vec<Vec<B>>>> = Vec::new();
    let mut xtpack: Vec<Vec<Vec<Vec<B>>>>;

    let w: Vec<B>;
    let mut wpack: Vec<Vec<B>> = Vec::new();

    // x and y have different packing structures

    let y_timer = start_timer!(|| "sample y");
    // y is a vector of DATA_SIZE
    // Stored as a cube of BATCHESx(BATCH_SIZE/PACKING_FACTOR)xPACKING_FACTOR
    // tempypack is a cube of BATCHESx(BATCH_SIZE/PACKING_FACTOR)xN_PARTIES
    // ypack is a cube of N_PARTIESxBATCHESx(BATCH_SIZE/PACKING_FACTOR)
    for i in 0..mlp.batches() {
        y.push(Vec::new());
        tempypack.push(Vec::new());

        for j in 0..mlp.batch_size / pp.l {
            y[i].push(rand_vector_range(pp.l, mlp.precision));

            tempypack[i].push(pp.pack_from_public(&y[i][j]));
        }
    }

    ypack = vec![vec![vec![B::ZERO; mlp.batch_size / pp.l]; mlp.batches()]; pp.n];
    for i in 0..pp.n {
        for j in 0..mlp.batches() {
            for k in 0..mlp.batch_size / pp.l {
                ypack[i][j][k] = tempypack[j][k][i];
            }
        }
    }
    drop(tempypack);
    end_timer!(y_timer);

    let raw_timer = start_timer!(|| "sample raw");
    // xraw is a BATCHESxBATCH_SIZExD matrix
    // xtraw is a BATCHESxDxBATCH_SIZE matrix
    let mut xraw: Vec<Vec<Vec<B>>> = Vec::new();
    let mut xtraw: Vec<Vec<Vec<B>>> = Vec::new();

    for i in 0..mlp.batches() {
        xraw.push(Vec::new());

        for _ in 0..mlp.batch_size {
            xraw[i].push(rand_vector_range(mlp.dim, mlp.precision));
        }

        xtraw.push(transpose(xraw[i].clone()));
    }
    end_timer!(raw_timer);

    let x_timer = start_timer!(|| "sample x");
    // x is stored as a hypercube of BATCHESxBATCH_SIZEx(D/PACKING_FACTOR)xPACKING_FACTOR
    // tempxpack is a hypercube of BATCHESxBATCH_SIZEx(D/PACKING_FACTOR)xN_PARTIES
    // xpack is a hypercube of N_PARTIESxBATCHESx(D/PACKING_FACTOR)x(BATCH_SIZE)
    for i in 0..mlp.batches() {
        x.push(Vec::new());
        tempxpack.push(Vec::new());

        for j in 0..mlp.batch_size {
            x[i].push(Vec::new());
            tempxpack[i].push(Vec::new());

            for k in 0..mlp.dim / pp.l {
                x[i][j].push(xraw[i][j][k * pp.l..(k + 1) * pp.l].to_vec());

                tempxpack[i][j].push(pp.pack_from_public(&x[i][j][k]));
            }
        }
    }

    xpack = vec![vec![vec![vec![B::ZERO; mlp.batch_size]; mlp.dim / pp.l]; mlp.batches()]; pp.n];
    for i in 0..pp.n {
        for j in 0..mlp.batches() {
            for k in 0..mlp.dim / pp.l {
                for ii in 0..mlp.batch_size {
                    xpack[i][j][k][ii] = tempxpack[j][ii][k][i];
                }
            }
        }
    }
    drop(tempxpack);
    end_timer!(x_timer);

    let xt_timer = start_timer!(|| "sample xt");
    // xt is stored as a hypercube of BATCHESxDx(BATCH_SIZE/PACKING_FACTOR)xPACKING_FACTOR
    // tempxtpack is a hypercube of BATCHESxDx(BATCH_SIZE/PACKING_FACTOR)xN_PARTIES
    // xtpack is a hypercube of N_PARTIESxBATCHESx(BATCH_SIZE/PACKING_FACTOR)x(D)
    for i in 0..mlp.batches() {
        xt.push(Vec::new());
        tempxtpack.push(Vec::new());

        for j in 0..mlp.dim {
            xt[i].push(Vec::new());
            tempxtpack[i].push(Vec::new());

            for k in 0..mlp.batch_size / pp.l {
                xt[i][j].push(xtraw[i][j][k * pp.l..(k + 1) * pp.l].to_vec());

                tempxtpack[i][j].push(pp.pack_from_public(&xt[i][j][k]));
            }
        }
    }

    xtpack = vec![vec![vec![vec![B::ZERO; mlp.dim]; mlp.batch_size / pp.l]; mlp.batches()]; pp.n];
    for i in 0..pp.n {
        for j in 0..mlp.batches() {
            for k in 0..mlp.batch_size / pp.l {
                for ii in 0..mlp.dim {
                    xtpack[i][j][k][ii] = tempxtpack[j][ii][k][i];
                }
            }
        }
    }
    drop(tempxtpack);
    end_timer!(xt_timer);
    // w is a vector of D
    // w is stored as a matrix of D
    // wpack is a matrix of N_PARTIESx(D)
    w = rand_vector_range(mlp.dim, mlp.precision);

    for i in 0..mlp.dim {
        wpack.push(pp.pack_from_public(&mut vec![w[i]; pp.l]));
    }

    wpack = transpose(wpack);

    (
        MPCData {
            y,
            ypack,
            x,
            xpack,
            xt,
            xtpack,
        },
        MPCWeights { w, wpack },
    )
}

/// Sampling dummy data and initial weights into shares
/// runs much faster than sample_data_weights
pub fn sample_dummy<B: StarkField>(
    mlp: &MLParams,
    pp: &PackedSharingParams<B>,
) -> (MPCData<B>, MPCWeights<B>) {
    let mut y: Vec<Vec<Vec<B>>> = Vec::new();
    let mut tempypack: Vec<Vec<Vec<B>>> = Vec::new();
    let mut ypack: Vec<Vec<Vec<B>>>;

    let mut x: Vec<Vec<Vec<Vec<B>>>> = Vec::new();
    let mut tempxpack: Vec<Vec<Vec<Vec<B>>>> = Vec::new();
    let mut xpack: Vec<Vec<Vec<Vec<B>>>>;

    let mut xt: Vec<Vec<Vec<Vec<B>>>> = Vec::new();
    let mut tempxtpack: Vec<Vec<Vec<Vec<B>>>> = Vec::new();
    let mut xtpack: Vec<Vec<Vec<Vec<B>>>>;

    let w: Vec<B>;
    let mut wpack: Vec<Vec<B>> = Vec::new();

    // x and y have different packing structures

    // y is a vector of DATA_SIZE
    // Stored as a cube of BATCHESx(BATCH_SIZE/PACKING_FACTOR)xPACKING_FACTOR
    // tempypack is a cube of BATCHESx(BATCH_SIZE/PACKING_FACTOR)xN_PARTIES
    // ypack is a cube of N_PARTIESxBATCHESx(BATCH_SIZE/PACKING_FACTOR)
    for i in 0..1 {
        y.push(Vec::new());
        tempypack.push(Vec::new());

        for j in 0..mlp.batch_size / pp.l {
            y[i].push(rand_vector_range(pp.l, mlp.precision));

            tempypack[i].push(pp.pack_from_public(&y[i][j]));
        }
    }

    for _ in 1..mlp.batches() {
        y.push(y[0].clone());
    }

    ypack = vec![vec![vec![B::ZERO; mlp.batch_size / pp.l]; mlp.batches()]; pp.n];
    for i in 0..pp.n {
        for j in 0..mlp.batches() {
            for k in 0..mlp.batch_size / pp.l {
                ypack[i][j][k] = tempypack[0][k][i];
            }
        }
    }
    drop(tempypack);

    // xraw is a BATCHESxBATCH_SIZExD matrix
    // xtraw is a BATCHESxDxBATCH_SIZE matrix
    let mut xraw: Vec<Vec<Vec<B>>> = Vec::new();
    let mut xtraw: Vec<Vec<Vec<B>>> = Vec::new();

    for i in 0..1 {
        xraw.push(Vec::new());

        for _ in 0..mlp.batch_size {
            xraw[i].push(rand_vector_range(mlp.dim, mlp.precision));
        }

        xtraw.push(transpose(xraw[i].clone()));
    }

    // x is stored as a hypercube of BATCHESxBATCH_SIZEx(D/PACKING_FACTOR)xPACKING_FACTOR
    // tempxpack is a hypercube of BATCHESxBATCH_SIZEx(D/PACKING_FACTOR)xN_PARTIES
    // xpack is a hypercube of N_PARTIESxBATCHESx(D/PACKING_FACTOR)x(BATCH_SIZE)
    for i in 0..1 {
        x.push(Vec::new());
        tempxpack.push(Vec::new());

        for j in 0..mlp.batch_size {
            x[i].push(Vec::new());
            tempxpack[i].push(Vec::new());

            for k in 0..mlp.dim / pp.l {
                x[i][j].push(xraw[i][j][k * pp.l..(k + 1) * pp.l].to_vec());

                tempxpack[i][j].push(pp.pack_from_public(&x[i][j][k]));
            }
        }
    }

    for _ in 1..mlp.batches() {
        x.push(x[0].clone());
    }

    xpack = vec![vec![vec![vec![B::ZERO; mlp.batch_size]; mlp.dim / pp.l]; mlp.batches()]; pp.n];
    for i in 0..pp.n {
        for j in 0..mlp.batches() {
            for k in 0..mlp.dim / pp.l {
                for ii in 0..mlp.batch_size {
                    xpack[i][j][k][ii] = tempxpack[0][ii][k][i];
                }
            }
        }
    }
    drop(tempxpack);

    // xt is stored as a hypercube of BATCHESxDx(BATCH_SIZE/PACKING_FACTOR)xPACKING_FACTOR
    // tempxtpack is a hypercube of BATCHESxDx(BATCH_SIZE/PACKING_FACTOR)xN_PARTIES
    // xtpack is a hypercube of N_PARTIESxBATCHESx(BATCH_SIZE/PACKING_FACTOR)x(D)
    for i in 0..1 {
        xt.push(Vec::new());
        tempxtpack.push(Vec::new());

        for j in 0..mlp.dim {
            xt[i].push(Vec::new());
            tempxtpack[i].push(Vec::new());

            for k in 0..mlp.batch_size / pp.l {
                xt[i][j].push(xtraw[i][j][k * pp.l..(k + 1) * pp.l].to_vec());

                tempxtpack[i][j].push(pp.pack_from_public(&xt[i][j][k]));
            }
        }
    }

    for _ in 1..mlp.batches() {
        xt.push(xt[0].clone());
    }

    xtpack = vec![vec![vec![vec![B::ZERO; mlp.dim]; mlp.batch_size / pp.l]; mlp.batches()]; pp.n];
    for i in 0..pp.n {
        for j in 0..mlp.batches() {
            for k in 0..mlp.batch_size / pp.l {
                for ii in 0..mlp.dim {
                    xtpack[i][j][k][ii] = tempxtpack[0][ii][k][i];
                }
            }
        }
    }
    drop(tempxtpack);

    // w is a vector of D
    // w is stored as a matrix of D
    // wpack is a matrix of N_PARTIESx(D)
    w = rand_vector_range(mlp.dim, mlp.precision);

    for i in 0..mlp.dim {
        wpack.push(pp.pack_from_public(&mut vec![w[i]; pp.l]));
    }

    wpack = transpose(wpack);

    (
        MPCData {
            y,
            ypack,
            x,
            xpack,
            xt,
            xtpack,
        },
        MPCWeights { w, wpack },
    )
}

// Tests
#[cfg(test)]
mod tests {
    use crate::MLParams;

    use super::*;
    use math::fields::f128::BaseElement;

    pub const BATCH_SIZE: usize = 256;
    pub const DATA_SIZE: usize = BATCH_SIZE;
    pub const D: usize = 128;

    pub const N_PARTIES: usize = 512;
    pub const PRECISION: usize = 16;

    #[test]
    fn test_sample() {
        let mlp = MLParams {
            batch_size: BATCH_SIZE,
            data_size: DATA_SIZE,
            dim: D,
            precision: PRECISION,
            fixp_size: 64,
        };

        let pp = PackedSharingParams::<BaseElement>::new(N_PARTIES / 8);
        let (_data, _w) = sample_data_weights(&mlp, &pp);
    }

    #[test]
    fn test_dummy_sample() {
        let mlp = MLParams {
            batch_size: BATCH_SIZE,
            data_size: DATA_SIZE,
            dim: D,
            precision: PRECISION,
            fixp_size: 64,
        };

        let pp = PackedSharingParams::<BaseElement>::new(N_PARTIES / 8);
        let (_data, _w) = sample_dummy(&mlp, &pp);
    }
}
