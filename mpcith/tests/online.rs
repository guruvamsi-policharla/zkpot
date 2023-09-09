use ark_std::{end_timer, start_timer};
use crypto::{hashers::Blake3_256, DefaultRandomCoin};
use fri::{BatchProver, DefaultProverChannel, FriOptions};
use human_bytes::human_bytes;
use math::fields::f128::BaseElement;
use mpcith::{
    online::{
        epoch::prove::prove_epoch,
        matvec::{prove::mat_vec_prove, verify::mat_vec_verify},
    },
    sample::data::sample_dummy,
    MLParams, FOLDING_FACTOR, FRI_QUERIES, LDE_BLOWUP, MAX_REMAINDER_DEGREE, N_PARTIES,
};
use secret_sharing::pss::PackedSharingParams;

type Blake3 = Blake3_256<BaseElement>;

#[test]
fn epoch_prove_verify() {
    pub const BATCH_SIZE: usize = 1024;
    pub const D: usize = 512;

    pub const DATA_SIZE: usize = 1 << 20;
    pub const BATCHES: usize = DATA_SIZE / BATCH_SIZE;

    let mlp = MLParams {
        batch_size: BATCH_SIZE,
        data_size: DATA_SIZE,
        dim: D,
        precision: 16,
        fixp_size: 64,
    };

    println!("mlp:{:?}", mlp);

    let pp = PackedSharingParams::<BaseElement>::new(N_PARTIES / 8);
    let options = FriOptions::new(LDE_BLOWUP, FOLDING_FACTOR, MAX_REMAINDER_DEGREE);

    let datapack_section = start_timer!(|| "Data packing");
    let (data, mut w) = sample_dummy::<BaseElement>(&mlp, &pp);
    end_timer!(datapack_section);

    let mut batch_prover_n = BatchProver::<
        BaseElement,
        BaseElement,
        Blake3,
        DefaultRandomCoin<Blake3_256<BaseElement>>,
    >::new(options.clone(), mlp.data_size - 1);

    let mut batch_prover_ndb = BatchProver::<
        BaseElement,
        BaseElement,
        Blake3,
        DefaultRandomCoin<Blake3_256<BaseElement>>,
    >::new(options.clone(), mlp.dim * mlp.batches() - 1);

    let train_section = start_timer!(|| "Training");
    let mut epoch_proofs = Vec::new();
    for i in 0..BATCHES {
        epoch_proofs.push(prove_epoch(&data, &mut w, i, &mlp, &pp));
        println!("Trained epoch {}", i);
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
    let batch_opening_n = batch_prover_n.batch_prove(&mut channel_n, open_pos, mlp.data_size - 1);

    let mut channel_ndb = DefaultProverChannel::new(
        mlp.dim * mlp.batches() * options.blowup_factor(),
        FRI_QUERIES,
    );
    let batch_opening_ndb =
        batch_prover_ndb.batch_prove(&mut channel_ndb, open_pos, mlp.dim * mlp.batches() - 1);
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
