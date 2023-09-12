# zkpot 🪴: Experimenting with Zero-Knowledge Proofs of Training [ePrint:2023/1345](https://eprint.iacr.org/2023/1345)

Rust implementation of a the Zero-Knowledge Proof of Training for Logistic Regression models introdued in [ePrint:2023/1345](https://eprint.iacr.org/2023/1345)

**WARNING:** This is an academic proof-of-concept prototype, and in particular has not received careful code review. This implementation is NOT ready for production use.

Requires Rust nightly.

## Overview
This project began as a fork of the [winterfell](https://github.com/facebook/winterfell/) crate and has been modified as follows:

* [`fri`](fri/): Added a batch prover for FRI
* [`mpcith/`](mpcith/): Contains the MPC-in-the-Head proof for the proof of training along with various consistency checks
  * [`consistency/online`](mpcith/src/online/): Contains the online phase of the MPC-in-the-Head protocol.
    * [`online/epoch`](mpcith/src/online/epoch/): Contains the information-theoretic online phase of the MPC-in-the-Head protocol for a training a single batch.
    * [`online/matvec`](mpcith/src/online/matvec/): Contains the cryptographic checks in the online phase.
  * [`consistency/offline/`](mpcith/src/offline/): Contains a script to estimate the offline proof size of the zkpot protocol.
  * [`consistency/sample`](mpcith/src/sample/): Samples a dummy training data set.
  * [`tests`](mpcith/tests/): Contains a test for the online phase of the MPC-in-the-Head protocol.
* [`naive-training/`](naive-training/): Contains an implementation of the training algorithm for logistic regression to estimate cryptographic overhead. [`main.rs`](naive-training/src/main.rs) contains a logistic regression implementation over the f64 rust data type and [`examples/train128.rs`](naive-training/examples/train128.rs) contains the same over the 128 bit field defined in [`f128`](math/src/field/f128/).
* [`poly-args/`](poly-args/): Contains implementations of sum check proofs and its variants.
* [`secret-sharing/`](secret-sharing/): Contains an implementation of packed secret sharing.

To run, execute individual tests with cargo. To run all use 
```
cargo test --release
```

## License
This library is released under the MIT License.
