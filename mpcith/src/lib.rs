pub mod consistency;
pub mod online;
pub mod sample;

#[derive(Clone, Debug)]
pub struct MLParams {
    pub batch_size: usize,
    pub data_size: usize,
    pub dim: usize,
    pub precision: usize,
    pub fixp_size: usize,
}

impl MLParams {
    pub fn new(
        batch_size: usize,
        data_size: usize,
        dim: usize,
        precision: usize,
        fixp_size: usize,
    ) -> Self {
        Self {
            batch_size,
            data_size,
            dim,
            precision,
            fixp_size,
        }
    }
    pub fn batches(&self) -> usize {
        self.data_size / self.batch_size
    }
}

pub const LDE_BLOWUP: usize = 1 << 1;
pub const FOLDING_FACTOR: usize = 1 << 2;
pub const MAX_REMAINDER_DEGREE: usize = 1 << 4;
pub const FRI_QUERIES: usize = 95;
pub const N_PARTIES: usize = 512;
