use core::marker::PhantomData;
use crypto::{ElementHasher, Hasher, RandomCoin};
use math::FieldElement;
use utils::collections::Vec;

// FIAT-SHAMIR CHANNEL TRAIT
// ================================================================================================

/// Defines an interface for a channel over which a prover communicates with a verifier.
///
/// The prover uses this channel to send messages to the verifier, and
/// (potentially, adaptively) draw random values from the channel
pub trait FSChannel<E: FieldElement> {
    /// Hash function used by the prover to derive random coins.
    type Hasher: Hasher;

    /// Sends a message to the verifier.
    /// The message must be a given as the Digest of a hash
    fn send_msg(&mut self, msg: <<Self as FSChannel<E>>::Hasher as Hasher>::Digest);

    /// Returns a random coin drawn uniformly at random from the entire field.
    fn draw_coin(&mut self) -> E;

    /// Returns num_positions number of random coins uniformly at random from [0, domain_size).
    fn draw_positions(&mut self, num_positions: usize, domain_size: usize) -> Vec<usize>;
}

// FIAT-SHAMIR CHANNEL IMPLEMENTATION
// ================================================================================================
/// Provides a default implementation of the [FSChannel] trait.
pub struct DefaultFSChannel<E, H, R>
where
    E: FieldElement,
    H: ElementHasher<BaseField = E::BaseField>,
    R: RandomCoin<BaseField = E::BaseField, Hasher = H>,
{
    public_coin: R,
    messages: Vec<H::Digest>,
    _field_element: PhantomData<E>,
}

impl<E, H, R> DefaultFSChannel<E, H, R>
where
    E: FieldElement,
    H: ElementHasher<BaseField = E::BaseField>,
    R: RandomCoin<BaseField = E::BaseField, Hasher = H>,
{
    /// Returns a new FS channel instantiated.
    pub fn new() -> Self {
        DefaultFSChannel {
            public_coin: RandomCoin::new(&[]),
            messages: Vec::new(),
            _field_element: PhantomData,
        }
    }

    /// Returns the list of messages written by the prover into this channel.
    pub fn messages(&self) -> &[H::Digest] {
        &self.messages
    }
}

impl<E, H, R> FSChannel<E> for DefaultFSChannel<E, H, R>
where
    E: FieldElement,
    H: ElementHasher<BaseField = E::BaseField>,
    R: RandomCoin<BaseField = E::BaseField, Hasher = H>,
{
    type Hasher = H;

    fn send_msg(&mut self, layer_root: H::Digest) {
        self.messages.push(layer_root);
        self.public_coin.reseed(layer_root);
    }

    fn draw_coin(&mut self) -> E {
        self.public_coin.draw().expect("failed to draw FRI alpha")
    }

    fn draw_positions(&mut self, num_positions: usize, domain_size: usize) -> Vec<usize> {
        self.public_coin
            .draw_integers(num_positions, domain_size)
            .expect("failed to draw query position")
    }
}
