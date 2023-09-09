use math::StarkField;

pub fn field_mod<B: StarkField>(dividend: B, divisor: u128) -> B {
    B::from(
        dividend
            .to_string()
            .parse::<u128>()
            .unwrap()
            .rem_euclid(divisor),
    )
}
