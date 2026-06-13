//! Channel application (quantum-jump) and per-shot RNG seeding.

/// Deterministic per-shot seed: a splitmix64 mix of `(seed, shot)` so shot
/// outcomes are reproducible regardless of rayon scheduling (spec §1).
///
/// # Why splitmix64?
/// Plain `seed + shot` leaves strong linear correlation between adjacent shots
/// that can skew Monte-Carlo estimators. The splitmix64 finalizer provides full
/// 64-bit avalanche with no correlation at a cost of ~4 instructions per shot.
// used by run_noisy in Task 7
#[allow(dead_code)]
pub(super) fn shot_seed(seed: u64, shot: u64) -> u64 {
    // splitmix64 finalizer applied to (seed·INC + shot + INC).
    // Adjacent shots differ by 1 pre-finalize; the three multiply-xorshift
    // rounds provide full 64-bit avalanche so post-finalize outputs are
    // statistically independent despite the linear pre-image spacing.
    let mut z = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(shot)
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shot_seed_is_deterministic_and_distinct() {
        assert_eq!(shot_seed(7, 3), shot_seed(7, 3));
        assert_ne!(shot_seed(7, 3), shot_seed(7, 4));
        assert_ne!(shot_seed(7, 3), shot_seed(8, 3));
    }
}
