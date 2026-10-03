//! Rank-slice exchange transports (filled in Task 2).

/// Moves amplitudes between rank slices for a `DistStep::Exchange`.
pub trait Exchange<B> {}

/// All ranks on one device (filled in Task 2).
pub struct LocalExchange;
