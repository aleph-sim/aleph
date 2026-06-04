//! `aleph-stab`: Stabilizer (Aaronson–Gottesman tableau) backend.
//!
//! Clifford circuits (H, S, CNOT, Paulis, …) simulate in O(n) time per
//! gate and O(n²) memory via the CHP tableau formalism. P3-01 provides
//! the tableau core, gate application, and signed-Pauli readout.
//! Measurement (P3-02) and the `Backend` trait impl (P3-03) land later.
//!
//! Reference: Aaronson & Gottesman, "Improved Simulation of Stabilizer
//! Circuits" (2004), <https://arxiv.org/abs/quant-ph/0406196>.

mod bits;
mod dispatch;
mod error;
mod tableau;

// Re-exports added as items land (Tasks 2-10):
// pub use dispatch::apply_gate;
// pub use error::StabError;
// pub use tableau::Tableau;
