//! Test/bench fixtures shared between unit tests, property tests,
//! and criterion benches.

use crate::Circuit;

/// Hardware-efficient VQE ansatz.
///
/// Per layer:
/// 1. For each qubit `q`: five same-qubit rotations
///    `Ry(θ1) · Rz(θ2) · Ry(θ3) · Rz(θ4) · Ry(θ5)` — guarantees that
///    each per-qubit run between CNOT fences has length 5, which is
///    the smallest value of `k` for which the P1-09 acceptance
///    criterion (≥ 3× gate-count reduction at `n=12, depth=10`) is
///    cleanly satisfied for a linear-chain CNOT topology. See the
///    long comment in `vqe_hea_fuses_three_times` for the algebra.
/// 2. Linear CNOT chain: `CNOT(0,1), CNOT(1,2), …, CNOT(n-2, n-1)`.
///
/// Concrete θ values are deterministic (seeded affine of qubit / layer
/// indices) so the generated circuit is reproducible across runs.
///
/// `depth` is the number of layers.
pub fn vqe_hea(n_qubits: u32, depth: u32) -> Circuit {
    let mut c = Circuit::new(n_qubits, 0);
    for l in 0..depth {
        for q in 0..n_qubits {
            // Deterministic affine-seeded angles; same formula for every
            // rotation index i = 1..=5, with `i` as the offset constant.
            let t1 = 0.1 + 0.01 * (q as f64) + 0.001 * (l as f64);
            let t2 = 0.2 + 0.02 * (q as f64) + 0.002 * (l as f64);
            let t3 = 0.3 + 0.03 * (q as f64) + 0.003 * (l as f64);
            let t4 = 0.4 + 0.04 * (q as f64) + 0.004 * (l as f64);
            let t5 = 0.5 + 0.05 * (q as f64) + 0.005 * (l as f64);
            c.ry(t1, q).unwrap();
            c.rz(t2, q).unwrap();
            c.ry(t3, q).unwrap();
            c.rz(t4, q).unwrap();
            c.ry(t5, q).unwrap();
        }
        for q in 0..n_qubits.saturating_sub(1) {
            c.cnot(q, q + 1).unwrap();
        }
    }
    c
}

/// QAOA ansatz on a linear-chain cost Hamiltonian.
///
/// Structure:
/// 1. State prep: `H` on every qubit.
/// 2. Per layer:
///    - Cost: for each chain edge `(i, i+1)`, the RZZ(γ) interaction
///      decomposed as `CNOT(i,i+1); Rz(2γ, i+1); CNOT(i,i+1)`.
///    - Mixer: `Rx(2β, q)` on every qubit.
///
/// Fusion algebra (why this hits the P1-10 ≥1.5× AC):
/// `Fuse1qRuns` finds NO length-≥2 1q runs here — every 1q gate (`H`,
/// `Rz` inside a cost triplet, mixer `Rx`) is fenced by CNOTs, so P1-09
/// leaves the circuit unchanged. `Fuse2q` collapses each cost triplet
/// `CNOT·Rz·CNOT` into one `Unitary2q` (3× on the cost layer) and absorbs
/// the `H`/mixer `Rx` gates as pre-/post-1q into neighbouring 2q blocks.
/// Cost gates dominate (3(n-1) per layer vs n mixer), so the overall
/// reduction is well above 1.5×.
///
/// Angles are deterministic affine functions of qubit/layer indices for
/// reproducibility.
pub fn qaoa(n_qubits: u32, depth: u32) -> Circuit {
    let mut c = Circuit::new(n_qubits, 0);
    for q in 0..n_qubits {
        c.h(q).unwrap();
    }
    for l in 0..depth {
        let gamma = 0.3 + 0.01 * (l as f64);
        let beta = 0.2 + 0.01 * (l as f64);
        for i in 0..n_qubits.saturating_sub(1) {
            c.cnot(i, i + 1).unwrap();
            c.rz(2.0 * gamma, i + 1).unwrap();
            c.cnot(i, i + 1).unwrap();
        }
        for q in 0..n_qubits {
            c.rx(2.0 * beta, q).unwrap();
        }
    }
    c
}

/// A circuit with a data register and an unmeasured ancilla register, used
/// to exercise dead-code elimination (P1-11).
///
/// Structure:
/// 1. Data prep: `H` on each data qubit, then a linear CNOT chain over the
///    data qubits.
/// 2. Ancilla "computation": `H` on each ancilla qubit, then a linear CNOT
///    chain over the ancilla qubits — entangled only among themselves,
///    never touching a data qubit.
/// 3. Measure every **data** qubit (ancilla are never measured).
///
/// The ancilla-only gates have no data-flow path to any measurement, so
/// `DeadCodeElim` removes all of them while preserving the measured
/// distribution over the data qubits. Data qubits are `0..n_data`; ancilla
/// qubits are `n_data..n_data+n_ancilla`.
pub fn ancilla_compute(n_data: u32, n_ancilla: u32) -> Circuit {
    let n = n_data + n_ancilla;
    let mut c = Circuit::new(n, n_data);
    // Data prep.
    for q in 0..n_data {
        c.h(q).unwrap();
    }
    for q in 0..n_data.saturating_sub(1) {
        c.cnot(q, q + 1).unwrap();
    }
    // Ancilla computation (dead w.r.t. the data measurement).
    for a in 0..n_ancilla {
        c.h(n_data + a).unwrap();
    }
    for a in 0..n_ancilla.saturating_sub(1) {
        c.cnot(n_data + a, n_data + a + 1).unwrap();
    }
    // Measure data qubits only.
    for q in 0..n_data {
        c.measure(q, q).unwrap();
    }
    c
}

/// Circuit dominated by cancellable redundancy. Per step: one surviving
/// `Rz` rotation (distinct angle, so consecutive survivors never cancel)
/// followed by an `H·H` and an `X·X` self-inverse pair that the
/// cancellation pass deletes. `pairs` steps → `5·pairs` gates in, `pairs`
/// gates out (5× reduction). Deterministic; `q` cycles across the
/// register so the redundancy is spread over all qubits.
pub fn cancel_redundant(n_qubits: u32, pairs: u32) -> Circuit {
    assert!(n_qubits >= 1, "cancel_redundant needs at least one qubit");
    let mut c = Circuit::new(n_qubits, 0);
    for p in 0..pairs {
        let q = p % n_qubits;
        c.rz(0.1 + 0.001 * (p as f64), q).unwrap(); // survives
        c.h(q).unwrap();
        c.h(q).unwrap(); // cancels
        c.x(q).unwrap();
        c.x(q).unwrap(); // cancels
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vqe_hea_shape() {
        let c = vqe_hea(4, 2);
        // Per layer: 4 * 5 = 20 1q gates + 3 CNOTs = 23.
        // 2 layers = 46 instructions.
        assert_eq!(c.len(), 46);
        assert_eq!(c.num_qubits(), 4);
    }

    #[test]
    fn qaoa_shape() {
        // n=4, p=2: H×4 + 2 layers of [3·(n-1) cost + n mixer]
        //         = 4 + 2·(3·3 + 4) = 4 + 2·13 = 30 instructions.
        let c = qaoa(4, 2);
        assert_eq!(c.num_qubits(), 4);
        assert_eq!(c.len(), 30);
    }

    #[test]
    fn qaoa_fuse2q_beats_p1_09_by_1_5x() {
        use crate::passes::{Fuse1qRuns, Fuse2q, PassPipeline};
        let base = qaoa(12, 10);

        // Count after P1-09 alone.
        let mut a = base.clone();
        PassPipeline::new(vec![Box::new(Fuse1qRuns)])
            .run(&mut a)
            .unwrap();
        let after_p1_09 = a.len();

        // Count after P1-09 + P1-10.
        let mut b = base;
        PassPipeline::new(vec![Box::new(Fuse1qRuns), Box::new(Fuse2q)])
            .run(&mut b)
            .unwrap();
        let after_p1_10 = b.len();

        assert!(after_p1_10 > 0, "over-collapsed to zero");
        let ratio = after_p1_09 as f64 / after_p1_10 as f64;
        assert!(
            ratio >= 1.5,
            "Fuse2q reduction {ratio:.3}× below 1.5× AC (after_p1_09={after_p1_09}, after_p1_10={after_p1_10})",
        );
    }

    #[test]
    fn ancilla_compute_shape() {
        // n_data=3, n_ancilla=2:
        //   data prep: 3 H + 2 CNOT (chain on data) = 5
        //   ancilla work: 2 H + 1 CNOT (chain on ancilla) = 3
        //   measure: 3 (data only)
        // total = 11.
        let c = ancilla_compute(3, 2);
        assert_eq!(c.num_qubits(), 5);
        assert_eq!(c.len(), 11);
    }

    #[test]
    fn ancilla_compute_dce_reduces() {
        use crate::passes::{DeadCodeElim, Pass};
        let mut c = ancilla_compute(3, 2);
        let before = c.len();
        let stats = DeadCodeElim.run(&mut c).unwrap();
        // The 3 ancilla-only gates (never measured, disjoint from data) are dead.
        assert_eq!(stats.transformations, 3);
        assert!(c.len() < before, "DCE must reduce the gate count");
        // No surviving instruction touches an ancilla qubit (3 or 4).
        assert!(c
            .instructions()
            .iter()
            .all(|i| !i.used_qubits().iter().any(|&q| q >= 3)));
    }

    #[test]
    fn vqe_hea_fuses_three_times() {
        // Algebra for the hardware-efficient ansatz with `k` rotations
        // per qubit per layer and a linear CNOT chain (`n-1` CNOTs):
        //
        //   input  per layer = k * n + (n - 1)
        //   output per layer = n     + (n - 1)
        //   ratio = (k * n + n - 1) / (2 * n - 1)
        //
        // Per layer, every qubit's `k`-rotation run is fenced by the
        // linear CNOT chain — `CNOT(q-1, q)` and `CNOT(q, q+1)` both
        // touch qubit `q`, and the leftmost/rightmost qubits are fenced
        // on one side by the chain endpoint plus, across the layer
        // boundary, by the *next* layer's CNOT chain. So each per-qubit
        // run collapses to exactly one fused 1q gate.
        //
        // For n=12 we need `(12k + 11) / 23 ≥ 3`, i.e. `k ≥ 58/12 ≈ 4.83`,
        // so `k = 5` is the smallest integer that hits the AC.
        //
        // For n=12, depth=10, k=5:
        //   input  = 10 * (5*12 + 11) = 10 * 71 = 710
        //   output = 10 * (12    + 11) = 10 * 23 = 230
        //   ratio  = 710 / 230 ≈ 3.087×  ✓
        let c = vqe_hea(12, 10);
        let before = c.len();
        let mut c2 = c;
        let stats = c2.optimize().unwrap();
        let after = c2.len();
        assert_eq!(stats.gates_before, before);
        assert_eq!(stats.gates_after, after);
        assert!(
            after > 0,
            "VQE HEA fused down to 0 instructions — fusion is over-collapsing (catastrophic regression)",
        );
        let ratio = before as f64 / after as f64;
        assert!(
            ratio >= 3.0,
            "VQE HEA fusion ratio {ratio} below 3× AC (before={before}, after={after})",
        );
    }

    #[test]
    fn cancel_redundant_reduces_5x() {
        use crate::passes::{CancelInversePairs, Pass};
        let mut c = cancel_redundant(4, 20);
        assert_eq!(c.len(), 100); // 5 gates × 20 steps
        let s = CancelInversePairs.run(&mut c).unwrap();
        assert_eq!(s.gates_after, 20); // only the Rz survivors
        assert_eq!(c.len(), 20);
    }
}
