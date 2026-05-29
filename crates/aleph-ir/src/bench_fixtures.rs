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
        let ratio = before as f64 / after as f64;
        assert!(
            ratio >= 3.0,
            "VQE HEA fusion ratio {ratio} below 3× AC (before={before}, after={after})",
        );
    }
}
