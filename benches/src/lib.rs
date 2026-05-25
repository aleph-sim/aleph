//! `aleph-benches`: workspace-level benchmark harness.
//!
//! The actual benchmarks live in `benches/*.rs` and are run via
//! `cargo bench --bench <name>` or `cargo bench --workspace`.  This
//! `lib.rs` exposes shared circuit-builders the individual bench
//! files invoke through `NaiveSvBackend` via `aleph_backend::run`.
//!
//! See `docs/benchmarking.md` at the repo root for the benchmark
//! policy (when to add, how to interpret results, how bencher.dev
//! tracks history). The link is intentionally not a `cargo doc`
//! intra-doc link because the file lives outside the crate.

use aleph_core::{Gate, GateInstance};
use aleph_ir::Circuit;
use smallvec::smallvec;

/// Bell pair on 2 qubits: `H q[0]; CX q[0], q[1]` → `(|00⟩ + |11⟩)/√2`.
#[must_use]
pub fn bell_circuit() -> Circuit {
    let mut c = Circuit::new(2, 0);
    let _ = c.h(0);
    let _ = c.cnot(0, 1);
    c
}

/// GHZ state on `n` qubits: `H q[0]; CX q[0],q[1]; CX q[1],q[2]; …`
/// → `(|0…0⟩ + |1…1⟩)/√2`.
#[must_use]
pub fn ghz_circuit(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    let _ = c.h(0);
    for t in 1..n {
        let _ = c.cnot(t - 1, t);
    }
    c
}

/// Textbook QFT on `n` qubits per Nielsen & Chuang § 5.1: per-qubit
/// `H` followed by a descending ladder of controlled-`Phase` gates.
/// (Closing SWAPs that reverse the qubit order are omitted — they
/// don't affect bench-relevant gate-application cost.)
#[must_use]
pub fn qft_circuit(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for j in 0..n {
        let _ = c.h(j);
        for k in (j + 1)..n {
            // Controlled-Phase(π / 2^(k-j)) with `k` as control, `j`
            // as target.  No builder shortcut for cphase yet —
            // construct via `GateInstance::controlled(Phase, ...)`.
            // Phase is diagonal so the controlled form commutes with
            // its qubit ordering, but we match the textbook (control
            // = higher-index qubit).
            let theta = std::f64::consts::PI / (1u64 << (k - j)) as f64;
            let _ = c.add_gate(GateInstance::controlled(
                Gate::Phase(theta.into()),
                smallvec![j],
                smallvec![k],
            ));
        }
    }
    c
}

/// Brick-wall random-circuit-shaped workload, `depth` layers of
/// alternating-pair CNOTs interleaved with random 1q rotations.  Not
/// a real Sycamore-style random circuit (no Haar-random SU(4)
/// blocks), but the bandwidth shape and gate count match what a
/// state-vector backend pays per layer.
///
/// The rotation angles are deterministic (function of `(layer, q)`)
/// so the bench is reproducible without bringing rand into the
/// dep tree.
#[must_use]
pub fn random_brickwall_circuit(n: u32, depth: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for layer in 0..depth {
        // 1q rotation on every qubit — fills the time-axis with
        // single-qubit work.
        for q in 0..n {
            let theta = ((layer as f64) + (q as f64) * 0.37).cos();
            let _ = c.rz(theta, q);
            let _ = c.rx(theta * 1.13, q);
        }
        // CNOT layer: even layers pair (0,1),(2,3),…; odd layers
        // offset by 1 to pair (1,2),(3,4),…  Standard brick-wall.
        let offset = (layer & 1) as u32;
        let mut q = offset;
        while q + 1 < n {
            let _ = c.cnot(q, q + 1);
            q += 2;
        }
    }
    c
}
