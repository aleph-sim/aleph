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

/// Inverse QFT on `n` qubits: `qft_circuit(n)`'s instructions in reverse order,
/// each gate replaced by its inverse (`H→H`, `Phase(θ)→Phase(−θ)`), preserving
/// the control/target qubits. `qft_circuit(n)` followed by `qft_inverse_circuit(n)`
/// is the identity.
#[must_use]
pub fn qft_inverse_circuit(n: u32) -> aleph_ir::Circuit {
    let fwd = qft_circuit(n);
    let mut inv = aleph_ir::Circuit::new(n, 0);
    for inst in fwd.instructions().iter().rev() {
        match inst {
            aleph_ir::Instruction::Gate(g) => {
                let mut g2 = g.clone(); // preserves qubits AND controls
                g2.gate = g.gate.inverse();
                inv.add_instruction(aleph_ir::Instruction::Gate(g2))
                    .unwrap();
            }
            other => {
                inv.add_instruction(other.clone()).unwrap();
            }
        }
    }
    inv
}

/// A small non-trivial state prep: H on every qubit, then a phase rotation on
/// each, producing a generic (non-basis) state for round-trip testing.
#[cfg(test)]
fn generic_prep_circuit(n: u32) -> aleph_ir::Circuit {
    use aleph_core::Param;
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(GateInstance::new(Gate::H, vec![q])).unwrap();
    }
    for q in 0..n {
        c.add_gate(GateInstance::new(
            Gate::Rz(Param::Concrete(0.1 * (q as f64 + 1.0))),
            vec![q],
        ))
        .unwrap();
    }
    c
}

/// A low-qubit-heavy circuit: `depth` layers of single-qubit Rz+Rx
/// rotations and nearest-neighbour CNOTs confined to the lowest `width`
/// qubits, on an `n`-qubit register (the high qubits stay idle).
///
/// This is the regime the P2-09 tile-major executor targets: most gates
/// are tile-confinable (`TileBlock` default `tile_bits = 15` ≥ `width`),
/// so the tile executor collapses many DRAM passes into one.  Angles are
/// deterministic (function of `(layer, q)`) so no rand dependency.
///
/// # Panics
/// Panics if `width < 2` or `width > n`.
#[must_use]
pub fn low_qubit_heavy_circuit(n: u32, width: u32, depth: usize) -> Circuit {
    assert!(width >= 2 && width <= n, "width must be in [2, n]");
    let mut c = Circuit::new(n, 0);
    for layer in 0..depth {
        // 1q rotations on every active qubit — same angle idiom as
        // random_brickwall_circuit so the builder stays consistent.
        for q in 0..width {
            let theta = ((layer as f64 + 1.0) * 0.123 + q as f64 * 0.071) % std::f64::consts::TAU;
            let _ = c.rz(theta, q);
            let _ = c.rx(theta * 1.13, q);
        }
        // Nearest-neighbour CNOT layer inside the active window.
        // Even layers: (0,1),(2,3),…; odd layers: (1,2),(3,4),…
        let offset = (layer & 1) as u32;
        let mut q = offset;
        while q + 1 < width {
            let _ = c.cnot(q, q + 1);
            q += 2;
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

#[cfg(test)]
mod qft_roundtrip_tests {
    use super::*;
    use aleph_backend::run;
    use aleph_core::Complex;
    use aleph_sv::NaiveSvBackend;

    /// Apply `circuit` to a freshly allocated |0…0⟩ and return amplitudes.
    fn run_amps(circuit: &aleph_ir::Circuit) -> Vec<Complex> {
        let mut b = NaiveSvBackend::with_seed(7);
        let state = run(&mut b, circuit).expect("run");
        // `HasAmplitudes`/state amplitudes accessor — match the crate's API.
        state.amplitudes().to_vec()
    }

    #[test]
    fn qft_then_inverse_is_identity_on_zero_state() {
        let n = 6;
        let mut c = qft_circuit(n);
        // Append the inverse so the combined circuit should be the identity.
        for inst in qft_inverse_circuit(n).instructions() {
            c.add_instruction(inst.clone()).unwrap();
        }
        let amps = run_amps(&c);
        assert!((amps[0].re - 1.0).abs() < 1e-10, "amp[0] should be 1");
        assert!(amps[0].im.abs() < 1e-10);
        for (k, a) in amps.iter().enumerate().skip(1) {
            assert!(a.norm() < 1e-10, "amp[{k}] should be ~0");
        }
    }

    #[test]
    fn qft_then_inverse_is_identity_on_generic_state() {
        // Per the P1-13 lesson, a |0…0⟩-only check misses bugs. Prep a generic
        // state with a layer of H + T-like rotations, snapshot it, then apply
        // QFT∘QFT⁻¹ and assert the state is unchanged.
        let n = 5;
        let prep = generic_prep_circuit(n); // defined below
        let before = run_amps(&prep);

        let mut c = prep.clone();
        for inst in qft_circuit(n).instructions() {
            c.add_instruction(inst.clone()).unwrap();
        }
        for inst in qft_inverse_circuit(n).instructions() {
            c.add_instruction(inst.clone()).unwrap();
        }
        let after = run_amps(&c);

        assert_eq!(before.len(), after.len());
        for (k, (x, y)) in before.iter().zip(after.iter()).enumerate() {
            assert!((x - y).norm() < 1e-10, "amp[{k}] changed: {x:?} vs {y:?}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_ir::Instruction;

    /// Sanity: the low-qubit-heavy circuit (width=6 < tile_bits=15)
    /// must produce at least one `TiledBlock` after `optimize()`, so the
    /// cache-blocking bench actually exercises the tile-major executor.
    #[test]
    fn low_qubit_heavy_optimize_produces_tiled_block() {
        let mut c = low_qubit_heavy_circuit(12, 6, 10);
        c.optimize().expect("optimize should not fail");
        let has_tiled = c
            .instructions()
            .iter()
            .any(|i| matches!(i, Instruction::TiledBlock(_)));
        assert!(
            has_tiled,
            "optimize() on low_qubit_heavy_circuit(12,6,10) must produce \
             at least one TiledBlock (width=6 < tile_bits=15 means all \
             active-window gates are tile-confinable)"
        );
    }
}
