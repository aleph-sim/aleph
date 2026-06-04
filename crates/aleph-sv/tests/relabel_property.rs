//! P2-09: property + thread-invariance tests for the relabel/tile pipeline.
//!
//! Guards three things:
//!
//! 1. `run_optimized` (full default pipeline: RelabelQubits + fusion +
//!    TileBlock + unpermute) matches raw `run` within 1e-12 on arbitrary
//!    small circuits.
//!
//! 2. The manual relabel(3)+tile(3)+unpermute composition matches raw `run`
//!    within 1e-12, directly stressing multi-tile and the unpermute gather.
//!
//! 3. The tiled executor is bit-identical across rayon thread counts (the
//!    parallel tile walk must not introduce FP non-determinism).

use aleph_backend::{run, run_optimized, Backend};
use aleph_ir::passes::{Pass, RelabelQubits, TileBlock};
use aleph_ir::{Circuit, Instruction};
use aleph_sv::{CpuState, NaiveSvBackend};
use aleph_test::circuit::arb_circuit_emittable;
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

/// Filter a generated circuit down to gate-only (drop Measure/Reset).
/// Barriers are kept so the fencing path is exercised on the optimized run.
/// Mirrors the helper in `fp32_equiv.rs::property`.
fn gate_only(src: &Circuit) -> Circuit {
    let mut out = Circuit::new(src.num_qubits(), src.num_clbits());
    for inst in src.instructions() {
        match inst {
            Instruction::Gate(g) => {
                out.add_gate(g.clone()).unwrap();
            }
            Instruction::Barrier(_) => {
                out.add_instruction(inst.clone()).unwrap();
            }
            _ => {}
        }
    }
    out
}

/// Assert two states agree elementwise within 1e-12.
fn states_close(a: &CpuState, b: &CpuState) -> bool {
    let xa = a.amplitudes();
    let xb = b.amplitudes();
    xa.len() == xb.len()
        && xa
            .iter()
            .zip(xb.iter())
            .all(|(x, y)| (x - y).norm() < 1e-12)
}

// ---------------------------------------------------------------------------
// Property tests.
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig { cases: 48, ..ProptestConfig::default() })]

    /// `run_optimized` (full default pipeline: RelabelQubits + fusion +
    /// TileBlock + unpermute) must reproduce raw `run` to within 1e-12.
    ///
    /// At the default tile_bits=15, small-n circuits produce a single tile so
    /// the tiled executor degenerates to a normal pass; this test guards the
    /// orchestration (pipeline wiring, permutation map-back) against regressions
    /// across many random circuits.
    ///
    /// Strategy: `arb_circuit_emittable(8, 2, 30)` — 8 qubits, 2 clbits,
    /// up to 30 ops, the same vocabulary as `fp32_equiv.rs::fp32_preserves_norm`.
    /// We strip Measure/Reset to avoid the stochastic-measure complication when
    /// comparing deterministic states.
    #[test]
    fn run_optimized_matches_raw(raw in arb_circuit_emittable(8, 2, 30)) {
        let c = gate_only(&raw);
        let reference = run(&mut NaiveSvBackend::with_seed(0), &c).unwrap();
        let optimized = run_optimized(&mut NaiveSvBackend::with_seed(0), &c).unwrap();
        prop_assert!(
            states_close(&reference, &optimized),
            "run_optimized diverged from raw run"
        );
    }

    /// Force relabel(3) + tile(3) + unpermute to stress the multi-tile path.
    ///
    /// At tile_bits=3 a 8-qubit circuit has 2^5 = 32 tiles so the multi-tile
    /// executor genuinely runs, and `RelabelQubits::new(3)` may permute high
    /// qubits to low positions so the relabelling + gather is exercised.
    ///
    /// This is the proptest analogue of `relabel_oracle.rs::relabel_tile_unpermute_transparent`.
    #[test]
    fn relabel_tile_small_width_matches_raw(raw in arb_circuit_emittable(8, 2, 30)) {
        let c = gate_only(&raw);
        let reference = run(&mut NaiveSvBackend::with_seed(0), &c).unwrap();

        // Manually compose relabel(3) + tile(3) + unpermute, mirroring what
        // `run_optimized` does internally when a permutation is present.
        let mut opt = c.clone();
        RelabelQubits::new(3).run(&mut opt).unwrap();
        TileBlock::new(3).run(&mut opt).unwrap();
        let perm = opt.qubit_permutation().map(|p| p.to_vec());
        let mut backend = NaiveSvBackend::with_seed(0);
        let mut state = run(&mut backend, &opt).unwrap();
        if let Some(ref p) = perm {
            backend.unpermute_state(&mut state, p).unwrap();
        }
        prop_assert!(
            states_close(&reference, &state),
            "relabel(3)+tile(3)+unpermute diverged from raw run"
        );
    }
}

// ---------------------------------------------------------------------------
// Thread-invariance test.
// ---------------------------------------------------------------------------

/// The tiled executor must produce bit-identical amplitudes regardless of
/// how many rayon threads are active during the run.
///
/// We build a deterministic n=10 circuit whose gate traffic is concentrated on
/// qubits 0, 1, 2 (and a few wider gates to increase entanglement). With
/// `TileBlock::new(3)` the low 3 bits define 2^3 = 8 confinable-qubit groups
/// and the high 7 bits define 2^7 = 128 tiles, so the multi-gate executor
/// is exercised in earnest.
///
/// `assert_eq!` on amplitude slices is exact (bit-for-bit equal). Any FP
/// reordering introduced by non-deterministic parallelism would fail here.
#[test]
fn tiled_pipeline_thread_invariant() {
    let n = 10u32;

    // Build a deterministic circuit with several gates on qubits 0, 1, 2
    // so that TileBlock(3) places them in a tile-confinable block.
    let mut c = Circuit::new(n, 0);
    // Several gates confined to the low-3 tile window — these will form
    // a TiledBlock and drive the multi-tile executor path.
    c.h(0).unwrap();
    c.h(1).unwrap();
    c.h(2).unwrap();
    c.cnot(0, 1).unwrap();
    c.rz(0.7, 1).unwrap();
    c.cnot(1, 2).unwrap();
    c.ry(1.3, 0).unwrap();
    c.cz(0, 2).unwrap();
    c.rx(-0.4, 2).unwrap();
    c.cnot(2, 0).unwrap();
    c.phase(0.9, 1).unwrap();
    // A wider gate to add entanglement beyond the tile window.
    c.cnot(0, 3).unwrap();
    c.cnot(3, 4).unwrap();
    c.h(5).unwrap();
    c.cnot(5, 6).unwrap();
    // More low-qubit traffic to bulk up the confinable block.
    c.h(0).unwrap();
    c.cnot(0, 1).unwrap();
    c.rz(0.3, 2).unwrap();
    c.cnot(1, 2).unwrap();
    c.h(2).unwrap();

    let mut opt = c.clone();
    TileBlock::new(3).run(&mut opt).unwrap();

    // Confirm that TileBlock actually produced a TiledBlock — if the circuit
    // has no confinable runs at tile_bits=3, the thread-invariance test would
    // be vacuous (it would still pass, but wouldn't exercise the tiled path).
    assert!(
        opt.instructions()
            .iter()
            .any(|i| matches!(i, Instruction::TiledBlock(_))),
        "TileBlock(3) produced no TiledBlock — circuit has no confinable run; \
         add more gates on qubits 0/1/2 to make the test non-vacuous"
    );

    let run_with_threads = |threads: usize| -> CpuState {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| run(&mut NaiveSvBackend::with_seed(0), &opt).unwrap())
    };

    let s1 = run_with_threads(1);
    let s8 = run_with_threads(8);

    let a1 = s1.amplitudes();
    let a8 = s8.amplitudes();
    assert_eq!(
        a1, a8,
        "tiled execution must be bit-identical across thread counts (1 vs 8)"
    );
}
