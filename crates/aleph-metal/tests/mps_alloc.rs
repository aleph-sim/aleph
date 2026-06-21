//! P5.8-02: device-buffer pool — steady-state allocation probe.
//!
//! The Phase 5.7 audit (`docs/perf/phase5.7-audit.md`) flagged that the per-gate
//! path allocated a fresh `MTLBuffer` for Θ, the Jacobi A/V/σ factors, and every
//! rebuilt site tensor — `~6 + 2·(centre steps)` `new_buffer*` calls per 2q gate,
//! versus the CPU MPS's zero-allocation `Scratch` arena. P5.8-02 pools those buffers
//! (capacity-aware [`DeviceBuffer`], in-place site rebuilds), so once bond dimensions
//! reach their high-water mark, additional gates of the same shape allocate nothing.
//!
//! This test drives the bond to the cap, then asserts the device-allocation counter
//! (`aleph_metal::device_alloc_count`) is flat across many further gates — the
//! `≈0 steady-state per-gate allocation` acceptance criterion.
//!
//! Run: `cargo test -p aleph-metal --features metal --test mps_alloc`

#![cfg(all(target_os = "macos", feature = "metal"))]

use aleph_backend::Backend;
use aleph_benches::brickwall_ry_cnot_rz;
use aleph_core::{Gate, GateInstance};
use aleph_ir::Instruction;
use aleph_metal::{device_alloc_count, MetalMpsBackend};

#[test]
fn steady_state_per_gate_device_allocs_are_zero() {
    let Ok(mut be) = MetalMpsBackend::with_max_bond(8) else {
        eprintln!("skipping alloc probe: no Metal device");
        return;
    };
    let n = 6u32;
    let mut s = be.allocate(n).expect("allocate");

    // Saturate every central bond to the cap (χ=8) so the pooled Θ / Jacobi / site
    // buffers all reach their high-water mark. After this, no later gate of equal or
    // smaller shape can force a growth realloc.
    let prep = brickwall_ry_cnot_rz(n, 10);
    for inst in prep.instructions() {
        if let Instruction::Gate(g) = inst {
            be.apply_gate(&mut s, g).expect("prep gate");
        }
    }

    // The probe gate: a NN 2q gate on a saturated interior bond. One settling apply
    // before the measurement window so the orthogonality centre and every pool slot
    // are at the exact shapes the measured repeats will reuse.
    let probe = GateInstance::new(Gate::Cnot, vec![2, 3]);
    be.apply_gate(&mut s, &probe).expect("settle gate");

    let before = device_alloc_count();
    let reps = 40u64;
    for _ in 0..reps {
        be.apply_gate(&mut s, &probe).expect("probe gate");
    }
    let delta = device_alloc_count() - before;
    eprintln!("steady-state device allocations: {delta} over {reps} gates");

    // Steady state: the pool must not allocate per gate. Allow a tiny slack (the
    // counter is process-global, so a stray allocation from elsewhere shouldn't fail
    // the test), but it must be far below 1-per-gate.
    assert!(
        delta <= 2,
        "steady-state device allocations grew by {delta} over {reps} identical gates \
         ({:.3}/gate); the P5.8-02 pool should hold this at ≈0",
        delta as f64 / reps as f64
    );
}
