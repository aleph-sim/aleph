//! P2-09: the tile-major driver (TileBlock pass + NaiveSvBackend executor)
//! must preserve the state exactly (AC #3, 1e-12) vs the raw reference,
//! both via the default pipeline (TileBlock(15) → degenerate single tile at
//! small n) and a forced small tile width (the real multi-tile path). No
//! relabelling here — RelabelQubits arrives in a later task.
//!
//! `run` (raw, gate-by-gate, no optimization) is the trusted oracle. We assert
//! that:
//!   1. the full default pipeline (cancel/dce/fuse_* + TileBlock(15)) and
//!   2. a forced `TileBlock::new(4)` (splits the state into 2^(n-4) tiles so
//!      the multi-tile executor genuinely runs on the grouped gate runs)
//!
//! both reproduce the reference amplitudes to within 1e-12.

use aleph_backend::run;
use aleph_ir::passes::{Pass, TileBlock};
use aleph_ir::{Circuit, Instruction};
use aleph_sv::{CpuState, NaiveSvBackend};

// ---------------------------------------------------------------------------
// Tier-1 circuit builders (copied from `fp32_equiv.rs`; same public-API idioms).
// ---------------------------------------------------------------------------

/// GHZ-n: `H q0; CX q0,q1; …`. Exercises H + a CNOT chain.
fn ghz(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.h(0).unwrap();
    for i in 0..n - 1 {
        c.cnot(i, i + 1).unwrap();
    }
    c
}

/// Controlled-phase via the textbook CNOT+Rz decomposition.
fn cphase(c: &mut Circuit, lambda: f64, control: u32, target: u32) {
    c.rz(lambda / 2.0, target).unwrap();
    c.cnot(control, target).unwrap();
    c.rz(-lambda / 2.0, target).unwrap();
    c.cnot(control, target).unwrap();
    c.rz(lambda / 2.0, control).unwrap();
}

/// QFT-n over a seeded non-trivial basis state. Exercises H, Rz, CNOT, Swap.
fn qft(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in (0..n).step_by(2) {
        c.x(q).unwrap();
    }
    for j in 0..n {
        c.h(j).unwrap();
        for k in (j + 1)..n {
            let lambda = std::f64::consts::PI / (1u64 << (k - j)) as f64;
            cphase(&mut c, lambda, k, j);
        }
    }
    let mut a = 0;
    let mut b = n - 1;
    while a < b {
        c.swap(a, b).unwrap();
        a += 1;
        b -= 1;
    }
    c
}

/// Deterministic "random-brickwall" circuit: alternating 1q-rotation layers and
/// CNOT/CZ entangler brickwall. Exercises generic 1q, diagonal-1q, and both 2q
/// kernels. The low-qubit pairs here also yield genuine multi-gate blocks at
/// tile_bits=4.
fn random_brickwall(n: u32, depth: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    let mut state: u64 = 0x9E3779B97F4A7C15u64
        .wrapping_add(n as u64)
        .wrapping_mul(depth as u64 + 1);
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for layer in 0..depth {
        for q in 0..n {
            let r = next();
            let angle =
                ((next() % 1000) as f64 / 1000.0) * std::f64::consts::TAU - std::f64::consts::PI;
            match r % 4 {
                0 => c.rx(angle, q).unwrap(),
                1 => c.ry(angle, q).unwrap(),
                2 => c.rz(angle, q).unwrap(),
                _ => c.phase(angle, q).unwrap(),
            };
        }
        let start = layer % 2;
        let mut q = start;
        while q + 1 < n {
            if next() % 2 == 0 {
                c.cnot(q, q + 1).unwrap();
            } else {
                c.cz(q, q + 1).unwrap();
            }
            q += 2;
        }
    }
    c
}

// ---------------------------------------------------------------------------
// Oracle.
// ---------------------------------------------------------------------------

/// Assert two states agree elementwise within the 1e-12 spec bound.
fn assert_close(a: &CpuState, b: &CpuState, ctx: &str) {
    let xa = a.amplitudes();
    let xb = b.amplitudes();
    assert_eq!(
        xa.len(),
        xb.len(),
        "{ctx}: len {} vs {}",
        xa.len(),
        xb.len()
    );
    let mut max_err = 0.0f64;
    let mut worst = 0usize;
    for (i, (x, y)) in xa.iter().zip(xb.iter()).enumerate() {
        let e = (x - y).norm();
        if e > max_err {
            max_err = e;
            worst = i;
        }
    }
    assert!(
        max_err < 1e-12,
        "{ctx}: max abs err {max_err:e} >= 1e-12 at amp[{worst}] ({:?} vs {:?})",
        xa[worst],
        xb[worst]
    );
}

#[test]
fn tiled_driver_matches_raw_reference() {
    // Tracks whether the forced tile_bits=4 path ever produced a real
    // multi-gate block; the random-brickwall fixture has dense low-qubit gates
    // so the multi-tile executor must run for at least one fixture/n.
    let mut saw_tiled_block = false;

    for n in [6u32, 8, 10, 12] {
        let cases: Vec<(String, Circuit)> = vec![
            (format!("ghz_n{n}"), ghz(n)),
            (format!("qft_n{n}"), qft(n)),
            (
                format!("random_n{n}_d{}", 2 * n),
                random_brickwall(n, 2 * n),
            ),
        ];
        for (name, c) in cases {
            // Trusted reference: raw, un-optimized, gate-by-gate.
            let reference = run(&mut NaiveSvBackend::with_seed(0), &c).unwrap();

            // Default pipeline: cancel/dce/fuse_* + TileBlock(15) (single tile at small n).
            let mut opt = c.clone();
            opt.optimize().unwrap();
            let tiled = run(&mut NaiveSvBackend::with_seed(0), &opt).unwrap();
            assert_close(&reference, &tiled, &format!("{name} default-pipeline"));

            // Forced multi-tile: tile_bits=4 ⇒ 2^(n-4) tiles for grouped runs.
            let mut small = c.clone();
            TileBlock::new(4).run(&mut small).unwrap();
            saw_tiled_block |= small
                .instructions()
                .iter()
                .any(|i| matches!(i, Instruction::TiledBlock(_)));
            let tiled4 = run(&mut NaiveSvBackend::with_seed(0), &small).unwrap();
            assert_close(&reference, &tiled4, &format!("{name} tile_bits=4"));
        }
    }

    assert!(
        saw_tiled_block,
        "tile_bits=4 produced no TiledBlock for any fixture — \
         the multi-tile executor path was never exercised"
    );
}
