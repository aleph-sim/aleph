//! Q6-30 (larger algorithm / T-count scaling) — emit multi-controlled-X (C^kX) trials whose T-gate
//! magic-state measurements are each resolved by the on-board decoder, swept over the control count k so
//! the T-gate count scales and the fidelity(T) curve shows the (1−LER) compounding: more non-Clifford
//! gates → sharper dependence on decoder quality.
//!
//! C^kX (the multi-controlled-X at the heart of Grover oracles and reversible arithmetic) is built from
//! a compute/uncompute cascade of 2(k−1) Toffolis on (k−1) ancillas; each Toffoli = 7 T/T† gates
//! (Q6-27). So the algorithm is 14(k−1) T-gate magic-state injections, each code-protected (raw = m ⊕ e)
//! and DECODED on the real Arty (one memory-Z block each, Q6-20 bitstream unchanged). X/CNOT are Clifford
//! (no decode). Sweeping k = 2,3,4,5 gives a clean T-count ladder 14, 28, 42, 56.
//!
//! Per trial we emit the k-bit control input and 14(k−1) independent memory-Z blocks (one per T-gate)
//! with their true logical flip e; the board decodes each on the Arty, drives the conditional S in an
//! n-qubit state-vector C^kX gadget (`uf_qubit_mcx.py`), and verifies the truth table (target flips iff
//! all k controls are 1).
//!
//! Usage:
//!   cargo run --release -p aleph-qec --example qec_q6_mcx -- [k] [d] [W] [C] [rounds] [trials] [seed] [p]
//!   # defaults: k=3 d=3 W=9 C=3 rounds=17 trials=800 seed=2024 p=0.002
//!   cargo run --release -p aleph-qec --example qec_q6_mcx -- 3 3 9 3 17 800 2024 0.002 > hw/cosim_mcx_k3.vec
//!
//! Output (stdout):
//!   # comment metadata (k/nbits/gates=14(k-1)/d/W/C/dpr/slices/...)
//!   P p=<p> trials=<N>
//!   T <control 0..2^k-1> <e_1> ... <e_gates>   ← one trial: control input + the T-gate blocks' true flips
//!   <dpr bits> × slices                        ← block 1, round 0 first; `gates` blocks total in order

use aleph_qec::{build_dem, sample_shots, SurfaceCode};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let k: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let d: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let w: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(9);
    let c: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(3);
    let rounds: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(17);
    let trials: u64 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(800);
    let seed: u64 = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(2024);
    let p: f64 = a.get(8).and_then(|s| s.parse().ok()).unwrap_or(0.002);
    assert!(k >= 2, "need at least 2 controls");
    let gates = 14 * (k - 1); // 2(k-1) Toffolis x 7 T

    let exp = SurfaceCode::new(d).memory_z_experiment(rounds);
    let det_round = exp.detector_rounds();
    let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();
    assert_eq!(
        dem.observables, 1,
        "C^kX T-gate decode assumes a single logical observable per block"
    );

    let n_slices = det_round.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    let mut by_round: Vec<Vec<usize>> = vec![Vec::new(); n_slices];
    for (dd, &r) in det_round.iter().enumerate() {
        by_round[r].push(dd);
    }
    let dpr = by_round[0].len();
    assert!(
        by_round.iter().all(|r| r.len() == dpr),
        "expected a fixed detectors-per-round for the streaming frame"
    );
    assert!(
        n_slices >= w && (n_slices - w).is_multiple_of(c),
        "slices ({n_slices}) must be W + k*C (W={w}, C={c}); pick rounds = W + k*C - 1"
    );

    let (syndromes, truths) = sample_shots(&dem, gates as u64 * trials, seed);
    let flip = |i: usize| truths[i].first().copied().unwrap_or(false);

    println!(
        "# Q6-30 T-count scaling (multi-controlled-X C^{k}X) trials — GENERATED, do not edit."
    );
    println!(
        "# k={k} nbits={} gates={gates} d={d} W={w} C={c} dpr={dpr} slices={n_slices} detectors={} observables=1 noise=phenom trials={trials} seed={seed}",
        2 * k,
        dem.detectors
    );
    println!(
        "# regenerate: cargo run --release -p aleph-qec --example qec_q6_mcx -- {k} {d} {w} {c} {rounds} {trials} {seed} {p}"
    );
    println!("P p={p} trials={trials}");

    let n_inputs = 1usize << k;
    let mut line = String::with_capacity(dpr);
    for t in 0..trials as usize {
        let base = gates * t;
        let control = t % n_inputs; // sweep all 2^k control inputs
        let es: Vec<String> = (0..gates)
            .map(|g| u8::from(flip(base + g)).to_string())
            .collect();
        println!("T {control} {}", es.join(" "));
        for g in 0..gates {
            let syn = &syndromes[base + g];
            for round in &by_round {
                line.clear();
                for &dd in round {
                    line.push(if syn.is_fired(dd as u32) { '1' } else { '0' });
                }
                println!("{line}");
            }
        }
    }
}
