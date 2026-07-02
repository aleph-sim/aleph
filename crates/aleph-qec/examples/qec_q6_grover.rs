//! Q6-28 (small logical algorithm) — emit 3-qubit **Grover search** trials whose 28 T-gate magic-state
//! measurements are each resolved by the on-board decoder, so a full non-Clifford *algorithm* (not just
//! one gate) runs end-to-end with the silicon decoder in the loop and is verified by its output
//! distribution.
//!
//! Grover on N=8 with one marked state runs H^⊗3 then 2 iterations of {oracle, diffusion}; the optimal
//! 2 iterations peak the marked-state probability at ~94.5%. Both the oracle and the diffusion contain a
//! CCZ, and CCZ = H·CCX·H where CCX (Toffoli) decomposes into 7 T/T† gates (Q6-27). So each iteration is
//! 2 CCZ = 14 T's, and the whole algorithm is **28 T-gate magic-state injections**, each a code-protected
//! logical measurement (raw = m ⊕ e) DECODED on the real Arty. X/H/CNOT are Clifford (no decode). A wrong
//! decode inserts an extra S mid-algorithm, corrupting the amplitude amplification — so the board
//! measures the marked-state probability with the decoder ON (corrected) vs OFF (raw).
//!
//! Per trial we emit the marked state (0..7) and 28 independent memory-Z blocks (one per T-gate) with
//! their true logical flip e; the board decodes each on the Arty and drives the conditional S in a
//! 3-qubit state-vector Grover gadget (`uf_qubit_grover.py`).
//!
//! Usage:
//!   cargo run --release -p aleph-qec --example qec_q6_grover -- [d] [W] [C] [rounds] [trials] [seed] [p]
//!   # defaults: d=3 W=9 C=3 rounds=17 trials=1024 seed=2024 p=0.003  (28 T-gates/trial fixed)
//!   cargo run --release -p aleph-qec --example qec_q6_grover -- 3 9 3 17 1024 2024 0.003 > hw/cosim_grover_d3.vec
//!
//! Output (stdout):
//!   # comment metadata (d/W/C/dpr/slices/trials/gates=28/...)
//!   P p=<p> trials=<N>
//!   T <marked 0..7> <e_1> ... <e_28>   ← one trial: marked state + the 28 T-gate blocks' true flips
//!   <dpr bits> × slices                ← block 1 (T-gate 1 magic-measurement decode), round 0 first
//!   ...                                ← 28 blocks total, in T-gate order

use aleph_qec::{build_dem, sample_shots, SurfaceCode};

const T_GATES: usize = 28; // 2 iterations x 2 CCZ x 7 T per Toffoli

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let d: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let w: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(9);
    let c: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let rounds: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(17);
    let trials: u64 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(1024);
    let seed: u64 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(2024);
    let p: f64 = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(0.003);

    let exp = SurfaceCode::new(d).memory_z_experiment(rounds);
    let det_round = exp.detector_rounds();
    let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();
    assert_eq!(
        dem.observables, 1,
        "Grover T-gate decode assumes a single logical observable per block"
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

    let (syndromes, truths) = sample_shots(&dem, T_GATES as u64 * trials, seed);
    let flip = |i: usize| truths[i].first().copied().unwrap_or(false);

    println!("# Q6-28 small logical algorithm (3-qubit Grover) trials — GENERATED, do not edit.");
    println!(
        "# d={d} W={w} C={c} dpr={dpr} slices={n_slices} detectors={} observables=1 noise=phenom trials={trials} gates={T_GATES} seed={seed}",
        dem.detectors
    );
    println!(
        "# regenerate: cargo run --release -p aleph-qec --example qec_q6_grover -- {d} {w} {c} {rounds} {trials} {seed} {p}"
    );
    println!("P p={p} trials={trials}");

    let mut line = String::with_capacity(dpr);
    for t in 0..trials as usize {
        let base = T_GATES * t;
        let marked = t % 8; // sweep all 8 possible marked states
        let es: Vec<String> = (0..T_GATES)
            .map(|g| u8::from(flip(base + g)).to_string())
            .collect();
        println!("T {marked} {}", es.join(" "));
        for g in 0..T_GATES {
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
