//! Q6-27 (multi-qubit non-Clifford algorithm) — emit logical **Toffoli (CCX)** trials whose 7 T-gate
//! magic-state measurements are each resolved by the on-board decoder, so a real 3-qubit non-Clifford
//! algorithm runs end-to-end with the silicon decoder in the loop.
//!
//! Q6-26 was a single-qubit T^8 chain. Toffoli is the canonical *multi-qubit* non-Clifford gate: its
//! standard decomposition (Nielsen & Chuang §4.3) is 7 T/T† gates + 6 CNOTs + 2 H. Each T is applied by
//! gate teleportation, whose magic-ancilla Z-measurement is code-protected (raw = m ⊕ e) and DECODED on
//! the real Arty. A wrong decode inserts an extra S mid-circuit → the Toffoli output is corrupted (a
//! genuine non-Clifford error, since S before the decomposition's H gates becomes a superposition), so
//! the board verifies the *classical truth table* (target flips iff both controls are 1) with the
//! decoder ON (corrected) vs OFF (raw). CNOT/H are Clifford and need no decode (transversal in an FT
//! machine); only the 7 non-Clifford T's consume a decoded magic measurement.
//!
//! Per trial we emit the 3-bit computational-basis input (0..7) and 7 independent memory-Z blocks (one
//! per T-gate) with their true logical flip e; the board decodes each on the Arty and drives the
//! conditional S in a 3-qubit state-vector Toffoli gadget (`uf_qubit_toffoli.py`).
//!
//! Usage:
//!   cargo run --release -p aleph-qec --example qec_q6_toffoli -- [d] [W] [C] [rounds] [trials] [seed] [p]
//!   # defaults: d=3 W=9 C=3 rounds=17 trials=2400 seed=2024 p=0.005  (7 T-gates/trial fixed)
//!   cargo run --release -p aleph-qec --example qec_q6_toffoli -- 3 9 3 17 2400 2024 0.005 > hw/cosim_toffoli_d3.vec
//!
//! Output (stdout):
//!   # comment metadata (d/W/C/dpr/slices/trials/gates=7/...)
//!   P p=<p> trials=<N>
//!   T <input 0..7> <e_1> ... <e_7>     ← one trial: basis input + the 7 T-gate blocks' true flips
//!   <dpr bits> × slices                ← block 1 (T-gate 1 magic-measurement decode), round 0 first
//!   ...                                ← 7 blocks total, in T-gate order

use aleph_qec::{build_dem, sample_shots, SurfaceCode};

const T_GATES: usize = 7; // Toffoli standard decomposition has 7 T/T† gates

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let d: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let w: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(9);
    let c: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let rounds: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(17);
    let trials: u64 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(2400);
    let seed: u64 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(2024);
    let p: f64 = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(0.005);

    let exp = SurfaceCode::new(d).memory_z_experiment(rounds);
    let det_round = exp.detector_rounds();
    let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();
    assert_eq!(
        dem.observables, 1,
        "Toffoli T-gate decode assumes a single logical observable per block"
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

    println!("# Q6-27 multi-qubit non-Clifford (logical Toffoli) trials — GENERATED, do not edit.");
    println!(
        "# d={d} W={w} C={c} dpr={dpr} slices={n_slices} detectors={} observables=1 noise=phenom trials={trials} gates={T_GATES} seed={seed}",
        dem.detectors
    );
    println!(
        "# regenerate: cargo run --release -p aleph-qec --example qec_q6_toffoli -- {d} {w} {c} {rounds} {trials} {seed} {p}"
    );
    println!("P p={p} trials={trials}");

    let mut line = String::with_capacity(dpr);
    for t in 0..trials as usize {
        let base = T_GATES * t;
        let input = t % 8; // sweep all 8 computational-basis inputs
        let es: Vec<String> = (0..T_GATES)
            .map(|g| u8::from(flip(base + g)).to_string())
            .collect();
        println!("T {input} {}", es.join(" "));
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
