//! Q6-32 Milestone A (genuine large algorithm / modular arithmetic) — emit n-bit quantum ripple-carry
//! adder `b := a + b` trials whose T-gate magic-state measurements are each resolved by the on-board
//! decoder. Unlike Q6-30's synthetic C^kX ladder, the T-count here is INTRINSIC to a REAL algorithm: the
//! Cuccaro ripple-carry adder (arXiv:quant-ph/0410184), the arithmetic core of Shor's algorithm.
//!
//! The n-bit adder uses n MAJ + n UMA gadgets = 2n Toffolis; each Toffoli = 7 T/T† (Q6-27). So the
//! algorithm is 14n T-gate magic-state injections, each code-protected (raw = m ⊕ e) and DECODED on the
//! real Arty (one memory-Z block each, Q6-20 bitstream unchanged). X/CNOT are Clifford (no decode).
//! Sweeping n = 2,3,4 gives a T-count ladder 28, 42, 56 — the SAME counts as Q6-30's k=3,4,5, but now
//! carried by a genuine arithmetic circuit rather than a control-count knob.
//!
//! Per trial we emit the (a,b) input register pair and 14n independent memory-Z blocks (one per T-gate)
//! with their true logical flip e; the board decodes each on the Arty, drives the conditional S in a
//! (2n+2)-qubit state-vector adder gadget (`uf_qubit_adder.py`), and verifies the sum register equals
//! (a+b): the low n bits in b, the carry-out in z, and a restored — decoder ON vs OFF.
//!
//! Usage:
//!   cargo run --release -p aleph-qec --example qec_q6_adder -- [n] [d] [W] [C] [rounds] [trials] [seed] [p]
//!   # defaults: n=2 d=3 W=9 C=3 rounds=17 trials=800 seed=2024 p=0.002
//!   cargo run --release -p aleph-qec --example qec_q6_adder -- 2 3 9 3 17 800 2024 0.002 > hw/cosim_adder_n2.vec
//!
//! Output (stdout):
//!   # comment metadata (n/nbits=2n+2/gates=14n/d/W/C/dpr/slices/...)
//!   P p=<p> trials=<N>
//!   T <a 0..2^n-1> <b 0..2^n-1> <e_1> ... <e_gates>   ← one trial: input pair + the T-gate blocks' flips
//!   <dpr bits> × slices                                ← block 1, round 0 first; `gates` blocks in order

use aleph_qec::{build_dem, sample_shots, SurfaceCode};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(2);
    let d: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let w: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(9);
    let c: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(3);
    let rounds: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(17);
    let trials: u64 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(800);
    let seed: u64 = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(2024);
    let p: f64 = a.get(8).and_then(|s| s.parse().ok()).unwrap_or(0.002);
    assert!(n >= 1, "need at least a 1-bit adder");
    let gates = 14 * n; // 2n Toffolis x 7 T

    let exp = SurfaceCode::new(d).memory_z_experiment(rounds);
    let det_round = exp.detector_rounds();
    let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();
    assert_eq!(
        dem.observables, 1,
        "adder T-gate decode assumes a single logical observable per block"
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

    println!("# Q6-32 ripple-carry adder (Cuccaro b:=a+b) trials — GENERATED, do not edit.");
    println!(
        "# n={n} nbits={} gates={gates} d={d} W={w} C={c} dpr={dpr} slices={n_slices} detectors={} observables=1 noise=phenom trials={trials} seed={seed}",
        2 * n + 2,
        dem.detectors
    );
    println!(
        "# regenerate: cargo run --release -p aleph-qec --example qec_q6_adder -- {n} {d} {w} {c} {rounds} {trials} {seed} {p}"
    );
    println!("P p={p} trials={trials}");

    let n_inputs = 1usize << (2 * n); // sweep all (a,b) input pairs
    let mask = (1usize << n) - 1;
    let mut line = String::with_capacity(dpr);
    for t in 0..trials as usize {
        let base = gates * t;
        let inp = t % n_inputs;
        let a_in = (inp >> n) & mask;
        let b_in = inp & mask;
        let es: Vec<String> = (0..gates)
            .map(|g| u8::from(flip(base + g)).to_string())
            .collect();
        println!("T {a_in} {b_in} {}", es.join(" "));
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
