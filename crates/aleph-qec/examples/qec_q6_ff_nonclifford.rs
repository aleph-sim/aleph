//! Q6-26 (non-Clifford feed-forward) — emit T-gate-teleportation trials whose magic-state measurement
//! outcomes are each resolved by the on-board decoder, so the on-silicon decode result conditions a
//! non-Clifford computation. This is the genuinely non-reducible feed-forward the Q6-25 note flagged as
//! the real next frontier.
//!
//! Q6-25 teleported *stabilizer* states: a missed byproduct is a deterministic Pauli flip, so success
//! reduces to composed memory-LER. Here we run a chain of `gates` T-gate teleportations on a logical
//! qubit (T is non-Clifford → the intermediate states are non-stabilizer, so the board simulates it with
//! a state vector, not a tableau). Applying T via gate teleportation needs a Z-measurement of a magic
//! ancilla and a *conditional S correction*; that measurement is code-protected (raw = m ⊕ e) and must be
//! DECODED. A wrong decode applies an extra S — and S is NOT a Pauli relative to the verification basis,
//! so a single wrong feed-forward turns the deterministic result into a **quantum-random** outcome
//! (50/50), not a bit flip. With `gates = 8`, T^8 = I, so the correct chain returns the input |+> and
//! the board bins the pass rate by the number of wrong decodes w: w=0→pass, w=1,3→random, w=2→flip,
//! w=4→pass — the period-4 S signature that no classical bit-flip (composed-LER) model can produce.
//!
//! Per trial we emit `gates` independent memory-Z blocks (one per magic measurement) with their true
//! logical flip e; the board decodes each on the real Arty, forms the corrected outcome, and drives the
//! conditional S in a state-vector T-teleportation gadget (`uf_qubit_ff_nonclifford.py`).
//!
//! Usage:
//!   cargo run --release -p aleph-qec --example qec_q6_ff_nonclifford -- [d] [W] [C] [rounds] [trials] [seed] [p] [gates]
//!   # defaults: d=3 W=9 C=3 rounds=17 trials=4000 seed=2024 p=0.005 gates=8
//!   cargo run --release -p aleph-qec --example qec_q6_ff_nonclifford -- 3 9 3 17 4000 2024 0.005 8 > hw/cosim_ffnc_d3.vec
//!
//! Output (stdout):
//!   # comment metadata (d/W/C/dpr/slices/trials/gates/...)
//!   P p=<p> trials=<N>
//!   T <e_1> <e_2> ... <e_gates>        ← one trial: the true logical flip of each magic-measurement block
//!   <dpr bits> × slices                ← block 1 (magic measurement 1 decode), round 0 first
//!   ...                                ← `gates` blocks total, in order

use aleph_qec::{build_dem, sample_shots, SurfaceCode};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let d: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let w: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(9);
    let c: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let rounds: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(17);
    let trials: u64 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(4000);
    let seed: u64 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(2024);
    let p: f64 = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(0.005);
    let gates: usize = a.get(8).and_then(|s| s.parse().ok()).unwrap_or(8);
    assert!(gates >= 1, "need at least one T-teleportation per trial");

    let exp = SurfaceCode::new(d).memory_z_experiment(rounds);
    let det_round = exp.detector_rounds();
    let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();
    assert_eq!(
        dem.observables, 1,
        "non-Clifford feed-forward decode assumes a single logical observable per block"
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

    // `gates` independent magic-measurement decodes per trial.
    let (syndromes, truths) = sample_shots(&dem, gates as u64 * trials, seed);
    let flip = |i: usize| truths[i].first().copied().unwrap_or(false);

    println!(
        "# Q6-26 non-Clifford (T-teleportation) feed-forward trials — GENERATED, do not edit."
    );
    println!(
        "# d={d} W={w} C={c} dpr={dpr} slices={n_slices} detectors={} observables=1 noise=phenom trials={trials} gates={gates} seed={seed}",
        dem.detectors
    );
    println!(
        "# regenerate: cargo run --release -p aleph-qec --example qec_q6_ff_nonclifford -- {d} {w} {c} {rounds} {trials} {seed} {p} {gates}"
    );
    println!("P p={p} trials={trials}");

    let mut line = String::with_capacity(dpr);
    for t in 0..trials as usize {
        let base = gates * t;
        let es: Vec<String> = (0..gates)
            .map(|g| u8::from(flip(base + g)).to_string())
            .collect();
        println!("T {}", es.join(" "));
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
