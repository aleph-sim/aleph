//! Q6-32 Milestone B (the Shor-relevant primitive) — emit n-bit modular-adder `b := (a+b) mod N` trials
//! whose T-gate magic-state measurements are each resolved by the on-board decoder. This is the
//! modular-arithmetic core that Shor's algorithm stacks into modular multiplication/exponentiation
//! (Vedral-Barenco-Ekert, arXiv:quant-ph/9511018), so its T-count is INTRINSIC and genuinely deep in the
//! high-T region — 70n T-gates (140/210/280 for n=2/3/4), 2.5–5× the plain adder of Milestone A.
//!
//! The VBE modular adder is FIVE ripple-carry (Cuccaro) adders + a conditional subtract of N — in order:
//! `b += a`, then `b -= N`, then `t ← overflow(b)`, then `b += (t? N : 0)`, then `b -= a`, then reset `t`,
//! then `b += a`. Each Cuccaro add/sub = 2n Toffolis; 5 adders = 10n Toffolis; each Toffoli = 7 T/T† (Q6-27). So the
//! circuit is 70n T-gate magic-state injections, each code-protected (raw = m ⊕ e) and DECODED on the real
//! Arty (one memory-Z block each, Q6-20 bitstream unchanged). X/CNOT — including the classical-N load and
//! the t-controlled load — are Clifford (no decode).
//!
//! Per trial we emit a valid (a,b) input pair (a,b < N) and 70n independent memory-Z blocks (one per
//! T-gate) with their true logical flip e; the board decodes each on the Arty, drives the conditional S in
//! a (3n+3)-qubit state-vector VBE adder (`uf_qubit_modadd.py`), and verifies b == (a+b) mod N — ON vs OFF.
//!
//! Usage:
//!   cargo run --release -p aleph-qec --example qec_q6_modadd -- [n] [N] [d] [W] [C] [rounds] [trials] [seed] [p]
//!   # defaults: n=2 N=2^n-1 d=3 W=9 C=3 rounds=17 trials=800 seed=2024 p=0.002
//!   cargo run --release -p aleph-qec --example qec_q6_modadd -- 2 3 3 9 3 17 128 2024 0.002 > hw/cosim_modadd_n2.vec
//!
//! Output (stdout):
//!   # comment metadata (n/N/nbits=3n+3/gates=70n/d/W/C/dpr/slices/...)
//!   P p=<p> trials=<N>
//!   T <a 0..N-1> <b 0..N-1> <e_1> ... <e_gates>   ← one trial: valid input pair + the T-gate blocks' flips
//!   <dpr bits> × slices                            ← block 1, round 0 first; `gates` blocks in order

use aleph_qec::{build_dem, sample_shots, SurfaceCode};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let n: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(2);
    let modulus: usize = a
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or((1usize << n) - 1);
    let d: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let w: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(9);
    let c: usize = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(3);
    let rounds: usize = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(17);
    let trials: u64 = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(800);
    let seed: u64 = a.get(8).and_then(|s| s.parse().ok()).unwrap_or(2024);
    let p: f64 = a.get(9).and_then(|s| s.parse().ok()).unwrap_or(0.002);
    assert!(n >= 1, "need at least a 1-bit adder");
    assert!(
        modulus >= 1 && modulus < (1usize << n),
        "modulus N={modulus} must be 1..=2^n-1 to fit in n={n} bits (VBE requires a,b < N < 2^n)"
    );
    let gates = 70 * n; // 5 adders x 2n Toffolis x 7 T

    let exp = SurfaceCode::new(d).memory_z_experiment(rounds);
    let det_round = exp.detector_rounds();
    let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();
    assert_eq!(
        dem.observables, 1,
        "modular-adder T-gate decode assumes a single logical observable per block"
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

    println!("# Q6-32 modular adder (VBE b:=(a+b) mod N) trials — GENERATED, do not edit.");
    println!(
        "# n={n} N={modulus} nbits={} gates={gates} d={d} W={w} C={c} dpr={dpr} slices={n_slices} detectors={} observables=1 noise=phenom trials={trials} seed={seed}",
        3 * n + 3,
        dem.detectors
    );
    println!(
        "# regenerate: cargo run --release -p aleph-qec --example qec_q6_modadd -- {n} {modulus} {d} {w} {c} {rounds} {trials} {seed} {p}"
    );
    println!("P p={p} trials={trials}");

    let n_inputs = modulus * modulus; // sweep all valid (a,b) pairs with a,b < N
    let mut line = String::with_capacity(dpr);
    for t in 0..trials as usize {
        let base = gates * t;
        let inp = t % n_inputs;
        let a_in = inp / modulus;
        let b_in = inp % modulus;
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
