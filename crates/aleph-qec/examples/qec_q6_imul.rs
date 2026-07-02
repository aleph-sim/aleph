//! Q6-32 Milestone D (the complete Shor multiplicative unitary) — emit n-bit IN-PLACE modular-multiplier
//! `x := (a·x) mod N` trials (|x⟩ → |(a·x) mod N⟩), i.e. exactly U_a — the unitary whose controlled powers
//! Shor's modular exponentiation is a product of. Each T-gate magic-state measurement is resolved by the
//! on-board decoder. In-place multiply is TWO out-of-place multiplies (forward by a, then clear-the-scratch
//! by a⁻¹) around a SWAP, so the T-count doubles again to 140n² (560/1260 for n=2/3), 2n VBE adders deep.
//!
//! Construction (textbook Shor, VBE arithmetic): with registers R1=x, R2=0 — R2 += a·R1 mod N (n
//! controlled-modular-adds of a·2^i mod N); SWAP R1↔R2; R2 −= a⁻¹·R1 mod N (n controlled-modular-adds of
//! −(a⁻¹·2^i)). Requires gcd(a,N)=1. Modular subtract of c is modular add of (N−c) mod N, so both passes
//! reuse the same forward machinery. Each modular add is one unconditional VBE adder (70n T = 10n Toffolis),
//! identity when its addend is 0; 2n of them = 140n² Toffoli-borne T-gate magic-state injections, each
//! code-protected (raw = m ⊕ e) and DECODED on the real Arty (Q6-20 bitstream unchanged). Control /
//! constant-load / SWAP are Clifford (no decode).
//!
//! Per trial we emit a residue input x (0 ≤ x < N) and 140n² independent memory-Z blocks (one per T-gate)
//! with their true logical flip e; the board decodes each on the Arty, drives the conditional S in a
//! (4n+3)-qubit state-vector in-place multiplier (`uf_qubit_imul.py`), and verifies R1 == (a·x) mod N.
//!
//! Usage:
//!   cargo run --release -p aleph-qec --example qec_q6_imul -- [n] [N] [a] [d] [W] [C] [rounds] [trials] [seed] [p]
//!   # defaults: n=2 N=2^n-1 a=2 d=3 W=9 C=3 rounds=17 trials=800 seed=2024 p=0.002
//!   cargo run --release -p aleph-qec --example qec_q6_imul -- 2 3 2 3 9 3 17 60 2024 0.002 > hw/cosim_imul_n2.vec
//!
//! Output (stdout):
//!   # comment metadata (n/N/a/nbits=4n+3/gates=140n^2/d/W/C/dpr/slices/...)
//!   P p=<p> trials=<N>
//!   T <x 0..N-1> <e_1> ... <e_gates>   ← one trial: residue input + the T-gate blocks' flips
//!   <dpr bits> × slices                 ← block 1, round 0 first; `gates` blocks in order

use aleph_qec::{build_dem, sample_shots, SurfaceCode};

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn main() {
    let arg: Vec<String> = std::env::args().collect();
    let n: usize = arg.get(1).and_then(|s| s.parse().ok()).unwrap_or(2);
    let modulus: usize = arg
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or((1usize << n) - 1);
    let a_const: usize = arg.get(3).and_then(|s| s.parse().ok()).unwrap_or(2);
    let d: usize = arg.get(4).and_then(|s| s.parse().ok()).unwrap_or(3);
    let w: usize = arg.get(5).and_then(|s| s.parse().ok()).unwrap_or(9);
    let c: usize = arg.get(6).and_then(|s| s.parse().ok()).unwrap_or(3);
    let rounds: usize = arg.get(7).and_then(|s| s.parse().ok()).unwrap_or(17);
    let trials: u64 = arg.get(8).and_then(|s| s.parse().ok()).unwrap_or(800);
    let seed: u64 = arg.get(9).and_then(|s| s.parse().ok()).unwrap_or(2024);
    let p: f64 = arg.get(10).and_then(|s| s.parse().ok()).unwrap_or(0.002);
    assert!(n >= 1, "need at least a 1-bit multiplier");
    assert!(
        modulus >= 1 && modulus < (1usize << n),
        "modulus N={modulus} must be 1..=2^n-1 to fit in n={n} bits"
    );
    assert!(
        (1..modulus).contains(&a_const),
        "multiplier a={a_const} must be 1..=N-1"
    );
    assert!(
        gcd(a_const, modulus) == 1,
        "in-place U_a requires gcd(a={a_const}, N={modulus})=1 so a⁻¹ exists"
    );
    let gates = 140 * n * n; // 2n VBE modular adders x 70n T (forward a + inverse a^-1)

    let exp = SurfaceCode::new(d).memory_z_experiment(rounds);
    let det_round = exp.detector_rounds();
    let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();
    assert_eq!(
        dem.observables, 1,
        "in-place-multiplier T-gate decode assumes a single logical observable per block"
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

    println!("# Q6-32 in-place modular multiplier (VBE U_a: x:=(a*x) mod N) trials — GENERATED, do not edit.");
    println!(
        "# n={n} N={modulus} a={a_const} nbits={} gates={gates} d={d} W={w} C={c} dpr={dpr} slices={n_slices} detectors={} observables=1 noise=phenom trials={trials} seed={seed}",
        4 * n + 3,
        dem.detectors
    );
    println!(
        "# regenerate: cargo run --release -p aleph-qec --example qec_q6_imul -- {n} {modulus} {a_const} {d} {w} {c} {rounds} {trials} {seed} {p}"
    );
    println!("P p={p} trials={trials}");

    let n_inputs = modulus; // sweep all residues x in 0..N
    let mut line = String::with_capacity(dpr);
    for t in 0..trials as usize {
        let base = gates * t;
        let x_in = t % n_inputs;
        let es: Vec<String> = (0..gates)
            .map(|g| u8::from(flip(base + g)).to_string())
            .collect();
        println!("T {x_in} {}", es.join(" "));
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
