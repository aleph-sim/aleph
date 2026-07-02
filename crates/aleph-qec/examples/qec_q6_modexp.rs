//! Q6-32 Milestone F (the front half of Shor) — emit a short MODULAR EXPONENTIATION `a^k mod N` mapping
//! |k⟩|1⟩ → |k⟩|a^k mod N⟩, the map at the heart of Shor's period-finding. It is a chain of m controlled
//! in-place multipliers: for each phase-register bit k[j], apply controlled-U_{a^{2^j}} to the work register
//! (U_b|x⟩ = |b·x mod N⟩), so the work register accumulates a^(Σ k[j]·2^j) = a^k mod N. Each T-gate
//! magic-state measurement is resolved by the on-board decoder.
//!
//! On a computational-basis k this computes the modexp truth table a^k mod N; on a superposition (Hadamards,
//! not emitted here) it prepares the periodic state |k⟩|a^k mod N⟩ whose period r = ord_N(a) the inverse QFT
//! (the remaining Clifford+T back half of Shor) would extract. Each c-U_{a^{2^j}} is the Milestone-E
//! controlled multiplier (control turns its constant-loads into Toffolis and its SWAP into a Fredkin), so the
//! whole exponentiation is Σ_j 7·(20n² + n + 2·Hamming(load consts of a^{2^j})) T-gate decodes on the real
//! Arty (Q6-20 bitstream unchanged).
//!
//! Per trial we emit an exponent input k (0 ≤ k < 2^m) and `gates` independent memory-Z blocks (one per
//! T-gate) with their true logical flip e; the board decodes each on the Arty, drives the conditional S in a
//! (m+4n+3)-qubit state-vector modexp (`uf_qubit_modexp.py`), and verifies the a^k mod N truth table.
//!
//! Usage:
//!   cargo run --release -p aleph-qec --example qec_q6_modexp -- [n] [N] [a] [m] [d] [W] [C] [rounds] [trials] [seed] [p]
//!   # defaults: n=2 N=2^n-1 a=2 m=2 d=3 W=9 C=3 rounds=17 trials=800 seed=2024 p=0.002
//!   cargo run --release -p aleph-qec --example qec_q6_modexp -- 2 3 2 2 3 9 3 17 24 2024 0.002 > hw/cosim_modexp_n2.vec
//!
//! Output (stdout):
//!   # comment metadata (n/N/a/m/nbits=m+4n+3/gates/d/W/C/dpr/slices/...)
//!   P p=<p> trials=<N>
//!   T <k 0..2^m-1> <e_1> ... <e_gates>   ← one trial: exponent input + the T-gate blocks' flips
//!   <dpr bits> × slices                   ← block 1, round 0 first; `gates` blocks in order

use aleph_qec::{build_dem, sample_shots, SurfaceCode};

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.abs()
}

/// a⁻¹ mod n via extended Euclid.
fn modinv(a: i64, n: i64) -> i64 {
    let (mut t, mut newt) = (0i64, 1i64);
    let (mut r, mut newr) = (n, a.rem_euclid(n));
    while newr != 0 {
        let q = r / newr;
        (t, newt) = (newt, t - q * newt);
        (r, newr) = (newr, r - q * newr);
    }
    assert!(r == 1, "a={a} has no inverse mod n={n}");
    t.rem_euclid(n)
}

/// b^e mod n by repeated squaring.
fn powmod(b: i64, mut e: i64, n: i64) -> i64 {
    let (mut acc, mut base) = (1i64, b.rem_euclid(n));
    while e > 0 {
        if e & 1 == 1 {
            acc = (acc * base).rem_euclid(n);
        }
        base = (base * base).rem_euclid(n);
        e >>= 1;
    }
    acc
}

/// Decoded-Toffoli count of one c-U_a: 20n² VBE + n Fredkins + 2·Hamming(load constants).
fn cua_toffolis(n: usize, modulus: usize, a: usize) -> usize {
    let (nn, mm, aa) = (n as i64, modulus as i64, a as i64);
    let ainv = modinv(aa, mm);
    let mut loads = 0usize;
    for i in 0..nn {
        let fwd = (aa * (1 << i)).rem_euclid(mm);
        let inv = (mm - (ainv * (1 << i)).rem_euclid(mm)).rem_euclid(mm);
        loads += fwd.count_ones() as usize + inv.count_ones() as usize;
    }
    20 * n * n + n + 2 * loads
}

/// Sum over the m chained controlled multipliers c-U_{a^{2^j}}.
fn total_toffolis(n: usize, modulus: usize, a: usize, m: usize) -> usize {
    (0..m)
        .map(|j| {
            cua_toffolis(
                n,
                modulus,
                powmod(a as i64, 1i64 << j, modulus as i64) as usize,
            )
        })
        .sum()
}

fn main() {
    let arg: Vec<String> = std::env::args().collect();
    let n: usize = arg.get(1).and_then(|s| s.parse().ok()).unwrap_or(2);
    let modulus: usize = arg
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or((1usize << n) - 1);
    let a_const: usize = arg.get(3).and_then(|s| s.parse().ok()).unwrap_or(2);
    let m: usize = arg.get(4).and_then(|s| s.parse().ok()).unwrap_or(2);
    let d: usize = arg.get(5).and_then(|s| s.parse().ok()).unwrap_or(3);
    let w: usize = arg.get(6).and_then(|s| s.parse().ok()).unwrap_or(9);
    let c: usize = arg.get(7).and_then(|s| s.parse().ok()).unwrap_or(3);
    let rounds: usize = arg.get(8).and_then(|s| s.parse().ok()).unwrap_or(17);
    let trials: u64 = arg.get(9).and_then(|s| s.parse().ok()).unwrap_or(800);
    let seed: u64 = arg.get(10).and_then(|s| s.parse().ok()).unwrap_or(2024);
    let p: f64 = arg.get(11).and_then(|s| s.parse().ok()).unwrap_or(0.002);
    assert!(n >= 1 && m >= 1, "need n≥1 and m≥1");
    assert!(
        modulus >= 1 && modulus < (1usize << n),
        "modulus N={modulus} must be 1..=2^n-1 to fit in n={n} bits"
    );
    assert!(
        (1..modulus).contains(&a_const),
        "base a={a_const} must be 1..=N-1"
    );
    assert!(
        gcd(a_const as i64, modulus as i64) == 1,
        "modexp requires gcd(a={a_const}, N={modulus})=1"
    );
    let gates = 7 * total_toffolis(n, modulus, a_const, m); // 7 T per decoded Toffoli, summed over m steps

    let exp = SurfaceCode::new(d).memory_z_experiment(rounds);
    let det_round = exp.detector_rounds();
    let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();
    assert_eq!(
        dem.observables, 1,
        "modexp T-gate decode assumes a single logical observable per block"
    );

    let n_slices = det_round
        .iter()
        .copied()
        .max()
        .map(|mx| mx + 1)
        .unwrap_or(0);
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

    println!("# Q6-32 modular exponentiation (a^k mod N, front half of Shor) trials — GENERATED, do not edit.");
    println!(
        "# n={n} N={modulus} a={a_const} m={m} nbits={} gates={gates} d={d} W={w} C={c} dpr={dpr} slices={n_slices} detectors={} observables=1 noise=phenom trials={trials} seed={seed}",
        m + 4 * n + 3,
        dem.detectors
    );
    println!(
        "# regenerate: cargo run --release -p aleph-qec --example qec_q6_modexp -- {n} {modulus} {a_const} {m} {d} {w} {c} {rounds} {trials} {seed} {p}"
    );
    println!("P p={p} trials={trials}");

    let n_inputs = 1usize << m; // sweep all exponents k in 0..2^m
    let mut line = String::with_capacity(dpr);
    for t in 0..trials as usize {
        let base = gates * t;
        let k_in = t % n_inputs;
        let es: Vec<String> = (0..gates)
            .map(|g| u8::from(flip(base + g)).to_string())
            .collect();
        println!("T {k_in} {}", es.join(" "));
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
