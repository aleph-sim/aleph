//! Q6-32 Milestone G (Shor end-to-end) — emit Shor ORDER-FINDING trials for `a mod N`: Hadamard an m-qubit
//! phase register into a superposition of exponents, run the modular exponentiation |k⟩|1⟩ → |k⟩|a^k mod N⟩
//! (Milestone F, every T-gate decoded on the Arty), apply the inverse QFT, and measure — the outcome
//! distribution concentrates on multiples of 2^m/r, revealing the period r = ord_N(a), the quantum heart of
//! Shor's factoring. For a=2, N=3 the order is r=2, so with m=2 phase qubits the ideal outcomes are exactly
//! {0, 2} and 2/2^m = 1/2 → r=2.
//!
//! For m=2 the inverse QFT is Clifford (H, controlled-S, SWAP), so every decoded T-gate lives in the modular
//! exponentiation — the emitter therefore samples the SAME `gates = Σ_j 7·(20n² + n + 2·Hamming(a^{2^j}))`
//! T-gate memory-Z blocks per trial as Milestone F (no per-trial input: the phase register is a
//! superposition, so the only per-trial variation is the decode noise). The driver (`uf_qubit_shor.py`)
//! runs the full circuit and reports how much measurement probability the decoder keeps on the ideal peaks.
//!
//! Usage:
//!   cargo run --release -p aleph-qec --example qec_q6_shor -- [n] [N] [a] [m] [d] [W] [C] [rounds] [trials] [seed] [p]
//!   # defaults: n=2 N=2^n-1 a=2 m=2 d=3 W=9 C=3 rounds=17 trials=800 seed=2024 p=0.002
//!   cargo run --release -p aleph-qec --example qec_q6_shor -- 2 3 2 2 3 9 3 17 24 2024 0.002 > hw/cosim_shor_n2.vec
//!
//! Output (stdout):
//!   # comment metadata (n/N/a/m/nbits=m+4n+3/gates/d/W/C/dpr/slices/...)
//!   P p=<p> trials=<N>
//!   T <e_1> ... <e_gates>   ← one trial: the T-gate blocks' true logical flips (no input; phase reg = superposition)
//!   <dpr bits> × slices      ← block 1, round 0 first; `gates` blocks in order

use aleph_qec::{build_dem, sample_shots, SurfaceCode};

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.abs()
}

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

/// Multiplicative order of a mod n (smallest r>0 with a^r ≡ 1).
fn order(a: i64, n: i64) -> i64 {
    let (mut x, mut r) = (a.rem_euclid(n), 1i64);
    while x != 1 && r < n {
        x = (x * a).rem_euclid(n);
        r += 1;
    }
    r
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
        "Shor requires gcd(a={a_const}, N={modulus})=1"
    );
    let r = order(a_const as i64, modulus as i64);
    assert!(
        r > 1 && (1i64 << m) % r == 0,
        "for clean peaks the order r={r} of {a_const} mod {modulus} must satisfy 1<r and r | 2^m (m={m})"
    );
    let gates = 7 * total_toffolis(n, modulus, a_const, m); // all decoded T-gates are in the modexp

    let exp = SurfaceCode::new(d).memory_z_experiment(rounds);
    let det_round = exp.detector_rounds();
    let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();
    assert_eq!(
        dem.observables, 1,
        "Shor T-gate decode assumes a single logical observable per block"
    );

    let n_slices = det_round
        .iter()
        .copied()
        .max()
        .map(|mx| mx + 1)
        .unwrap_or(0);
    let mut by_round: Vec<Vec<usize>> = vec![Vec::new(); n_slices];
    for (dd, &rr) in det_round.iter().enumerate() {
        by_round[rr].push(dd);
    }
    let dpr = by_round[0].len();
    assert!(
        by_round.iter().all(|rr| rr.len() == dpr),
        "expected a fixed detectors-per-round for the streaming frame"
    );
    assert!(
        n_slices >= w && (n_slices - w).is_multiple_of(c),
        "slices ({n_slices}) must be W + k*C (W={w}, C={c}); pick rounds = W + k*C - 1"
    );

    let (syndromes, truths) = sample_shots(&dem, gates as u64 * trials, seed);
    let flip = |i: usize| truths[i].first().copied().unwrap_or(false);

    println!("# Q6-32 Shor order-finding (a mod N, end-to-end) trials — GENERATED, do not edit.");
    println!(
        "# n={n} N={modulus} a={a_const} m={m} r={r} nbits={} gates={gates} d={d} W={w} C={c} dpr={dpr} slices={n_slices} detectors={} observables=1 noise=phenom trials={trials} seed={seed}",
        m + 4 * n + 3,
        dem.detectors
    );
    println!(
        "# regenerate: cargo run --release -p aleph-qec --example qec_q6_shor -- {n} {modulus} {a_const} {m} {d} {w} {c} {rounds} {trials} {seed} {p}"
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
