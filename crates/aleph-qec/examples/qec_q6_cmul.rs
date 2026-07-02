//! Q6-32 Milestone E (the exact Shor modular-exponentiation step) — emit n-bit CONTROLLED in-place
//! modular-multiplier `c-U_a` trials: controlled on a phase-register qubit, |ctrl⟩|x⟩ → |ctrl⟩|(a·x) mod N
//! if ctrl else x⟩. Shor's period-finding is a product of controlled-U_{a^{2^k}} against the phase register,
//! so this is the literal exponentiation primitive. Each T-gate magic-state measurement is resolved by the
//! on-board decoder.
//!
//! Adding the control makes the cost genuinely richer than U_a (Milestone D): the constant-loads that were
//! FREE CNOTs become Toffolis (control ∧ x[i]) and the SWAP becomes a Fredkin — both non-Clifford, both
//! DECODED. So c-U_a = 2 controlled out-of-place multiplies (forward a, inverse a⁻¹) around a controlled
//! SWAP, with a data-dependent T-count = 7·(20n² + n + 2·Hamming(load constants)). ctrl=0 ⇒ identity (every
//! controlled load gives 0 → every VBE adder is the identity; the Fredkin is skipped).
//!
//! Per trial we emit a (ctrl, x) input and `gates` independent memory-Z blocks (one per T-gate) with their
//! true logical flip e; the board decodes each on the Arty, drives the conditional S in a (4n+4)-qubit
//! state-vector controlled multiplier (`uf_qubit_cmul.py`), and verifies the controlled product.
//!
//! Usage:
//!   cargo run --release -p aleph-qec --example qec_q6_cmul -- [n] [N] [a] [d] [W] [C] [rounds] [trials] [seed] [p]
//!   # defaults: n=2 N=2^n-1 a=2 d=3 W=9 C=3 rounds=17 trials=800 seed=2024 p=0.002
//!   cargo run --release -p aleph-qec --example qec_q6_cmul -- 2 3 2 3 9 3 17 40 2024 0.002 > hw/cosim_cmul_n2.vec
//!
//! Output (stdout):
//!   # comment metadata (n/N/a/nbits=4n+4/gates=7·(20n²+n+2·Hamming)/d/W/C/dpr/slices/...)
//!   P p=<p> trials=<N>
//!   T <ctrl 0|1> <x 0..N-1> <e_1> ... <e_gates>   ← one trial: (ctrl,x) input + the T-gate blocks' flips
//!   <dpr bits> × slices                            ← block 1, round 0 first; `gates` blocks in order

use aleph_qec::{build_dem, sample_shots, SurfaceCode};

fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.abs()
}

/// a⁻¹ mod n via extended Euclid (n small; result in 0..n).
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

/// Decoded-Toffoli count of c-U_a: 20n² VBE (2 passes) + n Fredkins + 2·Hamming(load constants).
fn total_toffolis(n: usize, modulus: usize, a: usize) -> usize {
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
        gcd(a_const as i64, modulus as i64) == 1,
        "c-U_a requires gcd(a={a_const}, N={modulus})=1 so a⁻¹ exists"
    );
    let gates = 7 * total_toffolis(n, modulus, a_const); // 7 T per decoded Toffoli

    let exp = SurfaceCode::new(d).memory_z_experiment(rounds);
    let det_round = exp.detector_rounds();
    let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();
    assert_eq!(
        dem.observables, 1,
        "c-U_a T-gate decode assumes a single logical observable per block"
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

    println!("# Q6-32 controlled modular multiplier (VBE c-U_a: ctrl·(a*x) mod N) trials — GENERATED, do not edit.");
    println!(
        "# n={n} N={modulus} a={a_const} nbits={} gates={gates} d={d} W={w} C={c} dpr={dpr} slices={n_slices} detectors={} observables=1 noise=phenom trials={trials} seed={seed}",
        4 * n + 4,
        dem.detectors
    );
    println!(
        "# regenerate: cargo run --release -p aleph-qec --example qec_q6_cmul -- {n} {modulus} {a_const} {d} {w} {c} {rounds} {trials} {seed} {p}"
    );
    println!("P p={p} trials={trials}");

    let n_inputs = 2 * modulus; // sweep (ctrl, x): ctrl in {0,1}, x in 0..N
    let mut line = String::with_capacity(dpr);
    for t in 0..trials as usize {
        let base = gates * t;
        let inp = t % n_inputs;
        let ctrl = inp / modulus;
        let x_in = inp % modulus;
        let es: Vec<String> = (0..gates)
            .map(|g| u8::from(flip(base + g)).to_string())
            .collect();
        println!("T {ctrl} {x_in} {}", es.join(" "));
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
