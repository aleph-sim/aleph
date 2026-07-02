//! Q6-25 (feed-forward) — emit teleportation trials whose two byproduct-measurement outcomes are each
//! resolved by the on-board sliding-window decoder, so the on-silicon decode result *conditions the next
//! logical operation* (the teleportation byproduct Pauli) rather than just being scored offline.
//!
//! This complements the closed-loop memory lifetime (`qec_q6_stream_ler`/`uf_qubit_in_a_box.py`, Q6-24):
//! there the correction is applied to a passive frame; here the decode result drives a conditional
//! quantum gate in a genuine Clifford teleportation gadget on the board (`uf_qubit_feedforward.py`).
//!
//! Model: each trial teleports a single-qubit stabilizer input (|0>,|1>,|+>,|->). Teleportation needs
//! two logical-measurement outcomes (the Bell-measurement bits m_x, m_z) to select the byproduct
//! Paulis X^{m_x} Z^{m_z}. In a real FT machine each of those is a *code-protected* logical measurement
//! whose raw outcome is corrupted by a logical measurement error e and must be DECODED. So per trial we
//! emit two independent memory-Z blocks (one per byproduct bit) whose true logical flip is that e; the
//! board decodes each on the real Arty to get ê, forms the corrected byproduct bit, applies it as a real
//! conditional gate, and verifies the teleported state. The board contrasts:
//!   * ON  — byproduct = decoder-corrected outcome  -> teleportation succeeds at ~(1-p_L) per used bit;
//!   * OFF — byproduct = raw (undecoded) outcome     -> teleportation is corrupted at the raw error rate.
//!
//! The gap is the on-silicon proof that the real-time decode steers the computation (impossible open-loop).
//!
//! Honest scope: for Clifford inputs the teleportation success reduces to "was the byproduct decode
//! right" (= composed memory-LER); the new content is the conditional-operation control flow + the
//! ON/OFF contrast, not a new error mechanism. Non-Clifford (magic-state / adaptive-T) feed-forward that
//! is genuinely non-reducible needs a state-vector logical sim and is a separate track.
//!
//! Usage:
//!   cargo run --release -p aleph-qec --example qec_q6_feedforward -- [d] [W] [C] [rounds] [trials] [seed] [p]
//!   # defaults: d=3 W=9 C=3 rounds=17 trials=8000 seed=2024 p=0.01
//!   cargo run --release -p aleph-qec --example qec_q6_feedforward -- 3 9 3 17 8000 2024 0.01 > hw/cosim_ff_d3.vec
//!
//! Output (stdout) — the `.vec` the feed-forward driver reads:
//!   # comment metadata (d/W/C/dpr/slices/trials/...)
//!   P p=<p> trials=<N>
//!   T <input 0..3> <e_x> <e_z>         ← one trial: input label + the two blocks' true logical flips
//!   <dpr bits> × slices                ← block_x (byproduct m_x decode), round 0 first
//!   <dpr bits> × slices                ← block_z (byproduct m_z decode)

use aleph_qec::{build_dem, sample_shots, SurfaceCode};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let d: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let w: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(9);
    let c: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let rounds: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(17);
    let trials: u64 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(8000);
    let seed: u64 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(2024);
    let p: f64 = a.get(7).and_then(|s| s.parse().ok()).unwrap_or(0.01);

    let exp = SurfaceCode::new(d).memory_z_experiment(rounds);
    let det_round = exp.detector_rounds();
    let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();
    assert_eq!(
        dem.observables, 1,
        "feed-forward byproduct decode assumes a single logical observable per block"
    );

    // Round-major detector grouping (the RTL round handshake), same framing as qec_q6_stream_ler.
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

    // Two independent byproduct-measurement decodes per trial: draw 2*trials shots.
    let (syndromes, truths) = sample_shots(&dem, 2 * trials, seed);
    let flip = |i: usize| truths[i].first().copied().unwrap_or(false);

    println!("# Q6-25 feed-forward teleportation trials — GENERATED, do not edit.");
    println!(
        "# d={d} W={w} C={c} dpr={dpr} slices={n_slices} detectors={} observables=1 noise=phenom trials={trials} seed={seed}",
        dem.detectors
    );
    println!(
        "# regenerate: cargo run --release -p aleph-qec --example qec_q6_feedforward -- {d} {w} {c} {rounds} {trials} {seed} {p}"
    );
    println!("P p={p} trials={trials}");

    let mut line = String::with_capacity(dpr);
    let emit_block = |idx: usize, line: &mut String| {
        let syn = &syndromes[idx];
        for round in &by_round {
            line.clear();
            for &dd in round {
                line.push(if syn.is_fired(dd as u32) { '1' } else { '0' });
            }
            println!("{line}");
        }
    };

    for t in 0..trials as usize {
        let ix = 2 * t; // block_x (byproduct m_x)
        let iz = 2 * t + 1; // block_z (byproduct m_z)
        let input = t % 4; // sweep |0>,|1>,|+>,|->
        println!("T {input} {} {}", u8::from(flip(ix)), u8::from(flip(iz)));
        emit_block(ix, &mut line);
        emit_block(iz, &mut line);
    }
}
