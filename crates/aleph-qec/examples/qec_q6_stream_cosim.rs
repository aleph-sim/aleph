//! Q6-20 (on silicon) — emit a per-ROUND Monte-Carlo detector stream for the on-board sliding-window
//! streaming decoder (`hw/uf_stream_win_core.sv` over the AXI DMA, driven by `hw/sw/uf_dma_stream.py`).
//!
//! The block co-sim (`qec_q6_cosim`) dumps one whole syndrome per line for the *block* decoder. The
//! streaming decoder instead consumes a stream of measurement ROUNDS one at a time and commits one
//! window every `C` rounds, so this dumps a long continuous stream as `dpr`-bit round lines — the exact
//! round handshake the RTL wrapper drives — sampled from the phenomenological surface-code DEM so the
//! on-board traffic has realistic, space-time-correlated defect density.
//!
//! Correctness gate on the board is **validity** (not per-shot LER): a graphlike decoder always
//! produces a correction that reproduces the syndrome, so once the stream is drained (a tail of zero
//! rounds pushes every defect through the commit region) the residual must clear — exactly the software
//! `residual_after_decode == 0` criterion and the #399 Verilator proof, now reproduced on silicon. This
//! is tie-break- and boundary-independent (unlike an interior-window-only LER, which would systematically
//! omit the final W−C uncommitted rounds of a finite experiment — the documented Q6-20 caveat). The
//! driver appends the drain itself and also uses the stream for a sustained-throughput measurement.
//!
//! Usage:
//!   cargo run --release -p aleph-qec --example qec_q6_stream_cosim -- [d] [W] [C] [rounds] [seed] [p,..]
//!   # defaults: d=3 W=9 C=3 rounds=2000 seed=2024 p=0.01,0.02,0.03,0.04,0.05
//!   cargo run --release -p aleph-qec --example qec_q6_stream_cosim -- 3 9 3 2000 2024 > hw/cosim_stream_d3.vec
//!
//! Output (stdout) — the `.vec` the streaming DMA driver reads:
//!   # comment metadata (d/W/C/dpr/...)
//!   P p=<p> rounds=<R>                 ← one block per p; R = number of round lines that follow
//!   <dpr detector bits>               ← R such lines, round 0 first (round-major, detector 0 first)

use aleph_qec::{build_dem, sample_shots, SurfaceCode};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let d: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let w: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(9);
    let c: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    // Stream length in rounds. One big memory experiment gives `rounds+1` uniform-width detector slices;
    // we emit them as the continuous round stream.
    let rounds: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let seed: u64 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(2024);
    // Noise model: `phenom` (default) or `circuit` (full circuit-level gate noise + hook errors, pairs
    // with the `window-circuit` RTL build); circuit-level defaults to the lower prob grid.
    let noise = a.get(6).map(String::as_str).unwrap_or("phenom");
    assert!(
        matches!(noise, "phenom" | "circuit"),
        "noise must be phenom|circuit"
    );
    let circuit = noise == "circuit";
    let probs: Vec<f64> = a
        .get(7)
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| {
            if circuit {
                vec![0.002, 0.004, 0.006, 0.008, 0.010]
            } else {
                vec![0.01, 0.02, 0.03, 0.04, 0.05]
            }
        });

    let exp = SurfaceCode::new(d).memory_z_experiment(rounds);
    let det_round = exp.detector_rounds();
    let dem_at = |p: f64| {
        if circuit {
            exp.circuit_level_dem(aleph_qec::CircuitNoise::uniform(p))
                .unwrap()
        } else {
            build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap()
        }
    };
    let dets = dem_at(if circuit { 0.002 } else { 0.01 }).detectors;

    // Group detector ids by slice (round-major, index order): the round handshake the RTL feeds. Assert
    // a fixed detectors-per-round `dpr` so the fixed-width stream frame is exact end to end.
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
    assert_eq!(dets, n_slices * dpr, "detector count must be n_slices*dpr");

    println!(
        "# Q6-20 streaming co-sim vectors (per-round detector stream) — GENERATED, do not edit."
    );
    println!("# d={d} W={w} C={c} dpr={dpr} slices={n_slices} detectors={dets} noise={noise} seed={seed}");
    println!(
        "# regenerate: cargo run --release -p aleph-qec --example qec_q6_stream_cosim -- {d} {w} {c} {rounds} {seed} {noise}"
    );

    eprintln!(
        "# d={d} W={w} C={c} slices={n_slices} dpr={dpr} — realistic Monte-Carlo round streams"
    );

    for (bi, &p) in probs.iter().enumerate() {
        let dem = dem_at(p);
        assert_eq!(dem.detectors, dets, "p must not change detector count");
        // One shot = one long correlated round stream at this p (distinct seed per block).
        let (syndromes, _truths) = sample_shots(&dem, 1, seed.wrapping_add(bi as u64));
        let syn = &syndromes[0];

        let fired: usize = (0..dets).filter(|&dd| syn.is_fired(dd as u32)).count();
        eprintln!("  p={p:.3}  fired={fired}/{dets} detectors");

        println!("P p={p} rounds={n_slices}");
        let mut line = String::with_capacity(dpr);
        for round in &by_round {
            line.clear();
            for &dd in round {
                line.push(if syn.is_fired(dd as u32) { '1' } else { '0' });
            }
            println!("{line}");
        }
    }
}
