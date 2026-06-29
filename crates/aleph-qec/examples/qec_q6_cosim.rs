//! Q6-21 (sim) — emit a Monte-Carlo syndrome stream + software-UF baseline for the board-free
//! sim↔RTL co-simulation harness.
//!
//! This is the *driver* half of the board-free hardware-in-the-loop loop (ROADMAP §2.4 co-design):
//! the simulator plays QPU, drawing noisy shots from the **same** detector-error model the RTL
//! decoder's matching graph was generated from (`qec_surface_uf_graph -- graph <d> <rounds>`), and
//! dumps them for the Verilated `uf_surface_decoder` to decode (`hw/tb_uf_cosim.cpp`). The RTL's
//! logical-error rate is then checked against this software [`UnionFindDecoder`] baseline within
//! Monte-Carlo CI — closing the whole verification chain on realistic noise:
//! noise model → syndromes → **RTL** decode → logical error rate. When hardware lands (Q6-08) the
//! same vectors stream over the Q6-07 AXI link instead of into Verilator.
//!
//! Both sides sample *identically*: [`sample_shots`] derives each shot's RNG from `(seed, index)`,
//! so the dumped stream is exactly what the software baseline below decodes — RTL and software see
//! the same shots and their rates are directly comparable.
//!
//! The matching graph is `p`-independent (structure = which mechanisms exist; uniform-noise edges),
//! so one RTL build serves the whole `p` sweep — this emits every `p` block into one vectors file.
//!
//! Usage:
//!   cargo run --release -p aleph-qec --example qec_q6_cosim -- [d] [rounds] [shots] [seed] [p,p,..]
//!   # defaults: d=3 rounds=1 shots=20000 seed=2024 p=0.01,0.02,0.03,0.04,0.05
//!   cargo run --release -p aleph-qec --example qec_q6_cosim -- 3 1 20000 2024 > hw/cosim_d3.vec
//!
//! Output (stdout) — the `.vec` file the TB reads:
//!   # comment lines (metadata)
//!   P p=<p> shots=<n> sw_rate=<r> sw_ci=<c>          ← one block header per p
//!   <det-bits> <obs>                                  ← `shots` lines: char j = detector j fired
//!   ...
//! `<det-bits>` is `detectors` chars, detector 0 first; `<obs>` is the true logical flip (0/1).

use aleph_qec::{
    build_dem, sample_shots, Decoder, LogicalErrorResult, SurfaceCode, UnionFindDecoder,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let d: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(3);
    let rounds: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1);
    let shots: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let seed: u64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(2024);
    let probs: Vec<f64> = args
        .get(5)
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![0.01, 0.02, 0.03, 0.04, 0.05]);

    let code = SurfaceCode::new(d);
    let exp = code.memory_z_experiment(rounds);
    // Detector indexing is fixed by the experiment (p-independent), so it matches the RTL graph
    // emitted by `qec_surface_uf_graph -- graph d rounds` for any p>0.
    let dets = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(0.01, 0.01))
        .unwrap()
        .detectors;

    println!("# Q6-21 sim<->RTL co-sim vectors — GENERATED, do not edit.");
    println!("# d={d} rounds={rounds} detectors={dets} observables=1 shots={shots} seed={seed}");
    println!(
        "# regenerate: cargo run --release -p aleph-qec --example qec_q6_cosim -- {d} {rounds} {shots} {seed}"
    );

    eprintln!("# d={d} rounds={rounds} shots={shots} seed={seed} (software UnionFind baseline)");
    eprintln!("#   p      sw_rate      ci95     errors");

    for &p in &probs {
        let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();
        assert_eq!(dem.detectors, dets, "p must not change detector count");
        assert_eq!(
            dem.observables, 1,
            "co-sim assumes a single logical observable (RTL obs_flip is 1 bit)"
        );

        // The exact shots the RTL will decode (same seed → identical to the software baseline).
        let (syndromes, truths) = sample_shots(&dem, shots, seed);

        // Software UnionFind baseline over those same shots (unweighted, like the RTL engine).
        let dec = UnionFindDecoder::new(&dem).expect("graphlike dem");
        let preds = dec.decode_batch(&syndromes).expect("decode batch");
        let sw_errors = preds
            .iter()
            .zip(&truths)
            .filter(|(pred, truth)| {
                pred.observable_flips.first().copied().unwrap_or(false)
                    != truth.first().copied().unwrap_or(false)
            })
            .count() as u64;
        let sw = LogicalErrorResult::new(shots, sw_errors);

        eprintln!(
            "  {p:.3}  {:.4e}  {:.2e}  {}/{}",
            sw.rate, sw.ci95, sw_errors, shots
        );

        println!(
            "P p={p} shots={shots} sw_rate={} sw_ci={}",
            sw.rate, sw.ci95
        );
        let mut line = String::with_capacity(dets + 3);
        for (syn, truth) in syndromes.iter().zip(&truths) {
            line.clear();
            for j in 0..dets {
                line.push(if syn.is_fired(j as u32) { '1' } else { '0' });
            }
            line.push(' ');
            line.push(if truth.first().copied().unwrap_or(false) {
                '1'
            } else {
                '0'
            });
            println!("{line}");
        }
    }
}
