//! Q4-03 latency-budget instrumentation: per-stage decode latency for a single streaming window,
//! broken into **graph build → cluster growth → peel → commit**, for d ∈ {5,7,9,11}, against the
//! < 1 µs/round real-time target (the fault-tolerance constraint from the roadmap).
//!
//! Each parallel/sliding-window decode (Q4-01/Q4-02) rebuilds its matching graph + Union-Find from
//! the window DEM, grows clusters, peels the erasure, and commits the chosen edges' observable flips
//! into the running residual. A window of `W = 3d` rounds commits `C = d` rounds, so the per-round
//! latency is the window decode time divided by `d`. This example times every stage over many
//! syndromes and reports the distribution (p50/p90/p99/max) plus the gap to 1 µs/round — the gap
//! that motivates the Q6 FPGA decoder.
//!
//! Usage:
//! ```text
//! cargo run --release -p aleph-qec --example qec_q4_latency -- [samples] [seed]
//! # defaults: samples=4000 seed=2024
//! ```

use std::time::Instant;

use aleph_qec::{
    build_dem, DetectorErrorModel, MatchingGraph, SurfaceCode, Syndrome, UnionFindDecoder,
};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

fn sample_syndrome(dem: &DetectorErrorModel, rng: &mut Rng) -> Syndrome {
    let mut lit = vec![false; dem.detectors];
    for e in &dem.errors {
        let u = (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
        if u < e.prob {
            for &d in &e.dets {
                lit[d as usize] ^= true;
            }
        }
    }
    Syndrome::from_bits(&lit)
}

/// p-th percentile (0..=100) of a sample set, in microseconds (input is seconds).
fn pct(sorted_secs: &[f64], p: f64) -> f64 {
    if sorted_secs.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted_secs.len() - 1) as f64).round() as usize;
    sorted_secs[idx.min(sorted_secs.len() - 1)] * 1e6
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let samples: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4000);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2024);
    let p = 0.03;
    const TARGET_US_PER_ROUND: f64 = 1.0;

    eprintln!(
        "# Q4-03 latency budget, single window W=3d commit C=d, memory-Z p={p}, {samples} samples"
    );

    // Histogram CSV: one row per (stage, d) with the distribution.
    println!("d,stage,p50_us,p90_us,p99_us,max_us");
    // Budget CSV: per-round latency vs the 1 µs target (median totals).
    let mut budget_rows: Vec<String> = Vec::new();

    for &d in &[5usize, 7, 9, 11] {
        let window = 3 * d; // W = 3d rounds (commit C=d, buffer d each side)
        let exp = SurfaceCode::new(d).memory_z_experiment(window);
        let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();

        let mut rng = Rng(seed ^ (d as u64).wrapping_mul(0x9E37));
        let syns: Vec<Syndrome> = (0..samples)
            .map(|_| sample_syndrome(&dem, &mut rng))
            .collect();

        let mut build = Vec::with_capacity(samples);
        let mut grow = Vec::with_capacity(samples);
        let mut peel = Vec::with_capacity(samples);
        let mut commit = Vec::with_capacity(samples);
        let mut total = Vec::with_capacity(samples);

        for syn in &syns {
            // Stage 1: graph build (MatchingGraph + UnionFind), rebuilt per window per shot exactly
            // as the streaming decoders do.
            let t0 = Instant::now();
            let graph = MatchingGraph::from_dem(&dem).unwrap();
            let dec = UnionFindDecoder::from_graph(&graph);
            let t_build = t0.elapsed().as_secs_f64();

            // Stages 2+3: cluster growth and peel, timed inside the decoder.
            let (_corr, chosen, [g, pl]) = dec.decode_edges_timed(syn);

            // Stage 4: commit — accumulate the chosen edges' observable flips (the streaming window
            // commit work: XOR observables, read endpoints).
            let t3 = Instant::now();
            let mut logical = 0u64;
            let mut acc = 0usize;
            for &e in &chosen {
                let ed = &graph.edges()[e];
                for &o in &ed.observables {
                    if o < 64 {
                        logical ^= 1u64 << o;
                    }
                }
                acc += ed.a + ed.b;
            }
            std::hint::black_box((logical, acc));
            let t_commit = t3.elapsed().as_secs_f64();

            build.push(t_build);
            grow.push(g);
            peel.push(pl);
            commit.push(t_commit);
            total.push(t_build + g + pl + t_commit);
        }

        for v in [&mut build, &mut grow, &mut peel, &mut commit, &mut total] {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }

        for (stage, v) in [
            ("build", &build),
            ("growth", &grow),
            ("peel", &peel),
            ("commit", &commit),
            ("total", &total),
        ] {
            println!(
                "{d},{stage},{:.4},{:.4},{:.4},{:.4}",
                pct(v, 50.0),
                pct(v, 90.0),
                pct(v, 99.0),
                pct(v, 100.0),
            );
        }

        let total_p50 = pct(&total, 50.0);
        let per_round = total_p50 / d as f64; // window commits C=d rounds
        let gap = per_round / TARGET_US_PER_ROUND;
        budget_rows.push(format!(
            "{d},{window},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{per_round:.3},{gap:.0}",
            pct(&build, 50.0),
            pct(&grow, 50.0),
            pct(&peel, 50.0),
            pct(&commit, 50.0),
            total_p50,
            pct(&total, 99.0),
        ));
        eprintln!(
            "  d={d}: window total p50={total_p50:.2}µs (build {:.2} / grow {:.2} / peel {:.2} / commit {:.3}) \
             ⇒ {per_round:.2}µs/round = {gap:.0}× the 1µs target",
            pct(&build, 50.0),
            pct(&grow, 50.0),
            pct(&peel, 50.0),
            pct(&commit, 50.0),
        );
    }

    println!();
    println!("# budget: per-round latency = window total / C(=d) vs 1µs/round target");
    println!("d,window,build_p50_us,grow_p50_us,peel_p50_us,commit_p50_us,total_p50_us,total_p99_us,per_round_us,gap_x");
    for r in &budget_rows {
        println!("{r}");
    }

    eprintln!(
        "# graph build dominates the window budget (rebuilt per window per shot); growth+peel are the \
         actual matching work. Per-round latency is far above 1µs ⇒ the gap motivates Q6 (FPGA): a \
         fixed-array, build-free pipeline that amortises the graph and keeps growth/peel in hardware."
    );
}
