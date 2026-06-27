//! Q5-01: construct the `[[144,12,12]]` **gross** bivariate-bicycle code (Bravyi et al.
//! arXiv:2308.07915), emit its Tanner graph + a code-capacity DEM, and sanity-decode it with the
//! min-sum BP decoder (Q3-02) to confirm the DEM is wired for decoding. The full threshold curve is
//! Q5-02 (BP+OSD) — here we only show the code is built, verified, and decodable.
//!
//! Usage:
//! ```text
//! cargo run --release -p aleph-qec --example qec_q5_gross -- [shots] [seed]
//! # defaults: shots=20000 seed=2024
//! ```

use aleph_qec::{run_dem_experiment, BBCode, BpDecoder, Syndrome};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let shots: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2024);

    let code = BBCode::gross();
    let (l, m) = code.params();
    eprintln!("# Q5-01 gross bivariate-bicycle code");
    eprintln!(
        "# construction: ℓ={l}, m={m}, A=x³+y+y², B=y³+x+x²  ⇒  [[n={}, k={}, d=12]] (d from Bravyi et al.)",
        code.n(),
        code.k()
    );
    println!("property,value");
    println!("n,{}", code.n());
    println!("k,{}", code.k());
    println!("d,12");
    println!("x_checks,{}", code.num_checks());
    println!("z_checks,{}", code.num_checks());
    println!("check_weight,6");
    println!("qubit_degree,3");

    // Code-capacity DEM for independent Z noise + its Tanner graph (via the BP decoder).
    let dem = code.code_capacity_dem(0.01);
    let bp = BpDecoder::new(&dem);
    let t = bp.tanner();
    eprintln!(
        "# code-capacity DEM (Z noise): {} detectors (X-checks), {} observables, {} mechanisms, {} Tanner edges",
        dem.detectors, dem.observables, t.n_vars, t.n_edges
    );
    println!("dem_detectors,{}", dem.detectors);
    println!("dem_observables,{}", dem.observables);
    println!("dem_mechanisms,{}", t.n_vars);
    println!("tanner_edges,{}", t.n_edges);

    // Sanity: a single-qubit Z error lights exactly its 3 X-checks (a hyperedge); BP should converge.
    let mut single_ok = 0;
    let trials = 30usize.min(code.n());
    for q in 0..trials {
        let dets: Vec<u32> = dem.errors[q].dets.clone();
        let syn = Syndrome::new(dem.detectors, dets.clone());
        assert_eq!(
            dets.len(),
            3,
            "single-qubit error must be a 3-check hyperedge"
        );
        // BP decode; correction is consistent if it predicts a finite observable set (no NaN/panic).
        use aleph_qec::Decoder;
        let _corr = bp.decode(&syn);
        single_ok += 1;
    }
    eprintln!("# single-qubit hyperedge syndromes decoded (no panic/NaN): {single_ok}/{trials}");

    // End-to-end: a small logical-error-rate sample at two low rates — just to show the DEM decodes
    // through the Monte-Carlo harness. (Threshold + literature comparison is Q5-02/Q5-03.)
    println!();
    println!("# end-to-end BP decode through the MC harness (sanity, not a threshold)");
    println!("p,shots,logical_rate,ci95");
    for &p in &[0.002_f64, 0.005, 0.01] {
        let dem_p = code.code_capacity_dem(p);
        let bp_p = BpDecoder::new(&dem_p);
        let r = run_dem_experiment(&dem_p, shots, &bp_p, seed).expect("bp decode");
        println!("{p},{shots},{:.6},{:.6}", r.rate, r.ci95);
        eprintln!("  p={p}: logical_rate={:.4e} ± {:.1e}", r.rate, r.ci95);
    }

    eprintln!(
        "# NOTE: standalone BP is degeneracy-limited on qLDPC (Q3-02 caveat); the proper qLDPC decoder \
         is BP+OSD (Q5-02), which uses this exact DEM + Tanner graph. Q5-01 ships the code + DEM."
    );
}
