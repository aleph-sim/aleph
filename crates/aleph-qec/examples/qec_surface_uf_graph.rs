//! Q6-02 (sim) — emit the d=3 surface-code matching graph + an unweighted Union-Find oracle for the
//! RTL surface-code UF decoder.
//!
//! The RTL UF engine (`hw/uf_surface_decoder.sv`) is **unweighted** (uniform edge growth), so the
//! oracle is generated with the unweighted Rust `UnionFindDecoder` over the *same* graph. We use the
//! distance-3 rotated surface-code memory-Z experiment (1 round) — an 8-detector space-time matching
//! graph with a boundary — and dump:
//!
//!  - `hw/uf_surface_graph.svh`: a SystemVerilog header of compile-time constants (`UF_N`, `UF_M`,
//!    `UF_BOUNDARY`, and the edge tables `UF_EA`/`UF_EB`/`UF_ELOG`) the RTL `includes — the graph is
//!    baked in for this fixed-distance decoder.
//!  - `hw/uf_surface_oracle.mem`: one bit per syndrome index (LSB = detector 0) = the unweighted UF
//!    predicted logical flip (for `$readmemb` in the testbench).
//!
//! Usage:
//!   cargo run --release -p aleph-qec --example qec_surface_uf_graph -- graph  > hw/uf_surface_graph.svh
//!   cargo run --release -p aleph-qec --example qec_surface_uf_graph -- oracle > hw/uf_surface_oracle.mem

use std::collections::BTreeMap;

use aleph_qec::{build_dem, SurfaceCode, Syndrome, UnionFindDecoder};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).cloned().unwrap_or_else(|| "graph".into());
    // Optional distance argument (default 3 — keeps the d=3 Makefile targets working unchanged).
    let d: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    // Q6-19: optional measurement-round count (default 1 = code-capacity 2D graph; >1 = multi-round
    // phenomenological 3D space-time graph with time-like measurement-error edges between rounds).
    let rounds: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);

    let exp = SurfaceCode::new(d).memory_z_experiment(rounds);
    // Uniform noise so the unweighted graph the RTL decodes matches the oracle decoder's graph.
    let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(0.01, 0.01)).unwrap();
    let n = dem.detectors;
    let boundary = n; // boundary node id

    // Edges: dedup parallel mechanisms by node pair; `logical` = parity of obs over merged copies.
    let mut edges: BTreeMap<(usize, usize), u8> = BTreeMap::new();
    for e in &dem.errors {
        let (a, b) = match e.dets.as_slice() {
            [a, b] => (*a as usize, *b as usize),
            [a] => (*a as usize, boundary),
            _ => continue, // 0-detector or hyperedge (none here) — skip
        };
        let key = (a.min(b), a.max(b));
        let log = u8::from(!e.obs.is_empty());
        *edges.entry(key).or_insert(0) ^= log;
    }

    match mode.as_str() {
        "graph" => {
            let m = edges.len();
            let ea: Vec<String> = edges.keys().map(|(a, _)| a.to_string()).collect();
            let eb: Vec<String> = edges.keys().map(|(_, b)| b.to_string()).collect();
            let el: Vec<String> = edges.values().map(|l| l.to_string()).collect();
            println!("// d={d} rotated surface-code memory-Z ({rounds} round(s)) matching graph — GENERATED, do not edit.");
            println!("// regenerate: cargo run -p aleph-qec --example qec_surface_uf_graph -- graph {d} {rounds} > hw/uf_surface_graph_d{d}.svh");
            println!("localparam int UF_N = {};", n + 1);
            println!("localparam int UF_M = {m};");
            println!("localparam int UF_BOUNDARY = {boundary};");
            println!("localparam int UF_EA   [UF_M] = '{{{}}};", ea.join(", "));
            println!("localparam int UF_EB   [UF_M] = '{{{}}};", eb.join(", "));
            println!("localparam bit UF_ELOG [UF_M] = '{{{}}};", el.join(", "));
        }
        "oracle" => {
            // Unweighted UF to match the unweighted RTL engine.
            let decoder = UnionFindDecoder::new(&dem).expect("uf");
            println!("// d=3 surface-code unweighted Union-Find logical-flip oracle");
            println!("// detectors={n} entries={}", 1usize << n);
            for idx in 0u32..(1u32 << n) {
                let bits: Vec<bool> = (0..n).map(|b| (idx >> b) & 1 == 1).collect();
                let syn = Syndrome::from_bits(&bits);
                let corr = aleph_qec::Decoder::decode(&decoder, &syn);
                println!(
                    "{}",
                    u8::from(corr.observable_flips.first().copied().unwrap_or(false))
                );
            }
        }
        other => {
            eprintln!("unknown mode '{other}' (use 'graph' or 'oracle')");
            std::process::exit(2);
        }
    }
}
