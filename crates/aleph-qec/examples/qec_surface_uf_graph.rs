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

use aleph_qec::{build_dem, SlidingWindowDecoder, SurfaceCode, Syndrome, UnionFindDecoder};

/// Shared edge extraction: dedup parallel mechanisms by node pair, ELOG = parity of observable over
/// merged copies (matches `MatchingGraph::from_dem`). `boundary` is the 1-detector edges' far node.
fn extract_edges(errors: &[aleph_qec::DemError], boundary: usize) -> BTreeMap<(usize, usize), u8> {
    let mut edges: BTreeMap<(usize, usize), u8> = BTreeMap::new();
    for e in errors {
        let (a, b) = match e.dets.as_slice() {
            [a, b] => (*a as usize, *b as usize),
            [a] => (*a as usize, boundary),
            _ => continue,
        };
        let key = (a.min(b), a.max(b));
        *edges.entry(key).or_insert(0) ^= u8::from(!e.obs.is_empty());
    }
    edges
}

fn emit_graph_consts(n_nodes: usize, boundary: usize, edges: &BTreeMap<(usize, usize), u8>) {
    let m = edges.len();
    let ea: Vec<String> = edges.keys().map(|(a, _)| a.to_string()).collect();
    let eb: Vec<String> = edges.keys().map(|(_, b)| b.to_string()).collect();
    let el: Vec<String> = edges.values().map(|l| l.to_string()).collect();
    println!("localparam int UF_N = {n_nodes};");
    println!("localparam int UF_M = {m};");
    println!("localparam int UF_BOUNDARY = {boundary};");
    println!("localparam int UF_EA   [UF_M] = '{{{}}};", ea.join(", "));
    println!("localparam int UF_EB   [UF_M] = '{{{}}};", eb.join(", "));
    println!("localparam bit UF_ELOG [UF_M] = '{{{}}};", el.join(", "));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).cloned().unwrap_or_else(|| "graph".into());
    // Optional distance argument (default 3 — keeps the d=3 Makefile targets working unchanged).
    let d: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    // Q6-19: optional measurement-round count (default 1 = code-capacity 2D graph; >1 = multi-round
    // phenomenological 3D space-time graph with time-like measurement-error edges between rounds).
    let rounds: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);

    // Q6-20: `window <d> <W> <C>` — emit the steady-state interior streaming-window graph for the
    // sliding-window FPGA decoder: a W-round volume whose future/past time cuts route to temporal-sink
    // nodes (free obs-less drains to the boundary), plus per-active-detector round + commit metadata.
    // Built from the SAME SlidingWindowDecoder::window_dem the software decoder uses (single source of
    // truth). An interior offset s=W (over a 3W-round experiment) gives both past and future buffers.
    if mode == "window" || mode == "window-circuit" {
        let w: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(3 * d);
        let c: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(d);
        let total = 3 * w;
        let exp = SurfaceCode::new(d).memory_z_experiment(total);
        // `window-circuit`: build the window graph from the full circuit-level (gate-noise) DEM instead
        // of the phenomenological one — same detectors, but extra hook-error space-time edges. The bulk
        // is still translation-invariant, so one interior window graph serves every steady-state window;
        // the per-detector streaming metadata (round/shift/commit) is noise-model-independent.
        let win_circuit = mode == "window-circuit";
        let dem = if win_circuit {
            exp.circuit_level_dem(aleph_qec::CircuitNoise::uniform(0.001))
                .unwrap()
        } else {
            build_dem(&exp.annotated, &exp.phenomenological_mechanisms(0.01, 0.01)).unwrap()
        };
        let drounds = exp.detector_rounds();
        let sw = SlidingWindowDecoder::new(dem, drounds.clone(), w, c);
        let s = w;
        let we = sw.window_dem(s, s + w).unwrap();
        let boundary = we.dem.detectors;
        let edges = extract_edges(&we.dem.errors, boundary);
        let na = we.n_active;
        // Relative round of each real detector (0..W-1); detectors are round-major in the window DEM.
        let rr: Vec<usize> = (0..na).map(|l| drounds[we.globals[l]] - s).collect();
        let mut round_start = vec![usize::MAX; w]; // first local index of each relative round
        for (l, &r) in rr.iter().enumerate() {
            if round_start[r] == usize::MAX {
                round_start[r] = l;
            }
        }
        // Slide-by-C map: detector at round r, position p within its round, carries to the detector at
        // round r-C, same position (new local index `round_start[r-C] + p`). Rounds < C are committed
        // and dropped (sentinel = UF_ACTIVE). Rounds >= W-C are reloaded from the incoming stream.
        let droud: Vec<String> = rr.iter().map(|r| r.to_string()).collect();
        let dcommit: Vec<String> = rr.iter().map(|&r| u8::from(r < c).to_string()).collect();
        let shift: Vec<String> = (0..na)
            .map(|l| {
                let r = rr[l];
                if r >= c {
                    (round_start[r - c] + (l - round_start[r])).to_string()
                } else {
                    na.to_string() // sentinel: committed-and-dropped
                }
            })
            .collect();
        let load_lo = round_start[w - c]; // first local index of the C newest rounds (stream-loaded)
        let dpr = round_start[1]; // detectors per round (round 0 occupies [0, round_start[1]))
                                  // Per-edge commit-touch: 1 if either endpoint is a real detector (< n_active) in the commit
                                  // region (relative round < C). Edges are emitted in `edges` key order = UF_EA/EB order.
        let in_commit = |node: usize| node < na && rr[node] < c;
        let ecommit: Vec<String> = edges
            .keys()
            .map(|&(a, b)| u8::from(in_commit(a) || in_commit(b)).to_string())
            .collect();
        let noise = if win_circuit {
            "circuit-level"
        } else {
            "phenomenological"
        };
        println!("// d={d} W={w} C={c} {noise} streaming-window graph (interior; future/past cuts -> temporal sinks) — GENERATED, do not edit.");
        println!("// regenerate: cargo run -p aleph-qec --example qec_surface_uf_graph -- {mode} {d} {w} {c}");
        // Include guard: the Q6-20 streaming wrapper AND the per-window core both `include this header
        // in one compilation unit; the guard makes the $unit-scope localparams idempotent.
        println!("`ifndef UF_SURFACE_GRAPH_SVH");
        println!("`define UF_SURFACE_GRAPH_SVH");
        emit_graph_consts(boundary + 1, boundary, &edges);
        // Streaming metadata: detectors 0..UF_ACTIVE-1 are real/lit-able; UF_ACTIVE..UF_N-2 are
        // temporal sinks (never lit); UF_N-1 is the spatial boundary. The arrays cover only the real
        // detectors. Unused by the bare core (UF_N/M/edges only) -> pragma-guarded for -Wall.
        println!("/* verilator lint_off UNUSEDPARAM */");
        println!("localparam int UF_ACTIVE  = {na};");
        println!("localparam int UF_W       = {w};");
        println!("localparam int UF_C       = {c};");
        println!("localparam int UF_DPR     = {dpr};  // detectors per measurement round");
        println!("localparam int UF_LOAD_LO = {load_lo};  // real detectors [LOAD_LO, ACTIVE) reload from the stream each slide");
        println!(
            "localparam int UF_DROUND  [UF_ACTIVE] = '{{{}}};",
            droud.join(", ")
        );
        println!(
            "localparam bit UF_DCOMMIT [UF_ACTIVE] = '{{{}}};",
            dcommit.join(", ")
        );
        println!(
            "localparam int UF_SHIFT   [UF_ACTIVE] = '{{{}}};",
            shift.join(", ")
        );
        println!(
            "localparam bit UF_ECOMMIT [UF_M]      = '{{{}}};",
            ecommit.join(", ")
        );
        println!("/* verilator lint_on UNUSEDPARAM */");
        println!("`endif");
        return;
    }

    let exp = SurfaceCode::new(d).memory_z_experiment(rounds);
    // `graph-circuit`: build the matching graph from the full circuit-level DEM (Q-surface) instead
    // of the phenomenological one. It has the same detectors but extra **hook-error** edges (the
    // diagonal space-time edges from CNOT faults), so the RTL decoder sees the realistic graph. The
    // edge set is p-independent (structure only), so a fixed small rate suffices here.
    let circuit = mode == "graph-circuit";
    let dem = if circuit {
        exp.circuit_level_dem(aleph_qec::CircuitNoise::uniform(0.001))
            .unwrap()
    } else {
        // Uniform noise so the unweighted graph the RTL decodes matches the oracle decoder's graph.
        build_dem(&exp.annotated, &exp.phenomenological_mechanisms(0.01, 0.01)).unwrap()
    };
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
        "graph" | "graph-circuit" => {
            let m = edges.len();
            let ea: Vec<String> = edges.keys().map(|(a, _)| a.to_string()).collect();
            let eb: Vec<String> = edges.keys().map(|(_, b)| b.to_string()).collect();
            let el: Vec<String> = edges.values().map(|l| l.to_string()).collect();
            let model = if circuit {
                "circuit-level"
            } else {
                "phenomenological"
            };
            println!("// d={d} rotated surface-code memory-Z ({rounds} round(s)) {model} matching graph — GENERATED, do not edit.");
            println!("// regenerate: cargo run -p aleph-qec --example qec_surface_uf_graph -- {mode} {d} {rounds} > hw/uf_surface_graph_d{d}.svh");
            // Include guard: the board build compiles uf_axi_wrap.sv and uf_surface_decoder.sv (which
            // both `include this header) into one compilation unit, so without a guard the localparams
            // are declared twice. Matches the window-mode header. (Verilator/xsim read files as
            // separate units, so this is a no-op there.)
            println!("`ifndef UF_SURFACE_GRAPH_SVH");
            println!("`define UF_SURFACE_GRAPH_SVH");
            println!("localparam int UF_N = {};", n + 1);
            println!("localparam int UF_M = {m};");
            println!("localparam int UF_BOUNDARY = {boundary};");
            println!("localparam int UF_EA   [UF_M] = '{{{}}};", ea.join(", "));
            println!("localparam int UF_EB   [UF_M] = '{{{}}};", eb.join(", "));
            println!("localparam bit UF_ELOG [UF_M] = '{{{}}};", el.join(", "));
            println!("`endif");
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
            eprintln!(
                "unknown mode '{other}' (use 'graph', 'graph-circuit', 'oracle', 'window', or 'window-circuit')"
            );
            std::process::exit(2);
        }
    }
}
