//! Q7-02 milestone M1 — emit the gross-code Tanner graph + fixed-point parameters as a SystemVerilog
//! header, and generate check-update test vectors for the Verilator testbench.
//!
//! The RTL relay-BP decoder bakes the fixed `[[144,12,12]]` Tanner graph in at compile time (as the
//! UF decoder does with `uf_surface_graph.svh`). This dumps:
//!
//!  - `graph`   → `hw/bb_gross_tanner.svh`: compile-time constants (`BP_N`, `BP_C`, `BP_E`,
//!    `MSG_BITS`, `FRAC_BITS`, …) and the flattened CSR tables (`BP_VAR_OFF`, `BP_CHECK_OFF`,
//!    `BP_CHECK_EDGES`, `BP_EDGE_VAR`), quantised priors `BP_LAMBDA`, and disorder `BP_GAMMA`, at
//!    the M0-chosen width **(8,3)**.
//!  - `vectors` → `hw/bp_check_vectors.txt`: `T` random `(syndrome, m_vc)` inputs and the
//!    fixed-point golden `e_cv` from [`FixedRelayBp::check_update_once`], for the M1 check-update TB.
//!
//! Usage:
//!   cargo run --release -p aleph-qec --example qec_q7_bp_graph -- graph   > hw/bb_gross_tanner.svh
//!   cargo run --release -p aleph-qec --example qec_q7_bp_graph -- vectors > hw/bp_check_vectors.txt
//!
//! M9b (Q7-04) fit-de-risk probe — a single interior W-round sliding-window Tanner header (see
//! [`emit_win_graph`] for why one window stands in for every steady-state window):
//!   cargo run --release -p aleph-qec --example qec_q7_bp_graph -- \
//!       wingraph <rounds> <p> <W> <C> <bankW> <bankV> > hw/bb_win_tanner.svh
//!
//! M9b (Q7-04) streaming decoder header + golden vectors — the window graph/banking above plus the
//! per-detector slide (`BP_SHIFT`) and per-var commit (`BP_VAR_COMMIT`) metadata the RTL streaming FSM
//! needs to replay [`HwSlidingWindowBp`]'s baked-graph schedule, and the bit-exact co-sim vectors:
//!   cargo run --release -p aleph-qec --example qec_q7_bp_graph -- \
//!       streamgraph <rounds> <p> <W> <C> <bankW> <bankV> > hw/bb_stream_tanner.svh
//!   cargo run --release -p aleph-qec --example qec_q7_bp_graph -- \
//!       streamvectors <rounds> <p> <W> <C> <n> <seed> > hw/bp_stream_vectors.txt

use aleph_qec::{
    BBCode, CircuitNoise, DetectorErrorModel, FixedHwView, FixedRelayBp, HwSlidingWindowBp,
    SlidingWindowBp,
};
use std::collections::{HashMap, HashSet};

/// M0-chosen fixed-point word: 8-bit signed, 3 fractional bits (Q5.3).
const MSG_BITS: u32 = 8;
const FRAC_BITS: u32 = 3;

/// M5 relay-BP schedule: `LEGS × ITERS` message-passing sweeps. The M5 budget study (`qec_q7_budget`)
/// found **6×10** reproduces the full 4×25 LER within Monte-Carlo CI (1.06×) while cutting the RTL
/// schedule from 100 to 60 sweeps (301→181 cycles) — because relay-BP's strength is leg diversity, so
/// *many short legs* beat *few long ones* at equal sweep budget. `GAMMA`/`SEED` are the M0 golden
/// defaults (identical to `FixedRelayBp::new`), so legs 0–3 keep their original γ and 4–5 are new.
const LEGS: usize = 6;
const ITERS: u32 = 10;
const GAMMA: (f64, f64) = (-0.3, 0.9);
const SEED: u64 = 0x5E1A_4B9C;

/// Build the DEM + fixed-point relay-BP decoder. `circuit = Some((rounds, p))` selects the depth-7
/// circuit-level DEM (irregular, much larger graph — the co-sim target); `None` is the code-capacity
/// gross graph the RTL bakes in by default.
fn build(circuit: Option<(usize, f64)>) -> (DetectorErrorModel, FixedRelayBp) {
    let code = BBCode::gross();
    let dem = match circuit {
        Some((rounds, p)) => code
            .circuit_level_dem(rounds, CircuitNoise::uniform(p))
            .expect("circuit-level DEM"),
        None => code.code_capacity_dem(0.03),
    };
    let fx = FixedRelayBp::with_budget(&dem, LEGS, ITERS, GAMMA, SEED, MSG_BITS, FRAC_BITS);
    (dem, fx)
}

fn emit_graph() {
    let (_dem, fx) = build(None);
    print_graph(&fx.hw_view(), "Gross BB code [[144,12,12]] code capacity");
    println!("`endif");
}

/// Emit the SAME `.svh` format for the depth-7 **circuit-level** DEM (rounds × p) — an irregular,
/// much larger graph (e.g. rounds=1: 864 vars, 144 checks, max check-degree 25 vs code-capacity's
/// uniform 6). Written to a separate file and cp'd over `bb_gross_tanner.svh` for the M2 co-sim build,
/// so the parametric RTL decodes it unchanged (the `bpcirc` Makefile target).
///
/// M7 (Q7-02): also solves and appends the offline `K`-banked relay-BP datapath tables — which
/// physical slot each check occupies, which capacity-`bank_v` group each var occupies, and the
/// resulting per-edge `(check, position, β)` (spec A2.2; see `solve_banking`) — so the RTL bakes in a
/// fixed, mux-light bank/group schedule instead of computing one at runtime.
fn emit_circ_graph(rounds: usize, p: f64, bank_w: usize, bank_v: usize) {
    let (_dem, fx) = build(Some((rounds, p)));
    let view = fx.hw_view();
    print_graph(
        &view,
        &format!("Gross BB code [[144,12,12]] CIRCUIT-LEVEL (depth-7, rounds={rounds}, p={p})"),
    );
    let banking = solve_banking(view, bank_w, bank_v, BANK_SOLVE_SEED);
    print_banking(&banking);
    print_rom_rows(&view, &banking);
    println!("`endif");
}

/// M9b (Q7-04) **fit de-risk probe**: emit the SAME `.svh` format for a single **interior W-round
/// sliding window** of the `rounds`-round circuit-level gross DEM, instead of the whole-stream
/// `rounds=1` graph `emit_circ_graph` bakes. `SlidingWindowBp` interior windows are
/// translation-invariant (pinned by `relay_window::tests`), so one compiled window DEM stands in
/// for every steady-state window the M9b streaming RTL will decode — this is the graph an OOC
/// synthesis run checks for KV260 fit *before* the streaming FSM is written (design spec
/// `docs/superpowers/specs/2026-07-11-q7-04-streaming-relay-bp-design.md` § 4-M9b, "Fit de-risk
/// FIRST").
///
/// `s` is chosen away from both the head (round 0) and the boundary-widened final window: `s =
/// (rounds − W) / 2`, clamped to `>= 1`, with `s + W <= rounds` asserted so the window never
/// reaches the last detector round (`rounds`, inclusive — see `BBMemoryExperiment::detector_rounds`)
/// where the stream's boundary-commit special case lives.
fn emit_win_graph(rounds: usize, p: f64, w: usize, c: usize, bank_w: usize, bank_v: usize) {
    let code = BBCode::gross();
    let dem = code
        .circuit_level_dem(rounds, CircuitNoise::uniform(p))
        .expect("circuit-level DEM");
    let detector_rounds = code.memory_x_experiment(rounds).detector_rounds();
    let sw = SlidingWindowBp::new(dem, detector_rounds, w, c);

    let s = ((rounds.saturating_sub(w)) / 2).max(1);
    assert!(
        s + w <= rounds,
        "wingraph: window [{s}, {}) is not interior to a {rounds}-round stream (need s + W <= \
         rounds) — pick a smaller W or a longer stream",
        s + w
    );
    let win = sw.window_dem(s, s + w);

    let fx = FixedRelayBp::with_budget(&win.dem, LEGS, ITERS, GAMMA, SEED, MSG_BITS, FRAC_BITS);
    let view = fx.hw_view();
    // C is recorded for provenance only: the window graph depends on W alone; the commit
    // metadata (VAR_COMMIT/SHIFT/...) that consumes C is M9b's `streamgraph`, not this probe.
    println!(
        "// M9b fit-probe window: rounds={rounds}, p={p}, W={w}, C={c} (provenance only), s={s} — \
         window checks={}, vars={}, edges={}",
        view.n_checks, view.n_vars, view.n_edges
    );
    print_graph(
        &view,
        &format!(
            "Gross BB code [[144,12,12]] SLIDING WINDOW (rounds={rounds}, p={p}, W={w}, C={c}, s={s})"
        ),
    );
    let banking = solve_banking(view, bank_w, bank_v, BANK_SOLVE_SEED);
    print_banking(&banking);
    print_rom_rows(&view, &banking);
    println!("`endif");
}

/// M9b (Q7-04) **streaming decoder header**: the same interior-window Tanner graph + banking tables
/// `emit_win_graph` emits (M9b's fit-de-risk probe), PLUS the streaming metadata the RTL streaming FSM
/// needs to run [`HwSlidingWindowBp`]'s baked-graph schedule — the per-window-detector slide target
/// (`BP_SHIFT`) and the per-window-var commit bit (`BP_VAR_COMMIT`). Unlike `emit_win_graph` (whose `C`
/// is provenance-only), this mode's window graph AND metadata both key on `C`.
///
/// Usage:
///   cargo run --release -p aleph-qec --example qec_q7_bp_graph -- \
///       streamgraph <rounds> <p> <W> <C> <bankW> <bankV> > hw/bb_stream_tanner.svh
fn emit_stream_graph(rounds: usize, p: f64, w: usize, c: usize, bank_w: usize, bank_v: usize) {
    let code = BBCode::gross();
    let dem = code
        .circuit_level_dem(rounds, CircuitNoise::uniform(p))
        .expect("circuit-level DEM");
    let detector_rounds = code.memory_x_experiment(rounds).detector_rounds();
    // HwSlidingWindowBp::new takes the detector-round vector by value; keep our own copy to derive
    // the streaming metadata below (it also re-validates the round-major/uniform-dpr layout this
    // whole M9b mode depends on — see the type's doc comment).
    let hw = HwSlidingWindowBp::new(dem, detector_rounds.clone(), w, c);

    let export = hw.window_export();
    let fx = FixedRelayBp::with_budget(&export.dem, LEGS, ITERS, GAMMA, SEED, MSG_BITS, FRAC_BITS);
    let view = fx.hw_view();
    println!(
        "// M9b streaming window: rounds={rounds}, p={p}, W={w}, C={c}, bankW={bank_w}, \
         bankV={bank_v} — window checks={}, vars={}, edges={}",
        view.n_checks, view.n_vars, view.n_edges
    );
    println!(
        "// regenerate: cargo run --release -q -p aleph-qec --example qec_q7_bp_graph -- \
         streamgraph {rounds} {p} {w} {c} {bank_w} {bank_v} > hw/bb_stream_tanner.svh"
    );
    print_graph(
        &view,
        &format!(
            "Gross BB code [[144,12,12]] STREAMING WINDOW (rounds={rounds}, p={p}, W={w}, C={c})"
        ),
    );
    let banking = solve_banking(view, bank_w, bank_v, BANK_SOLVE_SEED);
    print_banking(&banking);
    print_rom_rows(&view, &banking);

    // --- Streaming metadata: the per-detector slide target and per-var commit bit the RTL FSM
    // reads every slot. Derived from the SAME window export the graph/banking tables above came
    // from, so a header regen can never see the graph and the metadata disagree.
    let dpr = hw.dpr();
    let bp_c = view.n_checks;
    assert_eq!(
        bp_c,
        w * dpr,
        "streamgraph: BP_C ({bp_c}) != BP_WIN_W*BP_DPR ({w}*{dpr}) — every window detector must be \
         a check (a hyperedge check would break this)"
    );
    let load_lo = (w - c) * dpr;

    // Relative round of each local window detector, computed from the ACTUAL global detector round
    // (not a closed-form `l / dpr`) — the UF `window` emitter's `round_start` pattern — so a future
    // non-uniform window layout fails this assert loudly instead of silently emitting a wrong shift.
    let base_round = detector_rounds[export.globals[0]];
    let rr: Vec<usize> = export
        .globals
        .iter()
        .map(|&g| detector_rounds[g] - base_round)
        .collect();
    let mut round_start = vec![usize::MAX; w]; // first local index of each relative round
    for (l, &r) in rr.iter().enumerate() {
        if round_start[r] == usize::MAX {
            round_start[r] = l;
        }
    }
    // Slide-by-C map: a kept (rel round >= C) detector at (round r, position p within its round)
    // carries to the detector at (round r-C, same position) — local index `round_start[r-C] + p`.
    // Rounds < C are committed and dropped this slide (sentinel = BP_C).
    let shift: Vec<usize> = (0..bp_c)
        .map(|l| {
            let r = rr[l];
            if r >= c {
                round_start[r - c] + (l - round_start[r])
            } else {
                bp_c // sentinel: committed-and-dropped
            }
        })
        .collect();

    // --- Inline asserts (house style: unconditional at generation time, like `verify_banking`).
    // (1) Commit tiling: every real stream detector's round is covered by the C*num_slots budget
    //     the slot schedule replays (this bounds every window var's in-window det rounds too, since
    //     those are a subset of the full stream's detector rounds).
    let num_slots = hw.num_slots();
    for &r in &detector_rounds {
        assert!(
            r < c * num_slots,
            "streamgraph: detector round {r} not covered by commit tiling (C={c} * \
             num_slots={num_slots} = {})",
            c * num_slots
        );
    }
    // (2) Every window var has >= 1 in-window detector (compile_window drops the rest, but a future
    //     refactor could reintroduce a degenerate all-truncated mechanism — catch it here).
    for (v, e) in export.dem.errors.iter().enumerate() {
        assert!(
            !e.dets.is_empty(),
            "streamgraph: window var {v} has no in-window detector"
        );
    }
    // (3) SHIFT is injective on kept (non-sentinel) local detectors — two window detectors sliding
    //     onto the same target index would silently corrupt the RTL's residual frame.
    let mut seen_shift: HashSet<usize> = HashSet::new();
    for (l, &sh) in shift.iter().enumerate() {
        if sh != bp_c {
            assert!(
                seen_shift.insert(sh),
                "streamgraph: BP_SHIFT not injective — det {l} collides with an earlier det at \
                 target {sh}"
            );
        }
    }

    let shift_str = shift
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let commit_str = hw
        .commit_mask()
        .iter()
        .map(|&cv| u8::from(cv).to_string())
        .collect::<Vec<_>>()
        .join(", ");

    println!("/* verilator lint_off UNUSEDPARAM */");
    println!("localparam int BP_DPR      = {dpr};  // detectors per round");
    println!("localparam int BP_WIN_W    = {w};  // window rounds");
    println!("localparam int BP_WIN_C    = {c};  // commit rounds per slide");
    println!(
        "localparam int BP_LOAD_LO  = {load_lo};  // (BP_WIN_W-BP_WIN_C)*BP_DPR, reload region base"
    );
    println!(
        "localparam int BP_SHIFT [BP_C]  = '{{{shift_str}}};  // det l -> l - BP_WIN_C*BP_DPR, \
         sentinel BP_C if dropped"
    );
    println!(
        "localparam bit BP_VAR_COMMIT [BP_N] = '{{{commit_str}}};  // 1 iff var has an in-window \
         det at rel round < C"
    );
    println!("/* verilator lint_on UNUSEDPARAM */");
    println!("`endif");
}

/// M9b (Q7-04) **streaming golden vectors**: `n` sampled multi-round shots, decoded slot-by-slot on
/// [`HwSlidingWindowBp::decode_stream_trace`] (the hardware-schedule golden — see that type's doc
/// comment), for the RTL streaming FSM co-sim gate. Round lines are the RAW stream detector bits: the
/// RTL XORs commit toggles into its own frame, so the vectors must NOT be pre-toggled.
///
/// With `early` the golden decodes first-valid instead of best-kept (the RTL core's early-exit
/// mode) — a genuinely different decision stream, so it gets its own vector file (M6–M8 house
/// pattern, like `circvectorsearly`).
///
/// Usage:
///   cargo run --release -p aleph-qec --example qec_q7_bp_graph -- \
///       streamvectors <rounds> <p> <W> <C> <n> <seed> > hw/bp_stream_vectors.txt
///   cargo run --release -p aleph-qec --example qec_q7_bp_graph -- \
///       streamvectorsearly <rounds> <p> <W> <C> <n> <seed> > hw/bp_stream_vectors_early.txt
fn emit_stream_vectors(
    rounds: usize,
    p: f64,
    w: usize,
    c: usize,
    n: usize,
    seed: u64,
    early: bool,
) {
    use aleph_qec::sample_shots;
    let code = BBCode::gross();
    let dem = code
        .circuit_level_dem(rounds, CircuitNoise::uniform(p))
        .expect("circuit-level DEM");
    let detector_rounds = code.memory_x_experiment(rounds).detector_rounds();
    let hw = HwSlidingWindowBp::new(dem.clone(), detector_rounds, w, c).with_early_exit(early);

    let dpr = hw.dpr();
    let slices = rounds + 1; // detector rounds run 0..=rounds (BBMemoryExperiment::detector_rounds)
    let slots = hw.num_slots();
    let export = hw.window_export();
    let bp_n = export.dem.errors.len();
    let bp_obs = export.dem.observables;

    let (syndromes, _truths) = sample_shots(&dem, n as u64, seed);

    let (mode_name, mode_arg, out_file) = if early {
        (
            "early-exit (first-valid)",
            "streamvectorsearly",
            "hw/bp_stream_vectors_early.txt",
        )
    } else {
        (
            "full-decode (best-kept)",
            "streamvectors",
            "hw/bp_stream_vectors.txt",
        )
    };
    println!(
        "# streaming-window {mode_name} golden vectors (rounds={rounds}, p={p}, W={w}, C={c}) — \
         GENERATED, do not edit."
    );
    println!(
        "# regenerate: cargo run --release -q -p aleph-qec --example qec_q7_bp_graph -- \
         {mode_arg} {rounds} {p} {w} {c} {n} {seed} > {out_file}"
    );
    println!(
        "# format: header 'T SLICES DPR SLOTS BP_N BP_OBS'; per trial: SLICES 'r' lines (DPR bits, \
         round-major, round 0 first, det 0 first) then SLOTS 'w' lines '<slot> <BP_N bits \
         committed> <BP_OBS bits obs> <0|1 vflag> <0|1 commit_clean>'"
    );
    println!("{n} {slices} {dpr} {slots} {bp_n} {bp_obs}");

    let total_dets = slices * dpr;
    for syn in &syndromes {
        let mut lit = vec![false; total_dets];
        for &d in &syn.fired {
            if (d as usize) < total_dets {
                lit[d as usize] = true;
            }
        }
        for r in 0..slices {
            let bits: String = lit[r * dpr..(r + 1) * dpr]
                .iter()
                .map(|&b| if b { '1' } else { '0' })
                .collect();
            println!("r {bits}");
        }

        let (_corr, _stats, trace) = hw.decode_stream_trace(syn);
        for (slot, t) in trace.iter().enumerate() {
            let committed: String = t
                .committed
                .iter()
                .map(|&b| char::from(b'0' + (b & 1)))
                .collect();
            let obs: String = (0..bp_obs)
                .map(|o| char::from(b'0' + (((t.obs >> o) & 1) as u8)))
                .collect();
            println!(
                "w {slot} {committed} {obs} {} {}",
                u8::from(t.valid),
                u8::from(t.commit_clean)
            );
        }
    }
}

/// Print the flattened Tanner graph + fixed-point params as an `.svh` header. Works for any DEM (the
/// node degrees are read from the CSR, so the irregular circuit-level graph emits identically).
fn print_graph(v: &FixedHwView, title: &str) {
    let ints = |xs: &[u32]| -> String {
        xs.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let signed = |xs: &[i32]| -> String {
        xs.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };

    println!("// {title} — Tanner graph + fixed-point relay-BP params — GENERATED, do not edit.");
    println!("// regenerate: cargo run -p aleph-qec --example qec_q7_bp_graph -- graph > hw/bb_gross_tanner.svh");
    println!("`ifndef BB_GROSS_TANNER_SVH");
    println!("`define BB_GROSS_TANNER_SVH");
    println!("localparam int BP_N        = {};", v.n_vars);
    println!("localparam int BP_C        = {};", v.n_checks);
    println!("localparam int BP_E        = {};", v.n_edges);
    println!("localparam int BP_OBS      = {};", v.num_observables);
    println!("localparam int MSG_BITS    = {};", v.msg_bits);
    println!("localparam int FRAC_BITS   = {};", v.frac_bits);
    println!("localparam int MAX_MAG     = {};", v.max_mag);
    println!("localparam int BP_LEGS     = {};", v.legs);
    println!("localparam int BP_ITERS    = {};", v.iters_per_leg);
    // Max node degrees (loop bounds for the per-node FSM passes); gross code is regular 6 / 3.
    let chk_deg = v
        .check_off
        .windows(2)
        .map(|w| w[1] - w[0])
        .max()
        .unwrap_or(0);
    let var_deg = v.var_off.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0);
    println!("localparam int BP_CHK_DEG  = {chk_deg};");
    println!("localparam int BP_VAR_DEG  = {var_deg};");
    println!();
    // Variable-major CSR (var_off has N+1 entries; edges of v are var_off[v]..var_off[v+1]).
    println!(
        "localparam int BP_VAR_OFF     [BP_N+1] = '{{{}}};",
        ints(v.var_off)
    );
    println!(
        "localparam int BP_EDGE_VAR    [BP_E]   = '{{{}}};",
        ints(v.edge_var)
    );
    // Check-major CSR (check_off has C+1 entries; edges of c are check_edges[check_off[c]..]).
    println!(
        "localparam int BP_CHECK_OFF   [BP_C+1] = '{{{}}};",
        ints(v.check_off)
    );
    println!(
        "localparam int BP_CHECK_EDGES [BP_E]   = '{{{}}};",
        ints(v.check_edges)
    );
    // Quantised priors and per-leg disorder (γ flattened row-major: BP_GAMMA[leg*BP_N + v]).
    println!(
        "localparam int BP_LAMBDA      [BP_N]   = '{{{}}};",
        signed(v.lambda_q)
    );
    let gamma_flat: Vec<i32> = v.gamma_q.iter().flatten().copied().collect();
    println!(
        "localparam int BP_GAMMA       [BP_LEGS*BP_N] = '{{{}}};",
        signed(&gamma_flat)
    );
    // Observable-flip mask per variable (BP_OBS ≤ 12 bits ⇒ fits an int).
    let obs_mask: Vec<i32> = v.obs.iter().map(|&m| (m & 0xFFF) as i32).collect();
    println!(
        "localparam int BP_OBS_MASK    [BP_N]   = '{{{}}};",
        signed(&obs_mask)
    );
}

/// Deterministic seed for the offline banking solve (`solve_banking`) — matches the verified Python
/// prototype's `seed=7` (`.superpowers/sdd/task-9-reference-solver.py::try_nopi`).
const BANK_SOLVE_SEED: u64 = 7;

/// `(slot, in-check-position)` bank cap within a single var group for the offline banking solve —
/// verified feasible at `cap=2` for `(bank_w, bank_v) ∈ {(8,24), (12,36), (16,48)}` by the reference
/// Python prototype (spec A2.2). Not exposed as a knob: cap=1 (LUTRAM) was the preferred-but-infeasible
/// case in the prototype sweep, cap=2 (RAMB18) is what ships.
const BANK_CAP: usize = 2;

/// Eviction-repair iteration budget for the var-grouping local search (spec A2.2 §3). Past this, the
/// greedy + local-search heuristic is presumed stuck and `solve_banking` panics rather than looping
/// forever — the spec's documented fallback is an exact König-theorem bipartite matching, out of scope
/// here since the heuristic is verified feasible at all three probe configs.
const BANK_EVICT_BUDGET: usize = 200_000;

/// A tiny deterministic xorshift64 PRNG (Marsaglia 2003, "Xorshift RNGs") for the offline banking
/// solve's tie-breaks and eviction-repair randomness. Not a new crate dependency — the solve only needs
/// a cheap, reproducible, non-cryptographic stream, and the whole generator is 15 lines.
struct Xorshift64(u64);

impl Xorshift64 {
    /// `seed` must be nonzero for xorshift64 (an all-zero state is a fixed point); remap 0 defensively.
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform float in `[0, 1)`, top 53 bits of the stream (fills an `f64` mantissa exactly).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in `[0, n)`. `n` must be nonzero.
    fn gen_range(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Offline bank/group assignment for the `K`-banked relay-BP RTL datapath (spec A2.2), solved once at
/// header-emit time (never on the RTL critical path). Fields mirror the emitted header tables 1:1.
struct Banking {
    /// `BP_BANK_W` — number of check slots.
    w: usize,
    /// `BP_BANK_V` — capacity (vars) per var group.
    v: usize,
    /// `BP_GC` — number of check groups (rows), `ceil(BP_C / w)`.
    gc: usize,
    /// `BP_GV` — number of var groups, `ceil(BP_N / v)`.
    gv: usize,
    /// `BP_CHK_AT[g*w+j]`: check id at (group `g`, slot `j`), or `-1` if empty.
    chk_at: Vec<i32>,
    /// `BP_VAR_AT[h*v+i]`: var id at (group `h`, position `i`), or `-1` if empty.
    var_at: Vec<i32>,
    /// `BP_EDGE_CHK[e]`: the check owning edge `e` (canonical edge order, same as `BP_EDGE_VAR`).
    edge_chk: Vec<u32>,
    /// `BP_EDGE_POS[e]`: `e`'s logical position within its check's edge list.
    edge_pos: Vec<u32>,
    /// `BP_EDGE_BETA[e]`: 0/1 half-bank split within `e`'s var group.
    edge_beta: Vec<u8>,

    // ---- M7 (Vivado-folding rescue) literal resolution tables: the offline inverses of the maps above,
    // baked so the RTL directly indexes them instead of scanning `BP_CHK_AT`/`BP_VAR_AT` at elaboration
    // (the scan loops materialise as ~386k-LUT hardware scanners under Vivado; Verilator const-folds them,
    // Vivado does not — see the file-header post-mortem in `hw/bp_relay_banked.sv`).
    /// `BP_CHK_GRP[c]`: check `c`'s group `g` (inverse of `BP_CHK_AT`; every check has one).
    chk_grp: Vec<i32>,
    /// `BP_CHK_SLOT[c]`: check `c`'s slot `j`.
    chk_slot: Vec<i32>,
    /// `BP_VAR_GRP[v]`: var `v`'s group `h` (inverse of `BP_VAR_AT`).
    var_grp: Vec<i32>,
    /// `BP_VAR_SLOT[v]`: var `v`'s position `i` within its group.
    var_slot: Vec<i32>,
    /// `BP_EDGE_HB[e]`: `e`'s m_cm half-bank id, `(slot_j*BP_CHK_DEG + pos_k)*2 + beta` (= `eb*2+beta`).
    edge_hb: Vec<i32>,
    /// `BP_EDGE_EB[e]`: `e`'s e_cm bank id, `slot_j*BP_CHK_DEG + pos_k`.
    edge_eb: Vec<i32>,
    /// `BP_EDGE_ROW[e]`: `e`'s m_cm/e_cm row = its check's group `g`.
    edge_row: Vec<i32>,
    /// `BP_EDGE_EPORT[e]`: 0/1 e_cm read port (A/B) — among the edges of `e`'s var-group that read the
    /// same e_cm bank, ordered by (var-slot `i`, then position `d`), the first gets 0, the second 1.
    edge_eport: Vec<i32>,
}

/// Solve the offline bank/group assignment: a direct Rust port of the verified Python prototype
/// `.superpowers/sdd/task-9-reference-solver.py::try_nopi` (spec A2.2), using an independent xorshift64
/// stream rather than Python's Mersenne Twister — the RNG only breaks ties / drives eviction order, so
/// the two implementations need not draw bit-identical numbers, only both find *a* feasible assignment
/// (verified exactly at the end, every time; see `verify_banking`).
///
/// Algorithm:
/// 1. **Slot assignment.** Checks in descending-degree order; each check picks the slot `j` (of `w`,
///    capacity `gc`) minimizing the count of `(in-check-position, var)` pairs already seen in `j`
///    (ties: lower current occupancy, then RNG). A check's row within its slot is its insertion order.
/// 2. **Var grouping (cap `BANK_CAP`).** Bank of edge `e` = `(slot_of[check(e)], pos_k[e])`. Vars in
///    descending-degree order; place into the emptiest group where every bank stays ≤ `BANK_CAP`
///    counting the var's own edges; on failure, an eviction-repair loop (random group, evict a
///    conflicting var, re-queue it) up to `BANK_EVICT_BUDGET` iterations.
/// 3. **β assignment.** Per `(group, bank)` pair with 2 edges, β = 0/1 split by edge-id order;
///    singletons get β = 0.
/// 4. **Exact verify**, always on (`verify_banking`).
///
/// # Panics
/// If the eviction-repair loop exceeds `BANK_EVICT_BUDGET`, or if the exact verify finds any bank/group
/// invariant violated (both would indicate a solver bug or an infeasible `(bank_w, bank_v)` choice).
fn solve_banking(hw: FixedHwView<'_>, bank_w: usize, bank_v: usize, seed: u64) -> Banking {
    let bp_n = hw.n_vars;
    let bp_c = hw.n_checks;
    let bp_e = hw.n_edges;
    let gc = bp_c.div_ceil(bank_w);
    let gv = bp_n.div_ceil(bank_v);

    // Canonical edge -> (check, in-check logical position `k`). Fixed forever by the CSR the graph
    // emitter already built above; the banking solve only ever reads this, never reorders it — the
    // golden bit-exactness of BP_EDGE_VAR / BP_CHECK_EDGES etc. does not depend on the banking choice.
    let mut pos_k = vec![0u32; bp_e];
    let mut check_of = vec![0u32; bp_e];
    for c in 0..bp_c {
        let lo = hw.check_off[c] as usize;
        let hi = hw.check_off[c + 1] as usize;
        for (k, &e) in hw.check_edges[lo..hi].iter().enumerate() {
            pos_k[e as usize] = k as u32;
            check_of[e as usize] = c as u32;
        }
    }

    let mut rng = Xorshift64::new(seed);

    // --- Step 1: slot assignment.
    let mut slot_of = vec![usize::MAX; bp_c];
    let mut chk_row = vec![0usize; bp_c];
    let mut slot_cnt = vec![0usize; bank_w];
    let mut seen: Vec<HashSet<(u32, u32)>> = (0..bank_w).map(|_| HashSet::new()).collect();

    let mut corder: Vec<usize> = (0..bp_c).collect();
    corder.sort_by_key(|&c| std::cmp::Reverse(hw.check_off[c + 1] - hw.check_off[c]));

    for c in corder {
        let lo = hw.check_off[c] as usize;
        let hi = hw.check_off[c + 1] as usize;
        let mut best_j = None;
        let mut best_cost = 0usize;
        let mut best_cnt = 0usize;
        let mut best_rand = 0.0f64;
        for j in 0..bank_w {
            if slot_cnt[j] >= gc {
                continue;
            }
            let mut cost = 0usize;
            for (k, &e) in hw.check_edges[lo..hi].iter().enumerate() {
                if seen[j].contains(&(k as u32, hw.edge_var[e as usize])) {
                    cost += 1;
                }
            }
            let r = rng.next_f64();
            let better = best_j.is_none()
                || (cost, slot_cnt[j]) < (best_cost, best_cnt)
                || ((cost, slot_cnt[j]) == (best_cost, best_cnt) && r < best_rand);
            if better {
                best_j = Some(j);
                best_cost = cost;
                best_cnt = slot_cnt[j];
                best_rand = r;
            }
        }
        let j = best_j.expect("banking solve: no free slot for check (bank_w too small for BP_C)");
        slot_of[c] = j;
        chk_row[c] = slot_cnt[j];
        slot_cnt[j] += 1;
        for (k, &e) in hw.check_edges[lo..hi].iter().enumerate() {
            seen[j].insert((k as u32, hw.edge_var[e as usize]));
        }
    }

    let bank_of = |e: usize| -> (usize, u32) { (slot_of[check_of[e] as usize], pos_k[e]) };
    let edges_of_var = |vv: usize| (hw.var_off[vv] as usize)..(hw.var_off[vv + 1] as usize);

    // --- Step 2: var grouping, cap BANK_CAP per bank within a group (greedy + eviction repair).
    let mut var_groups: Vec<Vec<usize>> = vec![Vec::new(); gv];
    let mut gb: Vec<HashMap<(usize, u32), usize>> = vec![HashMap::new(); gv];

    let fits = |vv: usize, g: usize, gb: &[HashMap<(usize, u32), usize>]| -> bool {
        let mut cnt: HashMap<(usize, u32), usize> = HashMap::new();
        for e in edges_of_var(vv) {
            *cnt.entry(bank_of(e)).or_insert(0) += 1;
        }
        cnt.iter()
            .all(|(b, c)| gb[g].get(b).copied().unwrap_or(0) + c <= BANK_CAP)
    };
    let add = |vv: usize,
               g: usize,
               var_groups: &mut [Vec<usize>],
               gb: &mut [HashMap<(usize, u32), usize>]| {
        var_groups[g].push(vv);
        for e in edges_of_var(vv) {
            *gb[g].entry(bank_of(e)).or_insert(0) += 1;
        }
    };
    let remove = |vv: usize,
                  g: usize,
                  var_groups: &mut [Vec<usize>],
                  gb: &mut [HashMap<(usize, u32), usize>]| {
        var_groups[g].retain(|&x| x != vv);
        for e in edges_of_var(vv) {
            let slot = gb[g]
                .get_mut(&bank_of(e))
                .expect("banking solve: gb entry missing on remove (internal invariant violated)");
            *slot -= 1;
        }
    };

    let mut vorder: Vec<usize> = (0..bp_n).collect();
    vorder.sort_by_key(|&vv| std::cmp::Reverse(hw.var_off[vv + 1] - hw.var_off[vv]));

    let mut pending: Vec<usize> = Vec::new();
    for vv in vorder {
        let cands: Vec<usize> = (0..gv)
            .filter(|&g| var_groups[g].len() < bank_v && fits(vv, g, &gb))
            .collect();
        if let Some(&g) = cands.iter().min_by_key(|&&g| var_groups[g].len()) {
            add(vv, g, &mut var_groups, &mut gb);
        } else {
            pending.push(vv);
        }
    }

    let mut t = 0usize;
    while !pending.is_empty() && t < BANK_EVICT_BUDGET {
        t += 1;
        let vv = pending.pop().expect("pending checked non-empty");
        let g = rng.gen_range(gv);
        loop {
            let ok = var_groups[g].len() < bank_v && fits(vv, g, &gb);
            if ok || var_groups[g].is_empty() {
                break;
            }
            let v_banks: HashSet<(usize, u32)> = edges_of_var(vv).map(&bank_of).collect();
            let blockers: Vec<usize> = var_groups[g]
                .iter()
                .copied()
                .filter(|&w| edges_of_var(w).map(&bank_of).any(|b| v_banks.contains(&b)))
                .collect();
            let w = if !blockers.is_empty() {
                blockers[rng.gen_range(blockers.len())]
            } else {
                var_groups[g][rng.gen_range(var_groups[g].len())]
            };
            remove(w, g, &mut var_groups, &mut gb);
            pending.push(w);
        }
        add(vv, g, &mut var_groups, &mut gb);
    }
    assert!(
        pending.is_empty(),
        "banking solve failed — see spec A2.2 König fallback ({} var(s) unplaced after {} eviction \
         iterations at (bank_w={bank_w}, bank_v={bank_v}))",
        pending.len(),
        BANK_EVICT_BUDGET
    );

    // --- Step 3: β assignment — per (group, bank) pair, split by edge-id order.
    let mut edge_beta = vec![0u8; bp_e];
    for grp in &var_groups {
        let mut bank_edges: HashMap<(usize, u32), Vec<u32>> = HashMap::new();
        for &vv in grp {
            for e in edges_of_var(vv) {
                bank_edges.entry(bank_of(e)).or_default().push(e as u32);
            }
        }
        for edges in bank_edges.values_mut() {
            edges.sort_unstable();
            for (i, &e) in edges.iter().enumerate() {
                edge_beta[e as usize] = u8::from(i > 0);
            }
        }
    }

    // --- Assemble the flat header tables.
    let mut chk_at = vec![-1i32; gc * bank_w];
    for c in 0..bp_c {
        chk_at[chk_row[c] * bank_w + slot_of[c]] = c as i32;
    }
    let mut var_at = vec![-1i32; gv * bank_v];
    for (h, grp) in var_groups.iter().enumerate() {
        for (i, &vv) in grp.iter().enumerate() {
            var_at[h * bank_v + i] = vv as i32;
        }
    }

    // --- M7 literal resolution tables (inverses of the maps above), computed offline so the RTL indexes
    // them directly. `chk_row[c]`/`slot_of[c]` ARE the group/slot of check c (chk_at[chk_row*W+slot]=c),
    // so grp_of_chk(c)=chk_row[c] and slot_of_chk(c)=slot_of[c] by construction.
    let chk_deg_max = hw
        .check_off
        .windows(2)
        .map(|w| (w[1] - w[0]) as usize)
        .max()
        .unwrap_or(0);
    let var_deg_max = hw
        .var_off
        .windows(2)
        .map(|w| (w[1] - w[0]) as usize)
        .max()
        .unwrap_or(0);

    let mut chk_grp = vec![0i32; bp_c];
    let mut chk_slot = vec![0i32; bp_c];
    for c in 0..bp_c {
        chk_grp[c] = chk_row[c] as i32;
        chk_slot[c] = slot_of[c] as i32;
    }
    let mut var_grp = vec![0i32; bp_n];
    let mut var_slot = vec![0i32; bp_n];
    for (h, grp) in var_groups.iter().enumerate() {
        for (i, &vv) in grp.iter().enumerate() {
            var_grp[vv] = h as i32;
            var_slot[vv] = i as i32;
        }
    }
    // Per-edge banks: eb = slot_of_chk(check) * BP_CHK_DEG + pos_k; hb = eb*2 + beta; row = grp_of_chk.
    let mut edge_eb = vec![0i32; bp_e];
    let mut edge_hb = vec![0i32; bp_e];
    let mut edge_row = vec![0i32; bp_e];
    for e in 0..bp_e {
        let c = check_of[e] as usize;
        let eb = slot_of[c] * chk_deg_max + pos_k[e] as usize;
        edge_eb[e] = eb as i32;
        edge_hb[e] = (eb * 2 + edge_beta[e] as usize) as i32;
        edge_row[e] = chk_row[c] as i32;
    }
    // Per-edge e_cm read port: replicate the RTL `ecm_port` (i,d)-ordered count exactly — within a var
    // group, edges reading the same e_cm bank get port 0 (first) then 1 (second), in (slot i, edge d)
    // order. The count-so-far of a bank at the moment we visit an edge IS its port.
    let mut edge_eport = vec![0i32; bp_e];
    for grp in &var_groups {
        let mut bank_cnt: HashMap<i32, i32> = HashMap::new();
        for i in 0..bank_v {
            let Some(&vv) = grp.get(i) else { continue }; // empty var slot -> vedge_at == -1
            let vdeg = (hw.var_off[vv + 1] - hw.var_off[vv]) as usize;
            for d in 0..var_deg_max {
                if d >= vdeg {
                    continue; // d beyond this var's degree -> vedge_at == -1
                }
                let e = hw.var_off[vv] as usize + d;
                let bank = edge_eb[e];
                let cnt = bank_cnt.entry(bank).or_insert(0);
                edge_eport[e] = *cnt;
                *cnt += 1;
            }
        }
    }

    let banking = Banking {
        w: bank_w,
        v: bank_v,
        gc,
        gv,
        chk_at,
        var_at,
        edge_chk: check_of,
        edge_pos: pos_k,
        edge_beta,
        chk_grp,
        chk_slot,
        var_grp,
        var_slot,
        edge_hb,
        edge_eb,
        edge_row,
        edge_eport,
    };
    verify_banking(hw, &banking);
    banking
}

/// Exact verify of a solved [`Banking`] (spec A2.2 step 4), run unconditionally at the end of
/// `solve_banking`. Operates on the emitted tables themselves (not the solver's internal state), so it
/// also catches bugs in the flat-array assembly, not just the placement heuristic.
///
/// # Panics
/// On any invariant violation: a check or var placed twice (or never) in `BP_CHK_AT`/`BP_VAR_AT`, a
/// duplicate `(slot, position)` bank within one check group, a duplicate `(slot, position, β)`
/// half-bank within one var group, or a `(slot, position)` bank hit more than `BANK_CAP` times within
/// one var group.
fn verify_banking(hw: FixedHwView<'_>, b: &Banking) {
    let bp_c = hw.n_checks;
    let bp_n = hw.n_vars;

    // Every check / var appears exactly once in BP_CHK_AT / BP_VAR_AT.
    let mut chk_seen = vec![false; bp_c];
    for &c in &b.chk_at {
        if c >= 0 {
            let c = c as usize;
            assert!(c < bp_c, "BP_CHK_AT: check id {c} out of range");
            assert!(!chk_seen[c], "BP_CHK_AT: check {c} placed twice");
            chk_seen[c] = true;
        }
    }
    assert!(
        chk_seen.iter().all(|&s| s),
        "BP_CHK_AT: not every check was placed"
    );

    let mut var_seen = vec![false; bp_n];
    for &v in &b.var_at {
        if v >= 0 {
            let v = v as usize;
            assert!(v < bp_n, "BP_VAR_AT: var id {v} out of range");
            assert!(!var_seen[v], "BP_VAR_AT: var {v} placed twice");
            var_seen[v] = true;
        }
    }
    assert!(
        var_seen.iter().all(|&s| s),
        "BP_VAR_AT: not every var was placed"
    );

    // Reconstruct check -> slot from the emitted table (rather than trusting solver-internal state).
    let mut check_slot = vec![-1i32; bp_c];
    for g in 0..b.gc {
        for j in 0..b.w {
            let c = b.chk_at[g * b.w + j];
            if c >= 0 {
                check_slot[c as usize] = j as i32;
            }
        }
    }

    // Per check group g: all its edges' (slot, position) are distinct.
    for g in 0..b.gc {
        let mut seen: HashSet<(usize, usize)> = HashSet::new();
        for j in 0..b.w {
            let c = b.chk_at[g * b.w + j];
            if c < 0 {
                continue;
            }
            let c = c as usize;
            let lo = hw.check_off[c] as usize;
            let hi = hw.check_off[c + 1] as usize;
            for k in 0..(hi - lo) {
                assert!(
                    seen.insert((j, k)),
                    "check group {g}: duplicate bank ({j},{k})"
                );
            }
        }
    }

    // Per var group h: (slot, position) hit <= BANK_CAP times, and (slot, position, beta) half-banks
    // are all distinct.
    for h in 0..b.gv {
        let mut bank_cnt: HashMap<(i32, u32), usize> = HashMap::new();
        let mut half_seen: HashSet<(i32, u32, u8)> = HashSet::new();
        for i in 0..b.v {
            let vv = b.var_at[h * b.v + i];
            if vv < 0 {
                continue;
            }
            let vv = vv as usize;
            for e in hw.var_off[vv]..hw.var_off[vv + 1] {
                let e = e as usize;
                let c = b.edge_chk[e] as usize;
                let j = check_slot[c];
                assert!(j >= 0, "edge {e}: check {c} not placed in BP_CHK_AT");
                let k = b.edge_pos[e];
                let beta = b.edge_beta[e];
                *bank_cnt.entry((j, k)).or_insert(0) += 1;
                assert!(
                    half_seen.insert((j, k, beta)),
                    "var group {h}: duplicate half-bank ({j},{k},{beta})"
                );
            }
        }
        for (&(j, k), &cnt) in &bank_cnt {
            assert!(
                cnt <= BANK_CAP,
                "var group {h}: bank ({j},{k}) hit {cnt} times (cap {BANK_CAP})"
            );
        }
    }

    // ---- M7 literal resolution tables: cross-check the inverses against the primary maps + solve.
    let chk_deg_max = hw
        .check_off
        .windows(2)
        .map(|w| (w[1] - w[0]) as usize)
        .max()
        .unwrap_or(0);

    // BP_CHK_GRP/BP_CHK_SLOT invert BP_CHK_AT: chk_at[grp*W + slot] == c for every check.
    for c in 0..bp_c {
        let g = b.chk_grp[c];
        let j = b.chk_slot[c];
        assert!(
            g >= 0 && (g as usize) < b.gc && j >= 0 && (j as usize) < b.w,
            "BP_CHK_GRP/SLOT: check {c} maps to out-of-range (g={g}, j={j})"
        );
        assert_eq!(
            b.chk_at[g as usize * b.w + j as usize],
            c as i32,
            "BP_CHK_GRP/SLOT: check {c} inverse ({g},{j}) does not point back in BP_CHK_AT"
        );
    }
    // BP_VAR_GRP/BP_VAR_SLOT invert BP_VAR_AT.
    for v in 0..bp_n {
        let h = b.var_grp[v];
        let i = b.var_slot[v];
        assert!(
            h >= 0 && (h as usize) < b.gv && i >= 0 && (i as usize) < b.v,
            "BP_VAR_GRP/SLOT: var {v} maps to out-of-range (h={h}, i={i})"
        );
        assert_eq!(
            b.var_at[h as usize * b.v + i as usize],
            v as i32,
            "BP_VAR_GRP/SLOT: var {v} inverse ({h},{i}) does not point back in BP_VAR_AT"
        );
    }
    // Per-edge EB/HB/ROW recompute from the (independently cross-checked) chk grp/slot tables.
    for e in 0..hw.n_edges {
        let c = b.edge_chk[e] as usize;
        let eb = b.chk_slot[c] * chk_deg_max as i32 + b.edge_pos[e] as i32;
        assert_eq!(b.edge_eb[e], eb, "BP_EDGE_EB[{e}] != slot*CHK_DEG + pos");
        assert_eq!(
            b.edge_hb[e],
            eb * 2 + b.edge_beta[e] as i32,
            "BP_EDGE_HB[{e}] != eb*2 + beta"
        );
        assert_eq!(
            b.edge_row[e], b.chk_grp[c],
            "BP_EDGE_ROW[{e}] != check group"
        );
    }
    // BP_EDGE_EPORT: within each (var-group h, e_cm bank EB) at most one port-0 and one port-1, and every
    // port is 0/1 (the offline cap-2 assignment guarantees <=2 readers/bank/group).
    let mut port_seen: HashMap<(i32, i32, i32), ()> = HashMap::new();
    for e in 0..hw.n_edges {
        let h = b.var_grp[hw.edge_var[e] as usize];
        let eb = b.edge_eb[e];
        let port = b.edge_eport[e];
        assert!(
            port == 0 || port == 1,
            "BP_EDGE_EPORT[{e}] = {port} (expected 0 or 1)"
        );
        assert!(
            port_seen.insert((h, eb, port), ()).is_none(),
            "BP_EDGE_EPORT: var group {h} e_cm bank {eb} has two edges on port {port}"
        );
    }
}

/// Print the [`Banking`] solve as `.svh` `localparam` tables — same `localparam int NAME [SIZE] =
/// '{...};` style as `print_graph`, appended after it (existing lines stay byte-identical).
fn print_banking(b: &Banking) {
    let ints = |xs: &[i32]| -> String {
        xs.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let uints = |xs: &[u32]| -> String {
        xs.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let bytes = |xs: &[u8]| -> String {
        xs.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };

    println!("localparam int BP_BANK_W = {};", b.w);
    println!("localparam int BP_BANK_V = {};", b.v);
    println!("localparam int BP_GC = {};", b.gc);
    println!("localparam int BP_GV = {};", b.gv);
    println!(
        "localparam int BP_CHK_AT [BP_GC*BP_BANK_W] = '{{{}}};",
        ints(&b.chk_at)
    );
    println!(
        "localparam int BP_VAR_AT [BP_GV*BP_BANK_V] = '{{{}}};",
        ints(&b.var_at)
    );
    println!(
        "localparam int BP_EDGE_CHK [BP_E] = '{{{}}};",
        uints(&b.edge_chk)
    );
    println!(
        "localparam int BP_EDGE_POS [BP_E] = '{{{}}};",
        uints(&b.edge_pos)
    );
    println!(
        "localparam int BP_EDGE_BETA [BP_E] = '{{{}}};",
        bytes(&b.edge_beta)
    );

    // ---- M7 literal resolution tables (Vivado-folding rescue): direct-index replacements for the RTL's
    // former scan-loop helpers (grp_of_chk/slot_of_chk/ecm_bank/hb_of_edge/ecm_port). See print_banking
    // header comment + hw/bp_relay_banked.sv post-mortem.
    println!(
        "localparam int BP_CHK_GRP [BP_C] = '{{{}}};",
        ints(&b.chk_grp)
    );
    println!(
        "localparam int BP_CHK_SLOT [BP_C] = '{{{}}};",
        ints(&b.chk_slot)
    );
    println!(
        "localparam int BP_VAR_GRP [BP_N] = '{{{}}};",
        ints(&b.var_grp)
    );
    println!(
        "localparam int BP_VAR_SLOT [BP_N] = '{{{}}};",
        ints(&b.var_slot)
    );
    println!(
        "localparam int BP_EDGE_HB [BP_E] = '{{{}}};",
        ints(&b.edge_hb)
    );
    println!(
        "localparam int BP_EDGE_EB [BP_E] = '{{{}}};",
        ints(&b.edge_eb)
    );
    println!(
        "localparam int BP_EDGE_ROW [BP_E] = '{{{}}};",
        ints(&b.edge_row)
    );
    println!(
        "localparam int BP_EDGE_EPORT [BP_E] = '{{{}}};",
        ints(&b.edge_eport)
    );
}

/// LSB-first packed bit row for the BRAM-ROM literal emission ([`print_rom_rows`]): the Rust twin of
/// one SystemVerilog packed row word, written field-by-field with the same `[off +: width]` layout the
/// RTL slices with.
struct RomRow {
    bits: Vec<bool>,
}

impl RomRow {
    fn new(width: usize) -> Self {
        assert!(
            width > 0,
            "RomRow: zero-width row (degenerate graph parameter)"
        );
        RomRow {
            bits: vec![false; width],
        }
    }

    /// Write `width` bits of `val` at bit offset `off`, LSB-first — mirrors SV `row[off +: width] = val`.
    fn set(&mut self, off: usize, width: usize, val: u64) {
        debug_assert!(
            width == 64 || val < (1u64 << width),
            "RomRow: value overflows field"
        );
        for i in 0..width {
            self.bits[off + i] = (val >> i) & 1 == 1;
        }
    }

    /// `<W>'h<hex>` SystemVerilog literal (MSB-first hex nibbles, exact `ceil(W/4)` digits).
    fn lit(&self) -> String {
        let w = self.bits.len();
        let nnib = w.div_ceil(4);
        let mut s = format!("{w}'h");
        for nib in (0..nnib).rev() {
            let mut val = 0u32;
            for j in 0..4 {
                let idx = nib * 4 + j;
                if idx < w && self.bits[idx] {
                    val |= 1 << j;
                }
            }
            s.push(char::from_digit(val, 16).expect("nibble < 16"));
        }
        s
    }
}

/// One `localparam logic [W-1:0] NAME [DEPTH] = '{ ... };` table, one hex row literal per line.
fn emit_rom_table(name: &str, depth_expr: &str, rows: &[RomRow]) {
    let wbits = rows[0].bits.len();
    let body = rows
        .iter()
        .map(|r| format!("  {}", r.lit()))
        .collect::<Vec<_>>()
        .join(",\n");
    println!(
        "localparam logic [{}:0] {name} [{depth_expr}] = '{{\n{body}}};",
        wbits - 1
    );
}

/// M9b (Q7-04) Task-4 probe rescue: emit the 14 BRAM-ROM row tables of `hw/bp_relay_banked_bram.sv`
/// as PACKED-ROW HEX LITERALS, so the RTL's ROM fills become pure literal copies with ZERO content
/// computation at elaboration. Probe runs 2-3 showed Vivado's ELABORATION of the computed `initial`
/// fills peaking ~50 GB RSS *independent of `rom_style`* — the cost was the fill computation, not the
/// ROM mapping — so per the 12c house lesson ALL content computation moves here (the offline emitter)
/// and the RTL only parses literals.
///
/// Probe run 4 narrowed the residual whale to ROM outputs used as RUNTIME INDICES into BP_C/BP_N-scale
/// register arrays (Vivado port-splits those instead of folding — M9a ops lesson), so the former
/// sbit-select tables (`CHK_SBIT_IDX/PRES`, indexing `s_reg[BP_C]`) and the S_EMIT observable tables
/// (`OBS_PRES/VAR/MASK`, indexing `ehat`/`best_e`/`corr_out[BP_N]`) are GONE — those sites reverted to
/// the LUT core's constant-folded `pc==g` form in the RTL. A ROM here feeds only data paths, memory
/// addresses/write-enables, or bank-output bus selects.
///
/// PACKING CONTRACT (must byte-match `bp_relay_banked_bram.sv`): slot `t`'s field of width `FW`
/// occupies row bits `[t*FW +: FW]`, slots being `(j,k)` lane (`j*BP_CHK_DEG+k`), var-slot `i`, or
/// `(i,d)` slot (`i*BP_VAR_DEG+d`); field widths are the RTL's `$clog2` localparams (`BWC/HBW/EBW`)
/// recomputed here by the same ceil-log2. Empty slots are all-zero fields, and present/eport bits are
/// 1-bit fields — exactly the old RTL fill expressions. The RTL's `rom_contract` elaboration guard
/// (simulation-only) recomputes EVERY row the old way and `$fatal`s on any mismatch, so this emitter
/// and the RTL slicing cannot silently disagree (M7 discipline).
fn print_rom_rows(view: &FixedHwView, b: &Banking) {
    let chk_deg = view
        .check_off
        .windows(2)
        .map(|w| w[1] - w[0])
        .max()
        .unwrap_or(0) as usize;
    let var_deg = view
        .var_off
        .windows(2)
        .map(|w| w[1] - w[0])
        .max()
        .unwrap_or(0) as usize;
    let (w, vcap, gc, gv) = (b.w, b.v, b.gc, b.gv);
    let neb = w * chk_deg; // e_cm banks   = (j,k) lanes
    let nvb = vcap * var_deg; // m_vm banks = (i,d) slots
    let nhb = 2 * neb; // m_cm half-banks
                       // SystemVerilog `$clog2` twin (ceil(log2 n); 0 for n <= 1) — widths must match the RTL exactly.
    let clog2 = |n: usize| -> usize {
        if n <= 1 {
            0
        } else {
            (usize::BITS - (n - 1).leading_zeros()) as usize
        }
    };
    let bwc = clog2(gc); // m_cm / e_cm row address width
    let hbw = clog2(nhb); // half-bank index width
    let ebw = clog2(neb); // e_cm bank index width
    let msg = view.msg_bits as usize;
    let mask = |val: i64, width: usize| -> u64 { (val as u64) & ((1u64 << width) - 1) };

    // ---- CHK-side rows (depth GC): gather half-bank taps + e_cm write present mask.
    let mut chk_hbsel = Vec::with_capacity(gc);
    let mut chk_epres = Vec::with_capacity(gc);
    let mut ecm_wpres = Vec::with_capacity(gc);
    for g in 0..gc {
        let mut r_hbsel = RomRow::new(neb * hbw);
        let mut r_epres = RomRow::new(neb);
        for j in 0..w {
            let c = b.chk_at[g * w + j];
            if c >= 0 {
                let c = c as usize;
                let deg = (view.check_off[c + 1] - view.check_off[c]) as usize;
                for k in 0..chk_deg.min(deg) {
                    let e = view.check_edges[view.check_off[c] as usize + k] as usize;
                    r_epres.set(j * chk_deg + k, 1, 1);
                    r_hbsel.set((j * chk_deg + k) * hbw, hbw, b.edge_hb[e] as u64);
                }
            }
        }
        // ecm_wpres[g] is bit-identical to chk_epres[g] (both = "lane (j,k) has a real edge"); the RTL
        // keeps them as two ROMs (different read addresses), so emit both.
        let mut r_wpres = RomRow::new(neb);
        r_wpres.bits.copy_from_slice(&r_epres.bits);
        chk_hbsel.push(r_hbsel);
        chk_epres.push(r_epres);
        ecm_wpres.push(r_wpres);
    }

    // ---- VAR-side rows (depth GV, gamma depth LEGS*GV): gather + scatter.
    let mut var_pres = Vec::with_capacity(gv);
    let mut var_lam = Vec::with_capacity(gv);
    let mut var_epres = Vec::with_capacity(gv);
    let mut var_ebsel = Vec::with_capacity(gv);
    let mut var_eport = Vec::with_capacity(gv);
    let mut var_erow = Vec::with_capacity(gv);
    let mut scat_pres = Vec::with_capacity(gv);
    let mut scat_hb = Vec::with_capacity(gv);
    let mut scat_row = Vec::with_capacity(gv);
    let mut scat_lam = Vec::with_capacity(gv);
    let mut var_gam: Vec<RomRow> = (0..view.legs * gv)
        .map(|_| RomRow::new(vcap * msg))
        .collect();
    for g in 0..gv {
        let mut r_pres = RomRow::new(vcap);
        let mut r_lam = RomRow::new(vcap * msg);
        let mut r_epres = RomRow::new(nvb);
        let mut r_ebsel = RomRow::new(nvb * ebw);
        let mut r_eport = RomRow::new(nvb);
        let mut r_erow = RomRow::new(nvb * bwc);
        let mut r_spres = RomRow::new(nvb);
        let mut r_shb = RomRow::new(nvb * hbw);
        let mut r_srow = RomRow::new(nvb * bwc);
        let mut r_slam = RomRow::new(nvb * msg);
        for i in 0..vcap {
            let var = b.var_at[g * vcap + i];
            if var >= 0 {
                let var = var as usize;
                r_pres.set(i, 1, 1);
                r_lam.set(i * msg, msg, mask(view.lambda_q[var] as i64, msg));
                for (l, gam_leg) in view.gamma_q.iter().enumerate() {
                    var_gam[l * gv + g].set(i * msg, msg, mask(gam_leg[var] as i64, msg));
                }
                let deg = (view.var_off[var + 1] - view.var_off[var]) as usize;
                for d in 0..var_deg.min(deg) {
                    let e = view.var_off[var] as usize + d;
                    let s = i * var_deg + d;
                    r_epres.set(s, 1, 1);
                    r_ebsel.set(s * ebw, ebw, b.edge_eb[e] as u64);
                    r_eport.set(s, 1, u64::from(b.edge_eport[e] == 1));
                    r_erow.set(s * bwc, bwc, b.edge_row[e] as u64);
                    r_spres.set(s, 1, 1);
                    r_shb.set(s * hbw, hbw, b.edge_hb[e] as u64);
                    r_srow.set(s * bwc, bwc, b.edge_row[e] as u64);
                    r_slam.set(
                        s * msg,
                        msg,
                        mask(view.lambda_q[view.edge_var[e] as usize] as i64, msg),
                    );
                }
            }
        }
        var_pres.push(r_pres);
        var_lam.push(r_lam);
        var_epres.push(r_epres);
        var_ebsel.push(r_ebsel);
        var_eport.push(r_eport);
        var_erow.push(r_erow);
        scat_pres.push(r_spres);
        scat_hb.push(r_shb);
        scat_row.push(r_srow);
        scat_lam.push(r_slam);
    }

    println!();
    println!(
        "// ---- M9b BRAM-ROM row literals for bp_relay_banked_bram (GENERATED; packing contract: \
         slot t's width-FW field at row bits [t*FW +: FW], widths BWC={bwc} HBW={hbw} EBW={ebw} \
         MSG={msg}). Data / memory-address / bank-output-select consumers ONLY (probe run 4 rule: \
         never a runtime index into a BP_C/BP_N-scale register array). The core's `rom_contract` \
         guard recomputes every row from the graph tables above and $fatal's on mismatch. Ignored by \
         every other core. ----"
    );
    println!("/* verilator lint_off UNUSEDPARAM */");
    emit_rom_table("BP_ROM_CHK_HBSEL", "BP_GC", &chk_hbsel);
    emit_rom_table("BP_ROM_CHK_EPRES", "BP_GC", &chk_epres);
    emit_rom_table("BP_ROM_ECM_WPRES", "BP_GC", &ecm_wpres);
    emit_rom_table("BP_ROM_VAR_PRES", "BP_GV", &var_pres);
    emit_rom_table("BP_ROM_VAR_LAM", "BP_GV", &var_lam);
    emit_rom_table("BP_ROM_VAR_GAM", "BP_LEGS*BP_GV", &var_gam);
    emit_rom_table("BP_ROM_VAR_EPRES", "BP_GV", &var_epres);
    emit_rom_table("BP_ROM_VAR_EBSEL", "BP_GV", &var_ebsel);
    emit_rom_table("BP_ROM_VAR_EPORT", "BP_GV", &var_eport);
    emit_rom_table("BP_ROM_VAR_EROW", "BP_GV", &var_erow);
    emit_rom_table("BP_ROM_SCAT_PRES", "BP_GV", &scat_pres);
    emit_rom_table("BP_ROM_SCAT_HB", "BP_GV", &scat_hb);
    emit_rom_table("BP_ROM_SCAT_ROW", "BP_GV", &scat_row);
    emit_rom_table("BP_ROM_SCAT_LAM", "BP_GV", &scat_lam);
    println!("/* verilator lint_on UNUSEDPARAM */");
}

/// Full-decode golden vectors: `T` sampled syndromes with the chosen `ehat`, observable flips, and
/// validity from `FixedRelayBp::decode_fixed_ehat` — for the M2 FSM testbench to match bit-for-bit.
fn emit_dec_vectors() {
    use aleph_qec::Syndrome;
    let dem = BBCode::gross().code_capacity_dem(0.03);
    let fx = FixedRelayBp::with_budget(&dem, LEGS, ITERS, GAMMA, SEED, MSG_BITS, FRAC_BITS);
    let n_vars = dem.errors.len();
    let n_checks = dem.detectors;
    let n_obs = 12usize;
    // Parity-reducing detector columns per variable (same reduction BpDecoder does).
    let cols: Vec<Vec<u32>> = dem.errors.iter().map(|e| e.dets.clone()).collect();

    // Test set: the empty syndrome, every single-variable error, then random low-weight errors.
    let mut z = 0xFEED_FACE_C0DE_0001u64;
    let mut next = || {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut x = z;
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    };

    let mut fired_sets: Vec<Vec<bool>> = Vec::new();
    fired_sets.push(vec![false; n_vars]); // empty
    for v in 0..24usize.min(n_vars) {
        let mut f = vec![false; n_vars];
        f[v] = true;
        fired_sets.push(f);
    }
    for _ in 0..40 {
        let mut f = vec![false; n_vars];
        for slot in f.iter_mut() {
            if next() % 30 == 0 {
                *slot = true;
            }
        }
        fired_sets.push(f);
    }
    let t = fired_sets.len();

    println!("# full-decode golden vectors — GENERATED, do not edit.");
    println!("# regenerate: cargo run -p aleph-qec --example qec_q7_bp_graph -- decvectors > hw/bp_dec_vectors.txt");
    println!("# format: header 'T BP_N BP_C BP_OBS'; per test: 's'(BP_C bits) 'h'(BP_N bits ehat) 'o'(BP_OBS bits) 'v'(valid)");
    println!("{t} {n_vars} {n_checks} {n_obs}");
    for f in &fired_sets {
        // Build the syndrome by XOR-ing flipped variables' detector columns (with parity).
        let mut lit = vec![false; n_checks];
        for (v, &fired) in f.iter().enumerate() {
            if fired {
                for &d in &cols[v] {
                    if (d as usize) < n_checks {
                        lit[d as usize] ^= true;
                    }
                }
            }
        }
        let syn = Syndrome::from_bits(&lit);
        let (ehat, obs, valid) = fx.decode_fixed_ehat(&syn);

        let s_str: String = lit.iter().map(|&b| if b { '1' } else { '0' }).collect();
        let h_str: String = ehat.iter().map(|&b| char::from(b'0' + (b & 1))).collect();
        let o_str: String = obs.iter().map(|&b| if b { '1' } else { '0' }).collect();
        println!("s {s_str}");
        println!("h {h_str}");
        println!("o {o_str}");
        println!("v {}", u8::from(valid));
    }
}

/// Full-decode golden vectors for the **circuit-level** DEM, sampled from **real DEM shots** (realistic
/// gate-noise syndromes, not synthetic low-weight errors), with the golden `ehat`/obs/validity from the
/// same `FixedRelayBp` the RTL implements. The M2 sequential decoder (graph-generic) decodes these
/// bit-for-bit in Verilator — the sim↔RTL co-sim proving the decoder generalises past code capacity.
fn emit_circ_vectors(rounds: usize, p: f64, n: usize, seed: u64, early: bool) {
    use aleph_qec::sample_shots;
    let (dem, fx) = build(Some((rounds, p)));
    let fx = fx.with_early_exit(early);
    let n_checks = dem.detectors;
    let n_vars = dem.errors.len();
    let n_obs = dem.observables;
    let (syndromes, _truths) = sample_shots(&dem, n as u64, seed);

    let mode = if early { "early-exit" } else { "full-decode" };
    let modearg = if early {
        "circvectorsearly"
    } else {
        "circvectors"
    };
    println!("# CIRCUIT-LEVEL {mode} golden vectors (depth-7, rounds={rounds}, p={p}) — GENERATED, do not edit.");
    println!("# regenerate: cargo run -p aleph-qec --example qec_q7_bp_graph -- {modearg} {rounds} {p} {n} {seed} > hw/bp_circ_vectors.txt");
    println!("# format: header 'T BP_N BP_C BP_OBS'; per test: 's'(BP_C bits) 'h'(BP_N bits ehat) 'o'(BP_OBS bits) 'v'(valid)");
    println!("{} {n_vars} {n_checks} {n_obs}", syndromes.len());
    for syn in &syndromes {
        let mut lit = vec![false; n_checks];
        for &d in &syn.fired {
            if (d as usize) < n_checks {
                lit[d as usize] = true;
            }
        }
        let (ehat, obs, valid) = fx.decode_fixed_ehat(syn);
        let s_str: String = lit.iter().map(|&b| if b { '1' } else { '0' }).collect();
        let h_str: String = ehat.iter().map(|&b| char::from(b'0' + (b & 1))).collect();
        let o_str: String = obs.iter().map(|&b| if b { '1' } else { '0' }).collect();
        println!("s {s_str}");
        println!("h {h_str}");
        println!("o {o_str}");
        println!("v {}", u8::from(valid));
    }
}

fn emit_vectors() {
    let dem = BBCode::gross().code_capacity_dem(0.03);
    let fx = FixedRelayBp::with_budget(&dem, LEGS, ITERS, GAMMA, SEED, MSG_BITS, FRAC_BITS);
    let v = fx.hw_view();
    let (n_checks, n_edges, max_mag) = (v.n_checks, v.n_edges, v.max_mag);

    const T: usize = 256;
    // Deterministic SplitMix64 stream for reproducible vectors.
    let mut z = 0x00C0_FFEE_1234_5678u64;
    let mut next = || {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut x = z;
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    };

    println!("# check-update test vectors — GENERATED, do not edit.");
    println!("# regenerate: cargo run -p aleph-qec --example qec_q7_bp_graph -- vectors > hw/bp_check_vectors.txt");
    println!("# format: header 'T BP_C BP_E'; then per test: line 's' (BP_C bits), line 'm' (BP_E ints), line 'e' (BP_E ints)");
    println!("{T} {n_checks} {n_edges}");
    for _ in 0..T {
        // Random syndrome bits.
        let s_bits: Vec<u8> = (0..n_checks).map(|_| (next() & 1) as u8).collect();
        // Random variable→check messages spanning the full representable signed range [-MAX_MAG, MAX_MAG].
        let span = (2 * max_mag + 1) as u64;
        let m_vc: Vec<i32> = (0..n_edges)
            .map(|_| (next() % span) as i32 - max_mag)
            .collect();
        let e_cv = fx.check_update_once(&m_vc, &s_bits);

        let s_str: String = s_bits.iter().map(|b| char::from(b'0' + b)).collect();
        let m_str = m_vc
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        let e_str = e_cv
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        println!("s {s_str}");
        println!("m {m_str}");
        println!("e {e_str}");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("graph");
    let rounds = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1usize);
    let p = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.003f64);
    let n = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(40usize);
    let seed = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(2024u64);
    match mode {
        "graph" => emit_graph(),
        "vectors" => emit_vectors(),
        "decvectors" => emit_dec_vectors(),
        "circgraph" => {
            // circgraph reuses positional args 4/5 as (bankW, bankV) for the offline banking solve,
            // not (n, seed) like the vector-emitting modes below.
            let bank_w = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(8usize);
            let bank_v = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(24usize);
            emit_circ_graph(rounds, p, bank_w, bank_v);
        }
        "circvectors" => emit_circ_vectors(rounds, p, n, seed, false),
        "circvectorsearly" => emit_circ_vectors(rounds, p, n, seed, true),
        "wingraph" => {
            // wingraph rounds p W C bankW bankV — positions 4/5 are (W, C) here (not bankW/bankV
            // like circgraph's 4/5), so bankW/bankV move to fresh positions 6/7.
            let w = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(6usize);
            let c = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(3usize);
            let bank_w = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(8usize);
            let bank_v = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(24usize);
            emit_win_graph(rounds, p, w, c, bank_w, bank_v);
        }
        "streamgraph" => {
            // streamgraph rounds p W C bankW bankV — same arg layout as wingraph (see above).
            let w = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(6usize);
            let c = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(2usize);
            let bank_w = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(8usize);
            let bank_v = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(24usize);
            emit_stream_graph(rounds, p, w, c, bank_w, bank_v);
        }
        "streamvectors" | "streamvectorsearly" => {
            // streamvectors[early] rounds p W C n seed — positions 4/5 are (W, C) like
            // streamgraph/wingraph, so (n, seed) move to fresh positions 6/7 (not the generic
            // 4/5 above).
            let w = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(6usize);
            let c = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(2usize);
            let sn = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(40usize);
            let sseed = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(7u64);
            emit_stream_vectors(rounds, p, w, c, sn, sseed, mode == "streamvectorsearly");
        }
        other => {
            eprintln!(
                "unknown mode '{other}'; use graph|vectors|decvectors|circgraph|circvectors|\
                 circvectorsearly|wingraph|streamgraph|streamvectors|streamvectorsearly"
            );
            std::process::exit(2);
        }
    }
}
