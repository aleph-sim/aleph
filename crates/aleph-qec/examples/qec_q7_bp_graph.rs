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

use aleph_qec::{BBCode, FixedRelayBp};

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

fn emit_graph() {
    let dem = BBCode::gross().code_capacity_dem(0.03);
    let fx = FixedRelayBp::with_budget(&dem, LEGS, ITERS, GAMMA, SEED, MSG_BITS, FRAC_BITS);
    let v = fx.hw_view();

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

    println!("// Gross BB code [[144,12,12]] Tanner graph + fixed-point relay-BP params — GENERATED, do not edit.");
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
    println!("`endif");
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
    let mode = std::env::args().nth(1).unwrap_or_else(|| "graph".into());
    match mode.as_str() {
        "graph" => emit_graph(),
        "vectors" => emit_vectors(),
        "decvectors" => emit_dec_vectors(),
        other => {
            eprintln!("unknown mode '{other}'; use 'graph' or 'vectors'");
            std::process::exit(2);
        }
    }
}
