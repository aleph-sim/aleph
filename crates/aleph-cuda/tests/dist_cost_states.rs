//! #538 Stage A: every kernel kind's per-launch time on the six prepared
//! states of the spec (§2), FP64 at m_ref=27 and FP32 at 28, two passes and
//! their mean. Prints the measured state classes (rule 1), each candidate
//! rule's predictions (rule 3), the per-kind split (rule 2) and the
//! `KindTimes` literals (rule 4). It measures only: the rules are fixed in
//! the spec, and the human applies the printed decision in `src/dist/cost.rs`.
//! Replaces `dist_cost_calibrate.rs` (P6-05 PR 2).
//! Run (idle box): cargo test --release -p aleph-cuda --features cuda --test dist_cost_states -- --ignored --nocapture
#![cfg(all(target_os = "linux", feature = "cuda"))]

mod common;

use std::time::Instant;

use aleph_backend::run;
use aleph_cuda::{makes_generic_under, CudaContext, CudaSvBackend, CudaSvBackendF32, StateRule};
use aleph_ir::Circuit;
use common::calib::{kinds, prefix, State, KIND_NAMES, STATES};

/// Spec §2 thresholds.
const GENERIC_STATE: f64 = 1.05;
const SPLIT_KIND: f64 = 1.05;

type Table = [[f64; 9]; 6]; // [state][kind]

fn pass(n: u32, time: &mut dyn FnMut(&Circuit) -> f64) -> Table {
    STATES.map(|state| kinds(n, state, time))
}

fn mean(a: &Table, b: &Table) -> Table {
    let mut m = *a;
    for ((row, ra), rb) in m.iter_mut().zip(a).zip(b) {
        for ((x, &p), &q) in row.iter_mut().zip(ra).zip(rb) {
            *x = 0.5 * (p + q);
        }
    }
    m
}

fn predicted(rule: StateRule, state: State, n: u32) -> bool {
    prefix(state, n)
        .instructions()
        .iter()
        .any(|i| makes_generic_under(i, rule).unwrap())
}

/// Prints the tables, decisions and literal for one precision.
fn report(tag: &str, name: &str, n: u32, r1: &Table, r2: &Table) {
    let m = mean(r1, r2);
    println!("\n### {tag} (n = m_ref = {n}), ms per launch, mean of 2 (run1/run2)\n");
    println!("| kind | a | b | c | d | e | f | d/a |");
    println!("|---|---|---|---|---|---|---|---|");
    for k in 0..9 {
        let cells: Vec<String> = (0..6)
            .map(|s| {
                format!(
                    "{:.3} ({:.3}/{:.3})",
                    1e3 * m[s][k],
                    1e3 * r1[s][k],
                    1e3 * r2[s][k]
                )
            })
            .collect();
        println!(
            "| {} | {} | {:.3} |",
            KIND_NAMES[k],
            cells.join(" | "),
            m[3][k] / m[0][k]
        );
    }
    // Rule 1 (dense2 is kind index 1) and rule 3.
    let a2 = m[0][1];
    let measured: Vec<bool> = (0..6).map(|s| m[s][1] / a2 >= GENERIC_STATE).collect();
    println!("\n| state | dense2 / a | measured | R1 predicts | R2 predicts |");
    println!("|---|---|---|---|---|");
    let mut r1_ok = true;
    let mut r2_ok = true;
    for (s, &state) in STATES.iter().enumerate() {
        let (p1, p2) = (
            predicted(StateRule::R1, state, n),
            predicted(StateRule::R2, state, n),
        );
        r1_ok &= p1 == measured[s];
        r2_ok &= p2 == measured[s];
        let cls = |g: bool| if g { "generic" } else { "simple" };
        println!(
            "| {state:?} | {:.3} | {} | {} | {} |",
            m[s][1] / a2,
            cls(measured[s]),
            cls(p1),
            cls(p2)
        );
    }
    let rule = match (r1_ok, r2_ok) {
        (true, _) => "R1 (matches every state; R1 preferred on a tie)",
        (false, true) => "R2 (only R2 matches every state)",
        (false, false) => "NONE MATCHES -> STOP, decide with the user (spec rule 3)",
    };
    println!("{tag} rule 3: {rule}");
    // Rule 2 and rule 4.
    let mut fields = Vec::new();
    let mut gens = Vec::new();
    for k in 0..9 {
        let (a, d) = (m[0][k], m[3][k]);
        let verdict = if !a.is_finite() || a <= 0.0 || !d.is_finite() {
            "INVALID (one constant)".to_string()
        } else if d / a >= SPLIT_KIND {
            gens.push(format!("{}: Some({d:.6e})", KIND_NAMES[k]));
            format!("SPLIT (d/a = {:.3})", d / a)
        } else {
            format!("one constant (d/a = {:.3})", d / a)
        };
        println!("{tag} rule 2: {} → {verdict}", KIND_NAMES[k]);
        fields.push(format!("{}: {a:.6e}", KIND_NAMES[k]));
    }
    let generic = if gens.is_empty() {
        "GenericTimes::NONE".to_string()
    } else {
        format!(
            "GenericTimes {{ {}, ..GenericTimes::NONE }}",
            gens.join(", ")
        )
    };
    println!(
        "const {name}: KindTimes = KindTimes {{ m_ref: {n}, {}, generic: {generic} }};",
        fields.join(", ")
    );
}

#[test]
#[ignore]
fn state_microbench() {
    let Ok(sync) = CudaContext::new(0) else {
        eprintln!("skipped: no CUDA");
        return;
    };
    let Ok(mut b64) = CudaSvBackend::with_seed(0) else {
        eprintln!("skipped: no CUDA");
        return;
    };
    let mut t64 = |c: &Circuit| {
        sync.synchronize().unwrap();
        let t = Instant::now();
        let st = run(&mut b64, c).unwrap();
        sync.synchronize().unwrap();
        let s = t.elapsed().as_secs_f64();
        drop(st);
        s
    };
    let a = pass(27, &mut t64);
    let b = pass(27, &mut t64);
    report("FP64", "RTX4000_FP64", 27, &a, &b);
    let Ok(mut b32) = CudaSvBackendF32::with_seed(0) else {
        eprintln!("skipped: no CUDA FP32");
        return;
    };
    let mut t32 = |c: &Circuit| {
        sync.synchronize().unwrap();
        let t = Instant::now();
        let st = run(&mut b32, c).unwrap();
        sync.synchronize().unwrap();
        let s = t.elapsed().as_secs_f64();
        drop(st);
        s
    };
    let a = pass(28, &mut t32);
    let b = pass(28, &mut t32);
    report("FP32", "RTX4000_FP32", 28, &a, &b);
}
