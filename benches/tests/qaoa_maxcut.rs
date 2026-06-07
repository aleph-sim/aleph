//! P4-05 CI gate: the QAOA Max-Cut pipeline produces a non-trivial p=1 optimum.
//! Deterministic grid search over (γ,β) on the committed n=6 3-regular graph,
//! computing ⟨H_C⟩ via build_qaoa + maxcut_pauli_sum + expectation_pauli_sum on
//! the state-vector backend. Asserts the grid-best approximation ratio clears a
//! robust p=1 bound (≥0.6; 3-regular p=1 lands ~0.7). This gates the Rust QAOA
//! pipeline in CI with no maturin/scipy. The ≥0.9@p=3 AC is gated by the Python
//! test_qaoa.py (COBYLA). Mirrors scripts/qaoa/qaoa.py's energy call.

use aleph_backend::{expectation_pauli_sum, run};
use aleph_ir::{build_qaoa, maxcut_pauli_sum};
use aleph_sv::NaiveSvBackend;
use std::f64::consts::PI;
use std::path::PathBuf;

fn graph_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("scripts/qaoa/graphs")
        .join(name)
}

fn load_edges(n: u32) -> Vec<(u32, u32)> {
    let txt = std::fs::read_to_string(graph_path(&format!("qaoa_n{n}.edges"))).unwrap();
    txt.lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .map(|l| {
            let mut it = l.split_whitespace();
            (
                it.next().unwrap().parse().unwrap(),
                it.next().unwrap().parse().unwrap(),
            )
        })
        .collect()
}

#[test]
fn qaoa_p1_grid_finds_nontrivial_cut() {
    let n = 6u32;
    let edges = load_edges(n);
    let maxcut: f64 = std::fs::read_to_string(graph_path("qaoa_n6.maxcut"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    let ham = maxcut_pauli_sum(n, &edges).unwrap();

    // Grid search over gamma in [0,pi), beta in [0,pi/2) (QAOA p=1 fundamental
    // domains for the Max-Cut cost/mixer).
    let steps = 48;
    let mut best = f64::NEG_INFINITY;
    for gi in 0..steps {
        let gamma = PI * gi as f64 / steps as f64;
        for bi in 0..steps {
            let beta = 0.5 * PI * bi as f64 / steps as f64;
            let circuit = build_qaoa(n, &edges, &[gamma], &[beta]).unwrap();
            let mut be = NaiveSvBackend::with_seed(0);
            let st = run(&mut be, &circuit).unwrap();
            let e = expectation_pauli_sum(&mut be, &st, &ham).unwrap();
            if e > best {
                best = e;
            }
        }
    }
    let ratio = best / maxcut;
    assert!(
        ratio >= 0.6,
        "p=1 grid best ⟨H_C⟩={best:.4}, max-cut={maxcut}, ratio={ratio:.4} < 0.6"
    );
}
