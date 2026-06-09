//! P4-07 oracle: surface-code cycle post-state matches Stim. Run our cycle,
//! collect ancilla outcomes b[], postselect Stim's ancillas to b[] in the same
//! order, compare canonical stabilizer groups. Requires python3 + stim;
//! `#[ignore]`d (run on the EPYC oracle venv):
//!
//!   cargo test -p aleph-benches --test surface_code_stim_oracle -- --ignored
//!
//! Comparison is sign-and-generator canonical (sorted set), not row-order
//! sensitive. Mirrors crates/aleph-stab/tests/stim_measure_oracle.rs.

use aleph_backend::Backend;
use aleph_benches::{cycle_stim_gates, SurfaceCode};
use aleph_core::Pauli;
use aleph_stab::{StabilizerBackend, Tableau};
use std::process::Command;

/// Our post-cycle stabilizer generators in Stim "+XZ_Y" format.
fn ours_generators(t: &Tableau, n: usize) -> Vec<String> {
    t.stabilizers()
        .iter()
        .map(|p| {
            let mut chars = vec![b'_'; n];
            for (q, pauli) in &p.terms {
                chars[*q as usize] = match pauli {
                    Pauli::I => b'_',
                    Pauli::X => b'X',
                    Pauli::Y => b'Y',
                    Pauli::Z => b'Z',
                };
            }
            let sign = if p.coefficient < 0.0 { '-' } else { '+' };
            format!("{sign}{}", String::from_utf8(chars).unwrap())
        })
        .collect()
}

/// Run our cycle from |0…0⟩, return (ancilla outcomes in order, our generators).
fn run_ours(sc: &SurfaceCode, seed: u64) -> (Vec<bool>, Vec<String>) {
    let mut be = StabilizerBackend::with_seed(seed);
    let mut t = be.allocate(sc.num_qubits as u32).unwrap();
    for g in sc.cycle_gates() {
        be.apply_gate(&mut t, &g).unwrap();
    }
    let outcomes: Vec<bool> = sc
        .ancilla_order()
        .iter()
        .map(|&a| be.measure(&mut t, a).unwrap())
        .collect();
    let gens = ours_generators(&t, sc.num_qubits);
    (outcomes, gens)
}

/// Returns (ref_canon, ours_canon) or None if the helper failed.
fn stim_canonical(
    d: usize,
    order: &[u32],
    outcomes: &[bool],
    ours: &[String],
) -> Option<(Vec<String>, Vec<String>)> {
    // stdin layout: gates --- "<a0> <b0>\n<a1> <b1>\n…" --- ours generators.
    let py = r#"
import sys, stim
parts = sys.stdin.read().split("---\n")
prog = parts[0]
post = [l for l in parts[1].splitlines() if l]
ours = [l for l in parts[2].splitlines() if l]
sim = stim.TableauSimulator()
sim.do(stim.Circuit(prog))
for line in post:
    a, b = line.split()
    sim.postselect_z(int(a), desired_value=(b == "1"))
ref = stim.Tableau.from_stabilizers(
    sim.canonical_stabilizers(), allow_redundant=False, allow_underconstrained=False
).to_stabilizers(canonicalize=True)
oursc = stim.Tableau.from_stabilizers(
    [stim.PauliString(s) for s in ours], allow_redundant=False, allow_underconstrained=False
).to_stabilizers(canonicalize=True)
print("\n".join(str(p) for p in ref))
print("===")
print("\n".join(str(p) for p in oursc))
"#;
    let mut input = cycle_stim_gates(d);
    input.push_str("---\n");
    for (a, b) in order.iter().zip(outcomes) {
        input.push_str(&format!("{a} {}\n", if *b { 1 } else { 0 }));
    }
    input.push_str("---\n");
    input.push_str(&ours.join("\n"));

    let out = Command::new("python3")
        .arg("-c")
        .arg(py)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .ok()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take()?.write_all(input.as_bytes()).ok()?;
            child.wait_with_output().ok()
        })?;
    if !out.status.success() {
        panic!(
            "stim helper exited with failure for d={d}:\n--- stderr ---\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut it = text.split("===");
    let refs: Vec<String> = it
        .next()?
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let oursc: Vec<String> = it
        .next()?
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    Some((refs, oursc))
}

#[test]
#[ignore = "requires python3 + stim; run on the EPYC oracle venv"]
fn surface_cycle_matches_stim() {
    for d in [3usize, 5, 7, 9, 11] {
        let sc = SurfaceCode::new(d);
        let (outcomes, ours) = run_ours(&sc, 0xC0FFEE ^ d as u64);
        let order = sc.ancilla_order();
        let (refs, oursc) = match stim_canonical(d, &order, &outcomes, &ours) {
            Some(v) => v,
            None => panic!("could not run python3 for d={d} (is python3 on PATH?)"),
        };
        let mut a = refs.clone();
        let mut b = oursc.clone();
        a.sort();
        b.sort();
        assert_eq!(
            a, b,
            "d={d}: post-cycle stabilizer group disagrees with Stim"
        );
    }
}
