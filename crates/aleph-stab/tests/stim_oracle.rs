//! Oracle: stabilizer group equivalence vs Stim on random Clifford
//! circuits. Requires Python + `stim` on PATH; gated `#[ignore]` so the
//! default `cargo test` (and CI without stim) skips it. Run explicitly:
//!
//!   cargo test -p aleph-stab --test stim_oracle -- --ignored
//!
//! Comparison is by *canonical* stabilizer group (Stim's
//! `canonical_stabilizers()`), not raw generator rows — generator choice
//! is non-unique; the group is the invariant.

use aleph_core::{Gate, GateInstance};
use aleph_stab::{apply_gate, Tableau};
use std::process::Command;

const N: usize = 12;
const DEPTH: usize = 30;
const CIRCUITS: usize = 100;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// One random circuit as a list of (gate, qubits), shared by both sides.
fn random_circuit(seed: u64) -> Vec<GateInstance> {
    let mut rng = Rng(seed | 1);
    let mut out = Vec::new();
    for _ in 0..DEPTH {
        for _ in 0..N {
            let q = rng.below(N as u64) as u32;
            match rng.below(7) {
                0 => out.push(GateInstance::new(Gate::H, vec![q])),
                1 => out.push(GateInstance::new(Gate::S, vec![q])),
                2 => out.push(GateInstance::new(Gate::X, vec![q])),
                3 => out.push(GateInstance::new(Gate::Y, vec![q])),
                4 => out.push(GateInstance::new(Gate::Z, vec![q])),
                _ => {
                    let a = q;
                    let mut b = rng.below(N as u64) as u32;
                    if a == b {
                        b = (b + 1) % N as u32;
                    }
                    out.push(GateInstance::new(Gate::Cnot, vec![a, b]));
                }
            }
        }
    }
    out
}

/// Our canonical stabilizer group as a sorted Vec of signed Pauli
/// strings, using the same text format Stim emits ("+XZ_Y" style).
fn ours_canonical(circ: &[GateInstance]) -> Vec<String> {
    let mut t = Tableau::new(N);
    for g in circ {
        apply_gate(&mut t, g).unwrap();
    }
    // Reduce to canonical (RREF) form to match Stim. For P3-01 we lean on
    // Stim's canonicalization on its side and canonicalize ours by the
    // same algorithm in Python (see emit below): simplest robust path is
    // to hand BOTH the raw circuit to Python and let Stim build the
    // reference, while we send our generators for the Python script to
    // canonicalize identically. To avoid duplicating RREF in Rust, we
    // instead compare against Stim's tableau built from the SAME circuit
    // and rely on Stim canonical form on both — see python script.
    t.stabilizers()
        .iter()
        .map(|p| {
            let mut chars = vec![b'_'; N];
            for (q, pauli) in &p.terms {
                chars[*q as usize] = match pauli {
                    aleph_core::Pauli::I => b'_',
                    aleph_core::Pauli::X => b'X',
                    aleph_core::Pauli::Y => b'Y',
                    aleph_core::Pauli::Z => b'Z',
                };
            }
            let sign = if p.coefficient < 0.0 { '-' } else { '+' };
            format!("{sign}{}", String::from_utf8(chars).unwrap())
        })
        .collect()
}

/// Encode the circuit as a Stim program string.
fn stim_program(circ: &[GateInstance]) -> String {
    let mut s = String::new();
    for g in circ {
        let q = &g.qubits;
        match g.gate {
            Gate::H => s.push_str(&format!("H {}\n", q[0])),
            Gate::S => s.push_str(&format!("S {}\n", q[0])),
            Gate::X => s.push_str(&format!("X {}\n", q[0])),
            Gate::Y => s.push_str(&format!("Y {}\n", q[0])),
            Gate::Z => s.push_str(&format!("Z {}\n", q[0])),
            Gate::Cnot => s.push_str(&format!("CX {} {}\n", q[0], q[1])),
            _ => unreachable!("oracle circuits only use H/S/Paulis/CX"),
        }
    }
    s
}

/// Run the python helper; return `(reference, ours)` canonical generator
/// lists (one per line, "+XZ_Y" format). Both sides are routed through
/// the SAME canonicalizer so the comparison depends only on the
/// stabilizer *group*, not the generator choice.
fn stim_canonical(circ: &[GateInstance], ours: &[String]) -> Option<(Vec<String>, Vec<String>)> {
    // Pinned to the stim 1.16 API: `stim.Tableau.to_stabilizers(...)`
    // (the older `Tableau.stabilizers` was removed). `from_stabilizers`
    // takes an n-element list of independent commuting generators — which
    // a well-formed tableau always supplies — so a malformed group on our
    // side surfaces as a hard error here (a real test failure).
    let py = r#"
import sys, stim
data = sys.stdin.read().split("---\n")
prog = data[0]
ours = [l for l in data[1].splitlines() if l]
# Reference: evolve the circuit, take its canonical stabilizers, then
# re-canonicalize through a Tableau so both sides use one canonical form.
sim = stim.TableauSimulator()
sim.do(stim.Circuit(prog))
ref_ps = sim.canonical_stabilizers()
ref_canon = stim.Tableau.from_stabilizers(
    ref_ps, allow_redundant=False, allow_underconstrained=False
).to_stabilizers(canonicalize=True)
# Ours: parse our generators and canonicalize identically.
ours_ps = [stim.PauliString(s) for s in ours]
ours_canon = stim.Tableau.from_stabilizers(
    ours_ps, allow_redundant=False, allow_underconstrained=False
).to_stabilizers(canonicalize=True)
print("\n".join(str(p) for p in ref_canon))
print("===")
print("\n".join(str(p) for p in ours_canon))
"#;
    let mut input = stim_program(circ);
    input.push_str("---\n");
    input.push_str(&ours.join("\n"));
    let out = Command::new("python3")
        .arg("-c")
        .arg(py)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .ok()
        .and_then(|mut child| {
            use std::io::Write;
            child.stdin.take()?.write_all(input.as_bytes()).ok()?;
            child.wait_with_output().ok()
        })?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut parts = text.split("===");
    let refs: Vec<String> = parts
        .next()?
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let ours_c: Vec<String> = parts
        .next()?
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    Some((refs, ours_c))
}

#[test]
#[ignore = "requires python3 + stim; run on the EPYC oracle venv"]
fn matches_stim_on_random_cliffords() {
    let mut failures = 0;
    for k in 0..CIRCUITS {
        let circ = random_circuit(0xABCDEF ^ (k as u64).wrapping_mul(0x100000001B3));
        let ours = ours_canonical(&circ);
        let (refs, ours_c) = match stim_canonical(&circ, &ours) {
            Some(v) => v,
            None => panic!("stim helper failed (is `stim` installed in the active python3?)"),
        };
        let mut a = refs.clone();
        let mut b = ours_c.clone();
        a.sort();
        b.sort();
        if a != b {
            failures += 1;
            eprintln!("circuit {k} mismatch:\n  stim: {a:?}\n  ours: {b:?}");
        }
    }
    assert_eq!(
        failures, 0,
        "{failures}/{CIRCUITS} circuits disagreed with Stim"
    );
}
