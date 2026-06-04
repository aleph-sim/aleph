//! Oracle: measurement + collapse equivalence vs Stim. For each random
//! Clifford circuit and target qubit, we measure in our tableau (outcome
//! `b`), then postselect Stim's qubit to `b` and compare post-measurement
//! canonical stabilizer groups. Also cross-checks determinism via Stim
//! `peek_z`. Requires python3 + stim; `#[ignore]`d (run on EPYC):
//!
//!   cargo test -p aleph-stab --test stim_measure_oracle -- --ignored
//!
//! Group comparison is sign-and-generator canonical (sorted set), not
//! row-order sensitive.

use aleph_core::{Gate, GateInstance};
use aleph_stab::{apply_gate, Tableau};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::process::Command;

const N: usize = 10;
const DEPTH: usize = 25;
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

/// Our post-measurement stabilizer generators in stim "+XZ_Y" format.
fn ours_generators(t: &Tableau) -> Vec<String> {
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

/// Returns `(peek, ref_canon, ours_canon)`: Stim's `peek_z(a)` (+1/-1/0),
/// the reference canonical stabilizers after postselecting `a→b`, and our
/// canonical generators. `None` if the helper failed to run.
fn stim_check(
    circ: &[GateInstance],
    a: usize,
    b: bool,
    ours: &[String],
) -> Option<(i64, Vec<String>, Vec<String>)> {
    // Pinned to stim 1.16. peek_z is read BEFORE postselect (peek does not
    // collapse). We always postselect to OUR outcome b, which has nonzero
    // probability in the same state, so postselect never rejects.
    let py = r#"
import sys, stim
data = sys.stdin.read().split("---\n")
prog = data[0]
meta = data[1].splitlines()
a, b = meta[0].split()
a = int(a); b = (b == "1")
ours = [l for l in meta[1:] if l]
sim = stim.TableauSimulator()
sim.do(stim.Circuit(prog))
peek = sim.peek_z(a)
sim.postselect_z(a, desired_value=b)
ref_canon = stim.Tableau.from_stabilizers(
    sim.canonical_stabilizers(), allow_redundant=False, allow_underconstrained=False
).to_stabilizers(canonicalize=True)
ours_canon = stim.Tableau.from_stabilizers(
    [stim.PauliString(s) for s in ours], allow_redundant=False, allow_underconstrained=False
).to_stabilizers(canonicalize=True)
print(peek)
print("===")
print("\n".join(str(p) for p in ref_canon))
print("===")
print("\n".join(str(p) for p in ours_canon))
"#;
    let mut input = stim_program(circ);
    input.push_str("---\n");
    input.push_str(&format!("{a} {}\n", if b { 1 } else { 0 }));
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
    let peek: i64 = parts.next()?.trim().parse().ok()?;
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
    Some((peek, refs, ours_c))
}

#[test]
#[ignore = "requires python3 + stim; run on the EPYC oracle venv"]
fn measurement_matches_stim() {
    let mut failures = 0;
    for k in 0..CIRCUITS {
        let circ = random_circuit(0xBEEF ^ (k as u64).wrapping_mul(0x100000001B3));
        let a = (k * 7) % N; // vary the measured qubit
        let mut rng = StdRng::seed_from_u64(0xC0FFEE ^ k as u64);

        let mut t = Tableau::new(N);
        for g in &circ {
            apply_gate(&mut t, g).unwrap();
        }
        let b = t.measure(a, &mut rng).unwrap();
        let ours = ours_generators(&t);

        let (peek, refs, ours_c) = match stim_check(&circ, a, b, &ours) {
            Some(v) => v,
            None => panic!("stim helper failed (is `stim` installed in the active python3?)"),
        };

        // Determinism cross-check: peek_z == +1 → outcome must be 0(false);
        // -1 → 1(true); 0 → random (no constraint).
        if peek == 1 {
            assert!(
                !b,
                "circuit {k}: stim says deterministic |0> but we measured 1"
            );
        } else if peek == -1 {
            assert!(
                b,
                "circuit {k}: stim says deterministic |1> but we measured 0"
            );
        }

        let mut x = refs.clone();
        let mut y = ours_c.clone();
        x.sort();
        y.sort();
        if x != y {
            failures += 1;
            eprintln!(
                "circuit {k} (measure q{a}→{b}) post-state mismatch:\n  stim: {x:?}\n  ours: {y:?}"
            );
        }
    }
    assert_eq!(
        failures, 0,
        "{failures}/{CIRCUITS} circuits disagreed with Stim"
    );
}
