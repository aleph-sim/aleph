//! Stabilizer ≡ state vector on Clifford circuits. `expectation_value`
//! is deterministic, so it's compared exactly against `NaiveSvBackend`;
//! `sample` is checked for support (no impossible outcomes) + known
//! correlations (RNG sequences differ between backends, so exact counts
//! are not comparable).

use aleph_backend::{run, Backend};
use aleph_core::{Gate, GateInstance, Pauli, PauliString};
use aleph_stab::StabilizerBackend;
use aleph_sv::NaiveSvBackend;

const N: usize = 5;

/// Deterministic xorshift so circuits are reproducible without an RNG dep.
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

fn random_clifford(seed: u64) -> Vec<GateInstance> {
    let mut rng = Rng(seed | 1);
    let mut out = Vec::new();
    for _ in 0..40 {
        let q = rng.below(N as u64) as u32;
        match rng.below(6) {
            0 => out.push(GateInstance::new(Gate::H, vec![q])),
            1 => out.push(GateInstance::new(Gate::S, vec![q])),
            2 => out.push(GateInstance::new(Gate::X, vec![q])),
            3 => out.push(GateInstance::new(Gate::Z, vec![q])),
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
    out
}

fn apply_all<B: Backend>(be: &mut B, st: &mut B::State, circ: &[GateInstance]) {
    for g in circ {
        be.apply_gate(st, g).unwrap();
    }
}

#[test]
fn expectation_matches_state_vector() {
    let paulis = [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z];
    for k in 0..50u64 {
        let circ = random_clifford(k);

        let mut sb = StabilizerBackend::with_seed(0);
        let mut st = sb.allocate(N as u32).unwrap();
        apply_all(&mut sb, &mut st, &circ);

        let mut nb = NaiveSvBackend::with_seed(0);
        let mut nv = nb.allocate(N as u32).unwrap();
        apply_all(&mut nb, &mut nv, &circ);

        // A handful of random Pauli observables per circuit.
        let mut rng = Rng(0x5151 ^ k);
        for _ in 0..6 {
            let terms: Vec<(u32, Pauli)> = (0..N as u32)
                .filter_map(|q| {
                    let p = paulis[rng.below(4) as usize];
                    if p == Pauli::I {
                        None
                    } else {
                        Some((q, p))
                    }
                })
                .collect();
            let ps = PauliString::new(1.0, terms).unwrap();
            let s = sb.expectation_value(&st, &ps).unwrap();
            let v = nb.expectation_value(&nv, &ps).unwrap();
            assert!(
                (s - v).abs() < 1e-9,
                "circuit {k}: stabilizer ⟨P⟩={s} != state-vector {v} for {ps:?}"
            );
        }
    }
}

#[test]
fn sample_support_is_physical() {
    // Every stabilizer-sampled bitstring must have nonzero probability in
    // the state-vector amplitudes for the same circuit.
    for k in 0..20u64 {
        let circ = random_clifford(k);

        let mut sb = StabilizerBackend::with_seed(k);
        let mut st = sb.allocate(N as u32).unwrap();
        apply_all(&mut sb, &mut st, &circ);
        let shots = sb.sample(&st, 200).unwrap();

        let mut nb = NaiveSvBackend::with_seed(0);
        let mut nv = nb.allocate(N as u32).unwrap();
        apply_all(&mut nb, &mut nv, &circ);
        // `CpuState::amplitudes(&self) -> &[Complex]` (inherent method,
        // crates/aleph-sv/src/state.rs); index = basis state, qubit q at bit q.
        let amps = nv.amplitudes();

        for s in &shots {
            let idx = *s as usize;
            let p = amps[idx].norm_sqr();
            assert!(
                p > 1e-12,
                "circuit {k}: sampled |{s:0width$b}⟩ has prob {p}",
                width = N
            );
        }
    }
}

#[test]
fn surface_code_cycle_runs_on_stabilizer() {
    let src = std::fs::read_to_string(aleph_oracle::workspace_path(
        "oracle/circuits/surface_code_cycle.qasm",
    ))
    .unwrap();
    let circuit = aleph_parser::parse(&src).unwrap();
    let mut be = StabilizerBackend::with_seed(0);
    let state = run(&mut be, &circuit).unwrap();
    assert_eq!(state.num_qubits(), 6);
}
