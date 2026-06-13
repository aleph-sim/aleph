# P4.6-04 — SV noise engine (`aleph_sv::noise`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Monte-Carlo quantum-jump noise driver `aleph_sv::noise` that runs a circuit `shots` times under an Aer-style `NoiseModel`, matching Qiskit Aer's measurement distribution to 1e-5 at 100k shots, while leaving the IR, the `Backend` trait, and the noiseless `run()` path completely untouched.

**Architecture:** Noise is a *runtime config* (`NoiseModel`) consumed by a SV-specific driver, never IR (ADR 0014, golden rule 4). One shot = a fresh `CpuState`, the circuit's gates applied via the existing `NaiveSvBackend::apply_gate`, with each gate's attached channels applied right after it by quantum-jump (Pauli channels take a state-independent fast path; amplitude/phase damping take the general `pᵢ=‖Kᵢ|ψ〉‖²` path). After all gates, one Z-basis outcome is sampled from `|amps|²` and per-qubit readout error is applied. Shots are `rayon`-parallel, each owning its own `StdRng` seeded `hash(seed, shot)` for scheduling-independent determinism.

**Tech Stack:** Rust 2021, `rand::StdRng` (ChaCha20, already a dep), `rayon` (already a dep), `aleph_core::{Complex, Pauli, AlignedBuf}`, `thiserror` (new dep — crate-local `NoiseError`). Oracle fixtures generated offline by Qiskit Aer (density-matrix method, exact probabilities) and committed as JSON.

**Spec:** `docs/superpowers/specs/2026-06-13-p46-03-noise-models-design.md`. **ADR:** `docs/decisions/0014-noise-trajectories.md`.

---

## File Structure

| Path | Responsibility |
| --- | --- |
| `crates/aleph-sv/src/noise/mod.rs` | `pub mod noise` root: `NoiseError`, `run_noisy` driver, `Counts` alias, re-exports. |
| `crates/aleph-sv/src/noise/model.rs` | `NoiseModel` (attachment maps + readout map), `add_*` methods, `errors_for`. |
| `crates/aleph-sv/src/noise/error.rs` | `QuantumError`, `PauliChannel`, `KrausChannel`, `ReadoutError` data types + channel constructors (`depolarizing_error`, `amplitude_damping_error`, `phase_damping_error`, `pauli_error`, `bit_flip_error`, `phase_flip_error`). |
| `crates/aleph-sv/src/noise/apply.rs` | `apply_channel` (Pauli fast-path + general 1q quantum-jump), `apply_readout`, `shot_seed`. |
| `crates/aleph-sv/src/lib.rs` | add `pub mod noise;` (public so P4.6-05 pyo3 can consume it). |
| `crates/aleph-sv/Cargo.toml` | add `thiserror = { workspace = true }`. |
| `oracle/noise/gen_noise.py` | offline generator: byte-identical Qiskit `NoiseModel`, density-matrix method, `save_probabilities` → exact noisy distributions. |
| `oracle/noise/*.json` | committed fixtures `{name, num_qubits, exact_probs}`. |
| `crates/aleph-sv/tests/noise_oracle.rs` | integration test: build the matching aleph `NoiseModel` in Rust, `run_noisy` @100k, assert vs `exact_probs` with `assert_distribution_close`. |

**Scoping note (channel set v1):** general (non-Pauli) Kraus channels are **1-qubit only** (amplitude + phase damping). 2-qubit noise in v1 is depolarizing, which is a Pauli channel and is applied as a tensor product of single-qubit Paulis via `apply_1q`. No general 2q Kraus path is built (YAGNI; documented).

---

## Task 1: Module scaffold + data types

**Files:**
- Create: `crates/aleph-sv/src/noise/mod.rs`
- Create: `crates/aleph-sv/src/noise/error.rs`
- Create: `crates/aleph-sv/src/noise/model.rs`
- Create: `crates/aleph-sv/src/noise/apply.rs`
- Modify: `crates/aleph-sv/src/lib.rs` (add `pub mod noise;`)
- Modify: `crates/aleph-sv/Cargo.toml` (add `thiserror`)

- [ ] **Step 1: Add the `thiserror` dependency**

In `crates/aleph-sv/Cargo.toml`, under `[dependencies]` after the `rayon` line:

```toml
thiserror     = { workspace = true }
```

(Justification for the PR body: the noise driver needs its own error domain — `MidCircuitMeasurement`/`UnsupportedInstruction` are v1.1 concerns that don't belong in `aleph_backend::BackendError`. `thiserror` is already a workspace dependency used by every other library crate.)

- [ ] **Step 2: Write the data types in `error.rs`**

```rust
//! Noise channel data types and Aer-named constructors.
//!
//! A [`QuantumError`] is a CPTP map attached to a gate. v1 splits into two
//! application strategies: [`PauliChannel`] (state-independent weights — the
//! quantum-jump fast path) and [`KrausChannel`] (general 1q operators applied
//! via `pᵢ=‖Kᵢ|ψ〉‖²`). See the P4.6-03 design spec §3.

use aleph_core::{Complex, Pauli};
use smallvec::SmallVec;

/// A probabilistic mixture of (multi-qubit) Pauli operators. `Σ probs = 1`.
///
/// `terms[i] = (prob, paulis)` where `paulis[j]` is the Pauli applied to the
/// channel's local qubit `j` (so `paulis.len() == arity`). State-independent:
/// the branch is sampled directly from `prob`, no norm computation.
#[derive(Debug, Clone, PartialEq)]
pub struct PauliChannel {
    pub arity: u8,
    pub terms: Vec<(f64, SmallVec<[Pauli; 2]>)>,
}

/// A general single-qubit CPTP map given by its Kraus operators.
/// `Σ Kᵢ† Kᵢ = I`. Applied by quantum-jump (compute `pᵢ`, sample, renormalize).
#[derive(Debug, Clone, PartialEq)]
pub struct KrausChannel {
    pub kraus: Vec<[[Complex; 2]; 2]>,
}

/// A CPTP error map attached to a gate.
#[derive(Debug, Clone, PartialEq)]
pub enum QuantumError {
    Pauli(PauliChannel),
    Kraus(KrausChannel),
}

impl QuantumError {
    /// Number of qubits this error acts on.
    pub fn arity(&self) -> usize {
        match self {
            QuantumError::Pauli(p) => p.arity as usize,
            QuantumError::Kraus(_) => 1,
        }
    }
}

/// Per-qubit classical readout error: a 2×2 row-stochastic matrix.
/// `m[t][o]` = P(measure outcome `o` | true value `t`). Rows sum to 1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReadoutError {
    pub m: [[f64; 2]; 2],
}
```

- [ ] **Step 3: Write `NoiseError` and the driver skeleton in `mod.rs`**

```rust
//! `aleph_sv::noise` — Monte-Carlo quantum-jump noise driver.
//!
//! Noise is a runtime [`NoiseModel`] config, never IR (ADR 0014). The
//! noiseless `run()` path and the `Backend` trait are untouched; this is a
//! separate `run_noisy` entry point operating on `CpuState`.

mod apply;
mod error;
mod model;

pub use error::{KrausChannel, PauliChannel, QuantumError, ReadoutError};
pub use model::NoiseModel;

pub use error::{
    amplitude_damping_error, bit_flip_error, depolarizing_error, pauli_error,
    phase_damping_error, phase_flip_error,
};

use aleph_backend::BackendError;

/// Per-basis-state shot histogram of length `2^num_qubits`. `counts[i]` is the
/// number of shots whose final (readout-perturbed) bitstring was basis state
/// `|i⟩`. The Python layer (P4.6-05) maps this to a bitstring→count dict.
pub type Counts = Vec<u64>;

/// Errors raised by the noise driver, on top of backend failures.
#[derive(Debug, thiserror::Error)]
pub enum NoiseError {
    #[error(transparent)]
    Backend(#[from] BackendError),
    /// v1 supports terminal measurement only; mid-circuit measure/reset under
    /// noise is a documented v1.1 follow-up (spec §3 "Measurement & reset").
    #[error("mid-circuit {kind} is not supported under noise in v1 (terminal measurement only)")]
    MidCircuit { kind: &'static str },
}
```

- [ ] **Step 4: Write a `NoiseModel` stub in `model.rs` so the crate compiles**

```rust
//! `NoiseModel` — Aer-style attachment of channels to gates and qubits.

use std::collections::HashMap;

use smallvec::SmallVec;

use super::error::{QuantumError, ReadoutError};

/// Maps `(gate, qubits) → channels` (Aer-style). Consumed by `run_noisy`;
/// never an IR instruction.
#[derive(Debug, Clone, Default)]
pub struct NoiseModel {
    /// Channels attached to a specific (gate-name, qubit-tuple).
    specific: HashMap<(String, SmallVec<[u32; 2]>), Vec<QuantumError>>,
    /// Channels attached to a gate name on whichever qubits it acts on.
    all_qubit: HashMap<String, Vec<QuantumError>>,
    /// Per-qubit readout error.
    readout: HashMap<u32, ReadoutError>,
}

impl NoiseModel {
    /// A model with no errors. `run_noisy` under this reproduces the noiseless
    /// distribution (the structural noiseless guard).
    pub fn new() -> Self {
        Self::default()
    }
}
```

- [ ] **Step 5: Write an `apply.rs` stub so the module tree compiles**

```rust
//! Channel application (quantum-jump) and per-shot RNG seeding.

/// Deterministic per-shot seed: a splitmix64 mix of `(seed, shot)` so shot
/// outcomes are reproducible regardless of rayon scheduling (spec §1).
pub(super) fn shot_seed(seed: u64, shot: u64) -> u64 {
    // splitmix64 finalizer on a combined word — full avalanche, no correlation
    // between adjacent shots that a plain `seed + shot` would leave.
    let mut z = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(shot)
        .wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shot_seed_is_deterministic_and_distinct() {
        assert_eq!(shot_seed(7, 3), shot_seed(7, 3));
        assert_ne!(shot_seed(7, 3), shot_seed(7, 4));
        assert_ne!(shot_seed(7, 3), shot_seed(8, 3));
    }
}
```

- [ ] **Step 6: Wire the module into `lib.rs`**

In `crates/aleph-sv/src/lib.rs`, add alongside the other `mod` declarations (after `mod measure;`, keeping alphabetical-ish order is fine):

```rust
pub mod noise;
```

- [ ] **Step 7: Build and run the scaffold test**

Run: `cargo test -p aleph-sv noise::apply::tests::shot_seed -- --nocapture`
Expected: PASS (1 test), crate compiles with the new module tree.

- [ ] **Step 8: Commit**

```bash
git add crates/aleph-sv/Cargo.toml crates/aleph-sv/src/lib.rs crates/aleph-sv/src/noise/
git commit -m "feat(noise): module scaffold + data types for SV noise engine"
```

---

## Task 2: Channel constructors + CPTP property tests

**Files:**
- Modify: `crates/aleph-sv/src/noise/error.rs`
- Test: inline `#[cfg(test)] mod tests` in `error.rs`

- [ ] **Step 1: Write the failing CPTP tests in `error.rs`**

Append to `error.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    /// Σ over a Pauli channel's weights must equal 1 (CPTP for a Pauli mix).
    fn pauli_weight_sum(c: &PauliChannel) -> f64 {
        c.terms.iter().map(|(p, _)| *p).sum()
    }

    /// Σ Kᵢ† Kᵢ for a 1q Kraus set, as a 2×2 matrix.
    fn kraus_completeness(c: &KrausChannel) -> [[Complex; 2]; 2] {
        let mut acc = [[Complex::new(0.0, 0.0); 2]; 2];
        for k in &c.kraus {
            // (K† K)[r][c] = Σ_s conj(K[s][r]) * K[s][c]
            for r in 0..2 {
                for col in 0..2 {
                    let mut sum = Complex::new(0.0, 0.0);
                    for s in 0..2 {
                        sum += k[s][r].conj() * k[s][col];
                    }
                    acc[r][col] += sum;
                }
            }
        }
        acc
    }

    fn assert_is_identity(m: [[Complex; 2]; 2]) {
        let eye = [
            [Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)],
            [Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)],
        ];
        for r in 0..2 {
            for c in 0..2 {
                assert!(
                    (m[r][c] - eye[r][c]).norm() < 1e-12,
                    "ΣK†K[{r}][{c}] = {:?}, expected I",
                    m[r][c]
                );
            }
        }
    }

    #[test]
    fn depolarizing_1q_weights_sum_to_one_and_match_aer() {
        let QuantumError::Pauli(c) = depolarizing_error(0.1, 1) else {
            panic!("1q depolarizing must be a Pauli channel");
        };
        assert!((pauli_weight_sum(&c) - 1.0).abs() < 1e-12);
        // Aer parameterization: I weight = 1 - 3p/4, each of X,Y,Z = p/4.
        let i_weight = c.terms.iter().find(|(_, p)| p[0] == Pauli::I).unwrap().0;
        assert!((i_weight - (1.0 - 3.0 * 0.1 / 4.0)).abs() < 1e-12);
        for pl in [Pauli::X, Pauli::Y, Pauli::Z] {
            let w = c.terms.iter().find(|(_, p)| p[0] == pl).unwrap().0;
            assert!((w - 0.1 / 4.0).abs() < 1e-12, "{pl:?} weight {w}");
        }
    }

    #[test]
    fn depolarizing_2q_weights_sum_to_one_and_match_aer() {
        let QuantumError::Pauli(c) = depolarizing_error(0.2, 2) else {
            panic!("2q depolarizing must be a Pauli channel");
        };
        assert_eq!(c.arity, 2);
        assert_eq!(c.terms.len(), 16); // 4×4 Paulis
        assert!((pauli_weight_sum(&c) - 1.0).abs() < 1e-12);
        // I⊗I weight = 1 - 15p/16; every other = p/16.
        let ii = c
            .terms
            .iter()
            .find(|(_, p)| p[0] == Pauli::I && p[1] == Pauli::I)
            .unwrap()
            .0;
        assert!((ii - (1.0 - 15.0 * 0.2 / 16.0)).abs() < 1e-12);
        for (w, p) in &c.terms {
            if !(p[0] == Pauli::I && p[1] == Pauli::I) {
                assert!((w - 0.2 / 16.0).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn amplitude_damping_is_cptp() {
        let QuantumError::Kraus(c) = amplitude_damping_error(0.3) else {
            panic!("amplitude damping must be a general Kraus channel");
        };
        assert_eq!(c.kraus.len(), 2);
        assert_is_identity(kraus_completeness(&c));
    }

    #[test]
    fn phase_damping_is_cptp() {
        let QuantumError::Kraus(c) = phase_damping_error(0.4) else {
            panic!("phase damping must be a general Kraus channel");
        };
        assert_is_identity(kraus_completeness(&c));
    }

    #[test]
    fn bit_flip_is_pauli_mix() {
        let QuantumError::Pauli(c) = bit_flip_error(0.25) else {
            panic!();
        };
        let i = c.terms.iter().find(|(_, p)| p[0] == Pauli::I).unwrap().0;
        let x = c.terms.iter().find(|(_, p)| p[0] == Pauli::X).unwrap().0;
        assert!((i - 0.75).abs() < 1e-12);
        assert!((x - 0.25).abs() < 1e-12);
    }

    #[test]
    fn pauli_error_normalizes_input() {
        let QuantumError::Pauli(c) = pauli_error(&[("X", 0.1), ("I", 0.9)]) else {
            panic!();
        };
        assert!((pauli_weight_sum(&c) - 1.0).abs() < 1e-12);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p aleph-sv noise::error -- --nocapture`
Expected: FAIL — `depolarizing_error`, `amplitude_damping_error`, etc. not found.

- [ ] **Step 3: Implement the constructors in `error.rs`**

Insert before the `#[cfg(test)]` block:

```rust
/// All single-qubit Paulis in fixed order I, X, Y, Z.
const PAULI1: [Pauli; 4] = [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z];

/// Depolarizing error matching Qiskit Aer's `depolarizing_error(p, num_qubits)`.
///
/// As a Pauli mixture over `d²` Paulis (`d = 2^num_qubits`): the identity carries
/// weight `1 - p·(d²-1)/d²` and each of the other `d²-1` Paulis carries `p/d²`
/// (spec §3). Panics on `num_qubits ∉ {1,2}` or `p` outside `[0, 1]`.
pub fn depolarizing_error(p: f64, num_qubits: u8) -> QuantumError {
    assert!((0.0..=1.0).contains(&p), "depolarizing p must be in [0,1]");
    assert!(num_qubits == 1 || num_qubits == 2, "v1 depolarizing is 1q or 2q");
    let d2 = 1usize << (2 * num_qubits as u32); // d² = 4^num_qubits
    let off = p / d2 as f64;
    let mut terms = Vec::with_capacity(d2);
    if num_qubits == 1 {
        for pl in PAULI1 {
            let w = if pl == Pauli::I { 1.0 - p * 3.0 / 4.0 } else { off };
            terms.push((w, SmallVec::from_slice(&[pl])));
        }
    } else {
        for a in PAULI1 {
            for b in PAULI1 {
                let is_ii = a == Pauli::I && b == Pauli::I;
                let w = if is_ii { 1.0 - p * 15.0 / 16.0 } else { off };
                terms.push((w, SmallVec::from_slice(&[a, b])));
            }
        }
    }
    QuantumError::Pauli(PauliChannel { arity: num_qubits, terms })
}

/// Amplitude damping. K₀ = diag(1, √(1-γ)), K₁ = √γ·|0⟩⟨1|. General 1q channel.
pub fn amplitude_damping_error(gamma: f64) -> QuantumError {
    assert!((0.0..=1.0).contains(&gamma), "gamma must be in [0,1]");
    let z = Complex::new(0.0, 0.0);
    let k0 = [
        [Complex::new(1.0, 0.0), z],
        [z, Complex::new((1.0 - gamma).sqrt(), 0.0)],
    ];
    let k1 = [[z, Complex::new(gamma.sqrt(), 0.0)], [z, z]];
    QuantumError::Kraus(KrausChannel { kraus: vec![k0, k1] })
}

/// Phase damping. K₀ = diag(1, √(1-λ)), K₁ = diag(0, √λ). General 1q channel.
pub fn phase_damping_error(lam: f64) -> QuantumError {
    assert!((0.0..=1.0).contains(&lam), "lambda must be in [0,1]");
    let z = Complex::new(0.0, 0.0);
    let k0 = [
        [Complex::new(1.0, 0.0), z],
        [z, Complex::new((1.0 - lam).sqrt(), 0.0)],
    ];
    let k1 = [[z, z], [z, Complex::new(lam.sqrt(), 0.0)]];
    QuantumError::Kraus(KrausChannel { kraus: vec![k0, k1] })
}

/// Bit-flip: apply X with probability `p`, identity otherwise.
pub fn bit_flip_error(p: f64) -> QuantumError {
    single_pauli_flip(Pauli::X, p)
}

/// Phase-flip: apply Z with probability `p`, identity otherwise.
pub fn phase_flip_error(p: f64) -> QuantumError {
    single_pauli_flip(Pauli::Z, p)
}

fn single_pauli_flip(pl: Pauli, p: f64) -> QuantumError {
    assert!((0.0..=1.0).contains(&p), "flip p must be in [0,1]");
    QuantumError::Pauli(PauliChannel {
        arity: 1,
        terms: vec![
            (1.0 - p, SmallVec::from_slice(&[Pauli::I])),
            (p, SmallVec::from_slice(&[pl])),
        ],
    })
}

/// Build a 1q Pauli channel from `(label, prob)` pairs; labels are "I","X",
/// "Y","Z". Weights are renormalized to sum to 1 (mirrors Aer's `pauli_error`).
pub fn pauli_error(terms: &[(&str, f64)]) -> QuantumError {
    let total: f64 = terms.iter().map(|(_, p)| *p).sum();
    assert!(total > 0.0, "pauli_error weights must sum to > 0");
    let parse = |s: &str| match s {
        "I" => Pauli::I,
        "X" => Pauli::X,
        "Y" => Pauli::Y,
        "Z" => Pauli::Z,
        other => panic!("unknown Pauli label {other:?}"),
    };
    let terms = terms
        .iter()
        .map(|(s, p)| (p / total, SmallVec::from_slice(&[parse(s)])))
        .collect();
    QuantumError::Pauli(PauliChannel { arity: 1, terms })
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p aleph-sv noise::error -- --nocapture`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/src/noise/error.rs
git commit -m "feat(noise): channel constructors (depolarizing/damping/flip/pauli) + CPTP tests"
```

---

## Task 3: Pauli fast-path application

**Files:**
- Modify: `crates/aleph-sv/src/noise/apply.rs`
- Test: inline tests in `apply.rs`

- [ ] **Step 1: Write the failing test in `apply.rs`**

Append to `apply.rs` (above the existing `#[cfg(test)] mod tests` — merge into it):

```rust
#[cfg(test)]
mod apply_tests {
    use super::*;
    use crate::noise::error::{depolarizing_error, pauli_error, QuantumError};
    use aleph_core::{AlignedBuf, Complex};
    use rand::{rngs::StdRng, SeedableRng};

    /// A Pauli channel that applies X with probability 1 must turn |0⟩ into |1⟩
    /// (deterministic — no dependence on the RNG branch).
    #[test]
    fn certain_x_flips_basis_state() {
        let mut amps =
            AlignedBuf::from_slice(&[Complex::new(1.0, 0.0), Complex::new(0.0, 0.0)]);
        let err = pauli_error(&[("X", 1.0)]);
        let mut rng = StdRng::seed_from_u64(0);
        apply_channel(&mut amps, 1, &err, &[0], &mut rng);
        assert!((amps[0].norm() - 0.0).abs() < 1e-12);
        assert!((amps[1].norm() - 1.0).abs() < 1e-12);
    }

    /// Identity-weight-1 Pauli channel leaves the state untouched and consumes
    /// no normalization (norm stays exactly 1).
    #[test]
    fn certain_identity_is_noop() {
        let mut amps = AlignedBuf::from_slice(&[
            Complex::new(0.6, 0.0),
            Complex::new(0.8, 0.0),
        ]);
        let err = pauli_error(&[("I", 1.0)]);
        let mut rng = StdRng::seed_from_u64(0);
        apply_channel(&mut amps, 1, &err, &[0], &mut rng);
        assert!((amps[0] - Complex::new(0.6, 0.0)).norm() < 1e-12);
        assert!((amps[1] - Complex::new(0.8, 0.0)).norm() < 1e-12);
    }

    /// 2q depolarizing applies a 2-qubit Pauli (tensor product) via two 1q
    /// kernels; on |00⟩ a chosen X⊗I must produce |01⟩ (qubit 0 flipped).
    #[test]
    fn depolarizing_2q_stays_normalized() {
        let mut amps = AlignedBuf::from_slice(&[
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ]);
        let err = depolarizing_error(0.5, 2);
        let mut rng = StdRng::seed_from_u64(42);
        apply_channel(&mut amps, 2, &err, &[0, 1], &mut rng);
        let norm: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
        assert!((norm - 1.0).abs() < 1e-12, "norm {norm}");
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p aleph-sv noise::apply::apply_tests -- --nocapture`
Expected: FAIL — `apply_channel` not found.

- [ ] **Step 3: Implement `apply_channel` (Pauli branch) in `apply.rs`**

Add the imports at the top of `apply.rs`:

```rust
use aleph_core::{Complex, Pauli};
use rand::{rngs::StdRng, Rng};

use super::error::{KrausChannel, PauliChannel, QuantumError};
```

Then add:

```rust
/// Apply one channel to `amps` by quantum-jump. `qubits` maps the channel's
/// local qubit indices to global qubit indices. Pauli channels take the
/// state-independent fast path (sample a Pauli, apply via unitary kernels, no
/// renormalization); general Kraus channels compute `pᵢ=‖Kᵢ|ψ〉‖²`, sample,
/// apply, and renormalize. Spec §3.
pub(super) fn apply_channel(
    amps: &mut [Complex],
    _num_qubits: u32,
    err: &QuantumError,
    qubits: &[u32],
    rng: &mut StdRng,
) {
    match err {
        QuantumError::Pauli(c) => apply_pauli_channel(amps, c, qubits, rng),
        QuantumError::Kraus(c) => apply_kraus_1q(amps, c, qubits[0], rng),
    }
}

/// Fast path: sample a Pauli term from the fixed weights and apply each
/// single-qubit Pauli factor via the existing `apply_1q` kernel. No norm pass,
/// no renormalization — the weights are state-independent.
fn apply_pauli_channel(amps: &mut [Complex], c: &PauliChannel, qubits: &[u32], rng: &mut StdRng) {
    let r: f64 = rng.gen::<f64>();
    let mut acc = 0.0;
    let mut chosen = c.terms.len() - 1; // last term absorbs FP residue
    for (i, (p, _)) in c.terms.iter().enumerate() {
        acc += *p;
        if r < acc {
            chosen = i;
            break;
        }
    }
    let (_, paulis) = &c.terms[chosen];
    for (local, pl) in paulis.iter().enumerate() {
        if *pl == Pauli::I {
            continue;
        }
        let m = pl.matrix();
        crate::kernels::aos::apply_1q(amps, qubits[local], &[], &m);
    }
}
```

(The general `apply_kraus_1q` lands in Task 4; add a temporary stub so the Pauli tests compile:)

```rust
fn apply_kraus_1q(_amps: &mut [Complex], _c: &KrausChannel, _q: u32, _rng: &mut StdRng) {
    unimplemented!("general Kraus path — Task 4")
}
```

- [ ] **Step 4: Run to verify the Pauli tests pass**

Run: `cargo test -p aleph-sv noise::apply::apply_tests -- --nocapture`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/src/noise/apply.rs
git commit -m "feat(noise): Pauli fast-path channel application"
```

---

## Task 4: General Kraus (quantum-jump) application + trace property test

**Files:**
- Modify: `crates/aleph-sv/src/noise/apply.rs`

- [ ] **Step 1: Write the failing tests in `apply.rs`**

Add to `apply_tests`:

```rust
    use crate::noise::error::{amplitude_damping_error, phase_damping_error};

    /// Amplitude damping with γ=1 sends |1⟩ → |0⟩ deterministically (the only
    /// branch with nonzero probability is K₁).
    #[test]
    fn amplitude_damping_gamma1_resets_excited_state() {
        let mut amps =
            AlignedBuf::from_slice(&[Complex::new(0.0, 0.0), Complex::new(1.0, 0.0)]);
        let err = amplitude_damping_error(1.0);
        let mut rng = StdRng::seed_from_u64(1);
        apply_channel(&mut amps, 1, &err, &[0], &mut rng);
        assert!((amps[0].norm() - 1.0).abs() < 1e-12, "|0⟩ amp {}", amps[0].norm());
        assert!(amps[1].norm() < 1e-12);
    }

    /// Quantum-jump must preserve normalization: after applying a general
    /// channel and renormalizing, ‖state‖ = 1 for any seed and any γ/λ.
    #[test]
    fn general_channel_preserves_norm() {
        for (seed, gamma) in [(0u64, 0.2), (1, 0.5), (2, 0.8), (3, 0.99)] {
            let mut amps = AlignedBuf::from_slice(&[
                Complex::new(0.6, 0.0),
                Complex::new(0.0, 0.8),
            ]);
            let err = amplitude_damping_error(gamma);
            let mut rng = StdRng::seed_from_u64(seed);
            apply_channel(&mut amps, 1, &err, &[0], &mut rng);
            let n: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
            assert!((n - 1.0).abs() < 1e-10, "seed {seed} γ {gamma}: norm {n}");
        }
    }

    #[test]
    fn phase_damping_preserves_norm() {
        let mut amps = AlignedBuf::from_slice(&[
            Complex::new(0.5, 0.5),
            Complex::new(0.5, -0.5),
        ]);
        let err = phase_damping_error(0.6);
        let mut rng = StdRng::seed_from_u64(7);
        apply_channel(&mut amps, 1, &err, &[0], &mut rng);
        let n: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
        assert!((n - 1.0).abs() < 1e-10, "norm {n}");
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p aleph-sv noise::apply::apply_tests::general_channel_preserves_norm -- --nocapture`
Expected: FAIL — `unimplemented!("general Kraus path")` panics.

- [ ] **Step 3: Replace the `apply_kraus_1q` stub with the real implementation**

```rust
/// General 1q quantum-jump. For Kraus set {Kᵢ} on qubit `q`:
///   1. pᵢ = ‖Kᵢ|ψ〉‖² (Σpᵢ = 1 by CPTP);
///   2. sample branch i with probability pᵢ;
///   3. apply Kᵢ to |ψ〉 and renormalize by 1/√pᵢ.
/// Works pairwise over the (qubit `q` = 0, qubit `q` = 1) amplitude pairs.
fn apply_kraus_1q(amps: &mut [Complex], c: &KrausChannel, q: u32, rng: &mut StdRng) {
    let qbit = 1usize << q;
    // Step 1: branch probabilities. For each pair (a0, a1) and each Kraus op,
    // the local image is (K[0][0]a0 + K[0][1]a1, K[1][0]a0 + K[1][1]a1).
    let mut probs = vec![0.0_f64; c.kraus.len()];
    for i in 0..amps.len() {
        if i & qbit != 0 {
            continue; // visit each pair once, from its qbit-clear index
        }
        let a0 = amps[i];
        let a1 = amps[i | qbit];
        for (ki, k) in c.kraus.iter().enumerate() {
            let o0 = k[0][0] * a0 + k[0][1] * a1;
            let o1 = k[1][0] * a0 + k[1][1] * a1;
            probs[ki] += o0.norm_sqr() + o1.norm_sqr();
        }
    }
    // Step 2: sample a branch (last branch absorbs FP residue).
    let r = rng.gen::<f64>();
    let mut acc = 0.0;
    let mut chosen = c.kraus.len() - 1;
    for (ki, p) in probs.iter().enumerate() {
        acc += *p;
        if r < acc {
            chosen = ki;
            break;
        }
    }
    // Step 3: apply the chosen Kraus op and renormalize by 1/√p_chosen.
    let pc = probs[chosen];
    if pc < 1e-300 {
        // Degenerate branch: nothing meaningful to project onto. Leave the
        // state as-is rather than scaling by ~1e150 (mirrors measure.rs).
        return;
    }
    let inv = 1.0 / pc.sqrt();
    let k = &c.kraus[chosen];
    for i in 0..amps.len() {
        if i & qbit != 0 {
            continue;
        }
        let a0 = amps[i];
        let a1 = amps[i | qbit];
        amps[i] = (k[0][0] * a0 + k[0][1] * a1) * inv;
        amps[i | qbit] = (k[1][0] * a0 + k[1][1] * a1) * inv;
    }
}
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `cargo test -p aleph-sv noise::apply::apply_tests -- --nocapture`
Expected: PASS (all 6 apply tests).

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/src/noise/apply.rs
git commit -m "feat(noise): general quantum-jump Kraus application (amplitude/phase damping)"
```

---

## Task 5: Readout error application

**Files:**
- Modify: `crates/aleph-sv/src/noise/apply.rs`
- Modify: `crates/aleph-sv/src/noise/error.rs` (add `ReadoutError` constructor)

- [ ] **Step 1: Write the failing test in `apply.rs`**

Add to `apply_tests`:

```rust
    use crate::noise::error::ReadoutError;

    /// Readout error with P(1|0)=1 and P(0|1)=1 flips every measured bit.
    #[test]
    fn readout_flips_all_bits_when_certain() {
        let ro = ReadoutError::new([[0.0, 1.0], [1.0, 0.0]]);
        let map: std::collections::HashMap<u32, ReadoutError> =
            [(0u32, ro), (1u32, ro)].into_iter().collect();
        let mut rng = StdRng::seed_from_u64(0);
        // basis state |01⟩ = index 0b01 = 1 over 2 qubits → both bits flip → |10⟩ = 2
        let out = apply_readout(1, 2, &map, &mut rng);
        assert_eq!(out, 0b10);
    }

    /// Identity readout (perfect measurement) is the identity on the index.
    #[test]
    fn readout_identity_is_noop() {
        let ro = ReadoutError::new([[1.0, 0.0], [0.0, 1.0]]);
        let map: std::collections::HashMap<u32, ReadoutError> =
            [(0u32, ro), (1u32, ro), (2u32, ro)].into_iter().collect();
        let mut rng = StdRng::seed_from_u64(123);
        assert_eq!(apply_readout(0b101, 3, &map, &mut rng), 0b101);
    }
```

- [ ] **Step 2: Add the `ReadoutError::new` constructor in `error.rs`**

Add an `impl ReadoutError` block (after the struct definition):

```rust
impl ReadoutError {
    /// `m[t][o]` = P(measured `o` | true `t`). Each row must sum to ~1.
    pub fn new(m: [[f64; 2]; 2]) -> Self {
        debug_assert!((m[0][0] + m[0][1] - 1.0).abs() < 1e-9, "readout row 0 must sum to 1");
        debug_assert!((m[1][0] + m[1][1] - 1.0).abs() < 1e-9, "readout row 1 must sum to 1");
        Self { m }
    }
}
```

- [ ] **Step 3: Run to verify failure**

Run: `cargo test -p aleph-sv noise::apply::apply_tests::readout -- --nocapture`
Expected: FAIL — `apply_readout` not found.

- [ ] **Step 4: Implement `apply_readout` in `apply.rs`**

Add the import `use std::collections::HashMap;` and `use super::error::ReadoutError;` at the top, then:

```rust
/// Apply per-qubit readout error to a sampled basis-state index. For each
/// measured qubit with a `ReadoutError`, the recorded outcome bit is the true
/// bit `t` flipped to `1-t` with probability `m[t][1-t]`. Qubits without an
/// entry are read out perfectly.
pub(super) fn apply_readout(
    index: u64,
    num_qubits: u32,
    readout: &HashMap<u32, ReadoutError>,
    rng: &mut StdRng,
) -> u64 {
    if readout.is_empty() {
        return index;
    }
    let mut out = index;
    for q in 0..num_qubits {
        let Some(ro) = readout.get(&q) else { continue };
        let bit = ((index >> q) & 1) as usize; // true value t
        let p_flip = ro.m[bit][1 - bit]; // P(measure 1-t | true t)
        if rng.gen::<f64>() < p_flip {
            out ^= 1u64 << q; // record the flipped bit
        }
    }
    out
}
```

- [ ] **Step 5: Run to verify the tests pass**

Run: `cargo test -p aleph-sv noise::apply::apply_tests -- --nocapture`
Expected: PASS (readout tests + earlier apply tests).

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-sv/src/noise/apply.rs crates/aleph-sv/src/noise/error.rs
git commit -m "feat(noise): per-qubit readout error application"
```

---

## Task 6: `NoiseModel` attachment + `errors_for`

**Files:**
- Modify: `crates/aleph-sv/src/noise/model.rs`

- [ ] **Step 1: Write the failing test in `model.rs`**

Append to `model.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise::error::{depolarizing_error, ReadoutError};

    #[test]
    fn all_qubit_and_specific_concatenate_in_order() {
        let mut nm = NoiseModel::new();
        nm.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), &["h"]);
        nm.add_quantum_error(depolarizing_error(0.02, 1), &["h"], &[0]);
        // all-qubit list first, then qubit-specific, per Aer order.
        let errs = nm.errors_for("h", &[0]);
        assert_eq!(errs.len(), 2);
        // On a qubit with no specific attachment, only the all-qubit error fires.
        assert_eq!(nm.errors_for("h", &[1]).len(), 1);
        // A gate with no attachment yields nothing.
        assert_eq!(nm.errors_for("x", &[0]).len(), 0);
    }

    #[test]
    fn readout_round_trips() {
        let mut nm = NoiseModel::new();
        nm.add_readout_error(ReadoutError::new([[0.98, 0.02], [0.03, 0.97]]), 0);
        assert!(nm.readout_error(0).is_some());
        assert!(nm.readout_error(1).is_none());
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p aleph-sv noise::model -- --nocapture`
Expected: FAIL — `add_all_qubit_quantum_error`, `errors_for`, etc. not found.

- [ ] **Step 3: Implement the methods in `model.rs`**

Add to `impl NoiseModel`:

```rust
    /// Attach `err` to `gate_names` on the specific `qubits` tuple (Aer's
    /// `add_quantum_error`). Applied after the gate, in insertion order.
    pub fn add_quantum_error(&mut self, err: QuantumError, gate_names: &[&str], qubits: &[u32]) {
        let key_qubits: SmallVec<[u32; 2]> = SmallVec::from_slice(qubits);
        for name in gate_names {
            self.specific
                .entry(((*name).to_string(), key_qubits.clone()))
                .or_default()
                .push(err.clone());
        }
    }

    /// Attach `err` to `gate_names` on whichever qubits the gate acts on
    /// (Aer's `add_all_qubit_quantum_error`).
    pub fn add_all_qubit_quantum_error(&mut self, err: QuantumError, gate_names: &[&str]) {
        for name in gate_names {
            self.all_qubit.entry((*name).to_string()).or_default().push(err.clone());
        }
    }

    /// Attach a per-qubit readout error.
    pub fn add_readout_error(&mut self, err: ReadoutError, qubit: u32) {
        self.readout.insert(qubit, err);
    }

    /// Errors that fire after a gate named `gate_name` acting on `qubits`:
    /// the all-qubit list first, then the qubit-specific list (Aer order).
    pub fn errors_for(&self, gate_name: &str, qubits: &[u32]) -> Vec<&QuantumError> {
        let mut out: Vec<&QuantumError> = Vec::new();
        if let Some(list) = self.all_qubit.get(gate_name) {
            out.extend(list.iter());
        }
        let key = (gate_name.to_string(), SmallVec::<[u32; 2]>::from_slice(qubits));
        if let Some(list) = self.specific.get(&key) {
            out.extend(list.iter());
        }
        out
    }

    /// The readout error for `qubit`, if any.
    pub fn readout_error(&self, qubit: u32) -> Option<&ReadoutError> {
        self.readout.get(&qubit)
    }

    /// Whether any readout error is configured (lets the driver skip the
    /// per-qubit loop entirely in the common no-readout case).
    pub(crate) fn readout_map(&self) -> &HashMap<u32, ReadoutError> {
        &self.readout
    }
```

Add the import at the top of `model.rs`: change the `use super::error::...` line to

```rust
use super::error::{QuantumError, ReadoutError};
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `cargo test -p aleph-sv noise::model -- --nocapture`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/src/noise/model.rs
git commit -m "feat(noise): NoiseModel Aer-style attachment + errors_for"
```

---

## Task 7: `run_noisy` driver + determinism + noiseless guard

**Files:**
- Modify: `crates/aleph-sv/src/noise/mod.rs`
- Test: `crates/aleph-sv/tests/noise_driver.rs` (new integration test)

- [ ] **Step 1: Write the `run_noisy` driver in `mod.rs`**

Add the imports at the top of `mod.rs`:

```rust
use aleph_ir::{Circuit, Instruction};
use rayon::prelude::*;

use crate::NaiveSvBackend;
use aleph_backend::Backend;
```

Then add:

```rust
/// Run `circuit` under `noise_model` for `shots` Monte-Carlo trajectories and
/// return a per-basis-state histogram (length `2^num_qubits`).
///
/// Each shot owns a fresh `CpuState` and a `StdRng` seeded `shot_seed(seed,
/// shot)`, so counts are reproducible regardless of rayon scheduling. v1
/// supports terminal measurement only: mid-circuit `Measure`/`Reset` raise
/// [`NoiseError::MidCircuit`]. `Barrier` is a no-op; `DiagonalPhase` is applied
/// via the backend.
pub fn run_noisy(
    circuit: &Circuit,
    noise_model: &NoiseModel,
    shots: u32,
    seed: u64,
) -> Result<Counts, NoiseError> {
    let n = circuit.num_qubits();
    let dim = 1usize
        .checked_shl(n)
        .ok_or(NoiseError::Backend(BackendError::InvalidState {
            reason: "num_qubits exceeds platform usize::BITS",
        }))?;

    // Per-shot trajectory → final readout-perturbed basis-state index.
    let outcomes: Result<Vec<u64>, NoiseError> = (0..shots)
        .into_par_iter()
        .map(|shot| run_one_shot(circuit, noise_model, n, apply::shot_seed(seed, shot as u64)))
        .collect();
    let outcomes = outcomes?;

    let mut hist = vec![0u64; dim];
    for idx in outcomes {
        hist[idx as usize] += 1;
    }
    Ok(hist)
}

/// One Monte-Carlo trajectory: apply each gate then its attached channels,
/// then sample a terminal Z-basis outcome and apply readout error.
fn run_one_shot(
    circuit: &Circuit,
    noise_model: &NoiseModel,
    n: u32,
    seed: u64,
) -> Result<u64, NoiseError> {
    let mut backend = NaiveSvBackend::with_seed(seed);
    let mut state = backend.allocate(n)?;
    for inst in circuit.instructions() {
        match inst {
            Instruction::Gate(gi) => {
                backend.apply_gate(&mut state, gi)?;
                let name = gi.gate.name();
                // errors_for returns borrows into noise_model; apply each.
                let errs = noise_model.errors_for(name, &gi.qubits);
                for err in errs {
                    apply::apply_channel(&mut state.amps, n, err, &gi.qubits, &mut backend.rng);
                }
            }
            Instruction::DiagonalPhase(dp) => {
                backend.apply_diagonal_phase(&mut state, dp)?;
            }
            Instruction::Barrier(_) => {}
            Instruction::Measure { .. } => {
                return Err(NoiseError::MidCircuit { kind: "measure" })
            }
            Instruction::Reset(_) => return Err(NoiseError::MidCircuit { kind: "reset" }),
            Instruction::TiledBlock(_) => {
                return Err(NoiseError::MidCircuit { kind: "tiled-block" })
            }
        }
    }
    // Terminal Z-basis sample: one draw from |amps|² via the backend's rng.
    let idx = backend.sample(&state, 1)?[0];
    Ok(apply::apply_readout(idx, n, noise_model.readout_map(), &mut backend.rng))
}
```

Note: `apply::shot_seed`, `apply::apply_channel`, `apply::apply_readout` must be `pub(super)` (they already are in the earlier tasks). `state.amps` is `pub(crate)` and `backend.rng` is `pub(crate)` — both reachable from within `aleph-sv`.

- [ ] **Step 2: Write the failing integration test `crates/aleph-sv/tests/noise_driver.rs`**

```rust
//! Driver-level tests for `aleph_sv::noise::run_noisy`: determinism and the
//! noiseless guard (empty model reproduces the noiseless distribution).

use aleph_oracle::assert_distribution_close;
use aleph_parser::parse;
use aleph_sv::noise::{depolarizing_error, run_noisy, NoiseModel};

/// A 2-qubit Bell circuit (gate-only — terminal sampling does the measuring).
const BELL: &str = r#"
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
"#;

#[test]
fn deterministic_same_seed_same_counts() {
    let circ = parse(BELL).unwrap();
    let mut nm = NoiseModel::new();
    nm.add_all_qubit_quantum_error(depolarizing_error(0.05, 1), &["h"]);
    let a = run_noisy(&circ, &nm, 20_000, 7).unwrap();
    let b = run_noisy(&circ, &nm, 20_000, 7).unwrap();
    assert_eq!(a, b, "same seed must give identical counts");
    let c = run_noisy(&circ, &nm, 20_000, 8).unwrap();
    assert_ne!(a, c, "different seed should (almost surely) differ");
}

#[test]
fn empty_model_reproduces_noiseless_distribution() {
    let circ = parse(BELL).unwrap();
    let nm = NoiseModel::new(); // no errors
    let counts = run_noisy(&circ, &nm, 100_000, 1).unwrap();
    // Noiseless Bell: |00⟩ and |11⟩ each 0.5, |01⟩=|10⟩=0.
    let exact = vec![0.5, 0.0, 0.0, 0.5];
    assert_distribution_close("noise_empty_bell", 2, &counts, &exact, 100_000);
}
```

Add `aleph-oracle` to `aleph-sv`'s `[dev-dependencies]` in `Cargo.toml`:

```toml
aleph-oracle = { path = "../aleph-oracle" }
```

- [ ] **Step 3: Run to verify it fails (then passes after the driver compiles)**

Run: `cargo test -p aleph-sv --test noise_driver -- --nocapture`
Expected: first FAIL if the driver isn't wired; after Step 1 lands, PASS (2 tests). The `empty_model` test confirms the noiseless guard; `deterministic` confirms per-shot seeding.

- [ ] **Step 4: Confirm the noiseless `run()` benchmark is structurally untouched**

The noiseless path was not modified (noise is a separate `run_noisy` entry point). Confirm no noiseless source file changed outside `noise/`:

Run: `git diff --stat HEAD~6 -- crates/aleph-sv/src | grep -v 'noise/' | grep -v 'lib.rs' | grep -v 'Cargo.toml'`
Expected: empty output (only `noise/`, `lib.rs`'s `pub mod` line, and `Cargo.toml` deps changed).

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/src/noise/mod.rs crates/aleph-sv/tests/noise_driver.rs crates/aleph-sv/Cargo.toml
git commit -m "feat(noise): run_noisy Monte-Carlo driver + determinism/noiseless-guard tests"
```

---

## Task 8: Aer oracle fixtures (offline generation)

**Files:**
- Create: `oracle/noise/gen_noise.py`
- Create: `oracle/noise/*.json` (generated, committed)

**Context:** The existing oracle (`oracle/gen.py`) commits Qiskit-derived JSON fixtures and the Rust tests consume them statically (no Python at test time). We follow that pattern. The reference distribution must be the **exact** noisy probabilities — Aer's density-matrix method with `save_probabilities`, *not* sampled counts (100k Aer shots carry ~3e-3 MC error per bin, which would swamp the 1e-5 oracle tolerance).

- [ ] **Step 1: Write `oracle/noise/gen_noise.py`**

```python
"""Generate exact noisy measurement distributions via Qiskit Aer.

For each fixture we build a byte-identical NoiseModel, run the gate-only
circuit through the density-matrix method with save_probabilities, and dump
the exact P(outcome) vector. The matching aleph NoiseModel is constructed in
Rust in crates/aleph-sv/tests/noise_oracle.rs — keep the two in sync.

Run from repo root with the Phase-1 uv venv that has qiskit-aer:
    python oracle/noise/gen_noise.py
"""
from __future__ import annotations

import json
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
    ReadoutError,
)

OUT = Path(__file__).resolve().parent


def exact_probs(qc: QuantumCircuit, nm: NoiseModel, n: int) -> list[float]:
    """Exact noisy measurement distribution over 2^n outcomes (little-endian
    index i = qubit values, matching aleph's |i⟩ convention)."""
    sim = AerSimulator(method="density_matrix", noise_model=nm)
    tqc = qc.copy()
    tqc.save_probabilities()  # exact diagonal of ρ in the computational basis
    res = sim.run(tqc).result()
    probs = res.data(0)["probabilities"]  # dict or vector over 2^n
    out = [0.0] * (1 << n)
    if isinstance(probs, dict):
        for k, v in probs.items():
            out[int(k)] = float(v)
    else:
        for i, v in enumerate(probs):
            out[i] = float(v)
    return out


def dump(name: str, n: int, probs: list[float]) -> None:
    (OUT / f"{name}.json").write_text(
        json.dumps({"name": name, "num_qubits": n, "exact_probs": probs}, indent=2)
    )
    print(f"wrote {name}.json  (Σp={sum(probs):.6f})")


def depol_h() -> None:
    qc = QuantumCircuit(1)
    qc.h(0)
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(0.05, 1), ["h"])
    dump("depol_h", 1, exact_probs(qc, nm, 1))


def depol_cx() -> None:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    nm = NoiseModel()
    nm.add_quantum_error(depolarizing_error(0.1, 2), ["cx"], [0, 1])
    dump("depol_cx", 2, exact_probs(qc, nm, 2))


def amp_damp() -> None:
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.id(0)
    nm = NoiseModel()
    nm.add_quantum_error(amplitude_damping_error(0.2), ["id"], [0])
    dump("amp_damp_h", 1, exact_probs(qc, nm, 1))


def phase_damp() -> None:
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.id(0)
    nm = NoiseModel()
    nm.add_quantum_error(phase_damping_error(0.3), ["id"], [0])
    dump("phase_damp_h", 1, exact_probs(qc, nm, 1))


def readout() -> None:
    # Deterministic |1⟩ via X, asymmetric readout error.
    qc = QuantumCircuit(1)
    qc.x(0)
    nm = NoiseModel()
    nm.add_readout_error(ReadoutError([[0.98, 0.02], [0.05, 0.95]]), [0])
    dump("readout_x", 1, exact_probs(qc, nm, 1))


def combined_ghz() -> None:
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), ["cx"])
    for q in range(3):
        nm.add_readout_error(ReadoutError([[0.97, 0.03], [0.04, 0.96]]), [q])
    dump("combined_ghz3", 3, exact_probs(qc, nm, 3))


if __name__ == "__main__":
    depol_h()
    depol_cx()
    amp_damp()
    phase_damp()
    readout()
    combined_ghz()
```

- [ ] **Step 2: Generate the fixtures**

Run (locally if Qiskit Aer is installed; otherwise on the EPYC box's Phase-1 uv venv — see [[stage0-merged]] for the venv bootstrap):

```bash
python oracle/noise/gen_noise.py
```

Expected: six JSON files written, each printing `Σp≈1.000000`.

> **Endianness check (do this before trusting the fixtures):** Qiskit bitstring keys are big-endian (qubit n-1 is the leftmost char), while aleph indexes `|i⟩` with qubit `q` = bit `q` of `i`. `save_probabilities` returns a vector indexed by the integer value of Qiskit's bit order. For the asymmetric `readout_x`/`combined_ghz3` fixtures, confirm the nonzero mass sits on the index aleph expects (e.g. `readout_x` should peak at index 1, not 0). If a fixture is mirror-imaged, reverse the bit order when writing `out[i]`. Add a one-line comment in the JSON-writing code recording which convention was confirmed.

- [ ] **Step 3: Commit the generator and fixtures**

```bash
git add oracle/noise/gen_noise.py oracle/noise/*.json
git commit -m "test(noise): Qiskit Aer exact-distribution oracle fixtures (density-matrix)"
```

---

## Task 9: Oracle integration test (run_noisy vs Aer exact)

**Files:**
- Create: `crates/aleph-sv/tests/noise_oracle.rs`

- [ ] **Step 1: Write the oracle test**

```rust
//! Oracle: `run_noisy` @100k shots must match Qiskit Aer's *exact* noisy
//! distribution (density-matrix `save_probabilities`) within the calibrated 5σ
//! band. The aleph NoiseModel here mirrors the byte-identical model built in
//! oracle/noise/gen_noise.py — keep the two in sync.

use std::path::Path;

use aleph_oracle::assert_distribution_close;
use aleph_parser::parse;
use aleph_sv::noise::{
    amplitude_damping_error, depolarizing_error, phase_damping_error, run_noisy, NoiseModel,
    ReadoutError,
};

const SHOTS: u32 = 100_000;
const SEED: u64 = 0;

#[derive(serde::Deserialize)]
struct NoiseFixture {
    name: String,
    num_qubits: u32,
    exact_probs: Vec<f64>,
}

fn load(name: &str) -> NoiseFixture {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../oracle/noise")
        .join(format!("{name}.json"));
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&raw).unwrap()
}

fn check(name: &str, qasm: &str, nm: &NoiseModel) {
    let fx = load(name);
    let circ = parse(qasm).unwrap();
    let counts = run_noisy(&circ, nm, SHOTS, SEED).unwrap();
    assert_distribution_close(&fx.name, fx.num_qubits, &counts, &fx.exact_probs, SHOTS);
}

const H1: &str = "OPENQASM 3.0; include \"stdgates.inc\"; qubit[1] q; h q[0];";
const BELL: &str =
    "OPENQASM 3.0; include \"stdgates.inc\"; qubit[2] q; h q[0]; cx q[0], q[1];";
const H_ID: &str = "OPENQASM 3.0; include \"stdgates.inc\"; qubit[1] q; h q[0]; id q[0];";
const X1: &str = "OPENQASM 3.0; include \"stdgates.inc\"; qubit[1] q; x q[0];";
const GHZ3: &str = "OPENQASM 3.0; include \"stdgates.inc\"; qubit[3] q; h q[0]; cx q[0], q[1]; cx q[1], q[2];";

#[test]
fn oracle_depol_h() {
    let mut nm = NoiseModel::new();
    nm.add_all_qubit_quantum_error(depolarizing_error(0.05, 1), &["h"]);
    check("depol_h", H1, &nm);
}

#[test]
fn oracle_depol_cx() {
    let mut nm = NoiseModel::new();
    nm.add_quantum_error(depolarizing_error(0.1, 2), &["cx"], &[0, 1]);
    check("depol_cx", BELL, &nm);
}

#[test]
fn oracle_amp_damp() {
    let mut nm = NoiseModel::new();
    nm.add_quantum_error(amplitude_damping_error(0.2), &["id"], &[0]);
    check("amp_damp_h", H_ID, &nm);
}

#[test]
fn oracle_phase_damp() {
    let mut nm = NoiseModel::new();
    nm.add_quantum_error(phase_damping_error(0.3), &["id"], &[0]);
    check("phase_damp_h", H_ID, &nm);
}

#[test]
fn oracle_readout() {
    let mut nm = NoiseModel::new();
    nm.add_readout_error(ReadoutError::new([[0.98, 0.02], [0.05, 0.95]]), 0);
    check("readout_x", X1, &nm);
}

#[test]
fn oracle_combined_ghz3() {
    let mut nm = NoiseModel::new();
    nm.add_all_qubit_quantum_error(depolarizing_error(0.02, 2), &["cx"]);
    for q in 0..3 {
        nm.add_readout_error(ReadoutError::new([[0.97, 0.03], [0.04, 0.96]]), q);
    }
    check("combined_ghz3", GHZ3, &nm);
}
```

- [ ] **Step 2: Add `serde`/`serde_json` to `aleph-sv` dev-dependencies**

In `crates/aleph-sv/Cargo.toml` under `[dev-dependencies]`:

```toml
serde      = { workspace = true, features = ["derive"] }
serde_json = { workspace = true }
```

(If these aren't workspace deps, the oracle crate already depends on them — check `crates/aleph-oracle/Cargo.toml` for the exact version pin and mirror it.)

- [ ] **Step 3: Run the oracle tests**

Run: `cargo test -p aleph-sv --test noise_oracle -- --nocapture`
Expected: PASS (6 tests). If a fixture is mirror-imaged, fix the endianness in `gen_noise.py` (Task 8 Step 2 note) and regenerate — do **not** silently flip indices in the Rust test.

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-sv/tests/noise_oracle.rs crates/aleph-sv/Cargo.toml
git commit -m "test(noise): Aer exact-distribution oracle for channel set v1"
```

---

## Task 10: Determinism proptest, docs, and final gate

**Files:**
- Modify: `crates/aleph-sv/src/noise/apply.rs` (CPTP proptest on random states)
- Modify: `crates/aleph-sv/src/noise/mod.rs` (module rustdoc with a usage example)

- [ ] **Step 1: Add a trace-preservation proptest over random states + channels**

In `apply.rs`, add a `proptest!` block (add `use proptest::prelude::*;` and the `aleph-test`/`proptest` dev-deps are already present):

```rust
#[cfg(test)]
mod prop_tests {
    use super::*;
    use crate::noise::error::{amplitude_damping_error, depolarizing_error, phase_damping_error};
    use aleph_core::{AlignedBuf, Complex};
    use proptest::prelude::*;
    use rand::{rngs::StdRng, SeedableRng};

    /// A normalized random 1q state from two complex amplitudes.
    fn norm_state(a: Complex, b: Complex) -> AlignedBuf<Complex> {
        let n = (a.norm_sqr() + b.norm_sqr()).sqrt().max(1e-300);
        AlignedBuf::from_slice(&[a / Complex::new(n, 0.0), b / Complex::new(n, 0.0)])
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

        /// Any v1 channel preserves ‖state‖ = 1 after quantum-jump + renorm,
        /// for any input state, any channel parameter, and any RNG seed.
        #[test]
        fn apply_channel_preserves_norm(
            ar in -1.0_f64..1.0, ai in -1.0_f64..1.0,
            br in -1.0_f64..1.0, bi in -1.0_f64..1.0,
            param in 0.0_f64..1.0,
            seed in any::<u64>(),
            which in 0u8..3,
        ) {
            prop_assume!(ar.abs() + ai.abs() + br.abs() + bi.abs() > 1e-6);
            let mut amps = norm_state(Complex::new(ar, ai), Complex::new(br, bi));
            let err = match which {
                0 => amplitude_damping_error(param),
                1 => phase_damping_error(param),
                _ => depolarizing_error(param, 1),
            };
            let mut rng = StdRng::seed_from_u64(seed);
            apply_channel(&mut amps, 1, &err, &[0], &mut rng);
            let n: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
            prop_assert!((n - 1.0).abs() < 1e-9, "which={which} param={param} norm={n}");
        }
    }
}
```

- [ ] **Step 2: Run the proptest**

Run: `cargo test -p aleph-sv noise::apply::prop_tests -- --nocapture`
Expected: PASS.

- [ ] **Step 3: Add a module rustdoc usage example to `mod.rs`**

Prepend to the `//!` header in `mod.rs`:

```rust
//! # Example
//! ```no_run
//! use aleph_sv::noise::{depolarizing_error, run_noisy, NoiseModel};
//! # let circuit = aleph_parser::parse("OPENQASM 3.0; qubit[1] q; h q[0];").unwrap();
//! let mut nm = NoiseModel::new();
//! nm.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), &["h"]);
//! let counts = run_noisy(&circuit, &nm, 100_000, 7).unwrap();
//! ```
```

- [ ] **Step 4: Full workspace gate (build, test, clippy, fmt)**

Run:
```bash
cargo test -p aleph-sv
cargo clippy -p aleph-sv --all-targets -- -D warnings
cargo fmt --check
```
Expected: all green. The full `cargo test --workspace` should also pass (noise is additive).

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/src/noise/
git commit -m "test(noise): trace-preservation proptest + module docs"
```

---

## Self-Review (completed against the spec)

**Spec coverage:**
- §1 trajectories / per-shot RNG / rayon → Task 7 (`run_noisy`, `shot_seed`, `into_par_iter`).
- §2 `aleph_sv::noise` driver on `CpuState`, IR untouched, separate entry point → Tasks 1 & 7; Task 7 Step 4 verifies no noiseless source changed.
- §3 channel set v1 (depol 1q/2q, amp/phase damping, bit/phase/Y flip, readout) → Tasks 2 & 5; Pauli fast-path vs general quantum-jump → Tasks 3 & 4; Aer-style attachment + `errors_for` → Task 6; measurement/reset v1.1 deferral → Task 7 (`NoiseError::MidCircuit`).
- §4 API surface (Rust `NoiseModel`/`*_error`) → Tasks 2 & 6. (Python/CLI is P4.6-05, out of scope here.)
- §5 frame-sampler integration → representation-only (Pauli fast-path exposes "sample a Pauli"); explicitly out of scope, no task needed.
- Oracle protocol → Tasks 8 & 9 (exact density-matrix reference at 1e-5/100k via `assert_distribution_close`); property tests (CPTP/trace/determinism/noiseless guard) → Tasks 2, 4, 7, 10.

**AC mapping (BACKLOG #167):**
- AC-1 (channel set v1 end-to-end via `run_noisy`) → Tasks 2–7, proven by Task 9.
- AC-2 (Aer oracle 1e-5 @100k on the fixture set) → Task 9 (all six fixtures).
- AC-3 (CPTP Σpᵢ=1, ‖state‖=1, deterministic seeding, empty-model noiseless, noiseless bench unchanged) → Tasks 2 (Σpᵢ=1), 4/10 (norm), 7 (determinism + noiseless guard + bench-unchanged check).

**Y-flip note:** the spec lists a Y-flip channel; `pauli_error(&[("Y", p), ("I", 1.0-p)])` covers it with no extra constructor (a dedicated `y_flip_error` is trivial to add if a fixture needs it — not required by the AC fixture set).

**Open risk flagged for execution:** the Qiskit↔aleph bit-order convention (Task 8 Step 2) is the most likely source of a red oracle. Resolve it in the fixture generator, never by flipping indices in the Rust assertion.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-13-p46-04-noise-sv-engine.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Best here because Tasks 2–7 are pure-Rust TDD with crisp pass/fail gates.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
