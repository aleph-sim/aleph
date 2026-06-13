# 0014 — Noise via stochastic Kraus trajectories, driver-applied (P4.6-03)

## Status

Accepted, 2026-06-13. Spec:
`docs/superpowers/specs/2026-06-13-p46-03-noise-models-design.md`.

## Context

Phase 4.6 adds noise simulation (ROADMAP §7 exit metric: channel set v1 with an
Aer oracle at 1e-5 / 100k shots). Two representation choices had to be fixed
before implementation:

1. **How to simulate noise.** A density-matrix backend is exact but O(4ⁿ)
   (~14-qubit ceiling) and is a whole new backend. Stochastic Kraus trajectories
   (the Monte-Carlo wavefunction / quantum-jump method) reuse the existing O(2ⁿ)
   statevector — each shot samples one Kraus branch per channel — and reproduce
   the noisy measurement distribution to sampling accuracy. Qiskit Aer ships both
   and uses trajectories for shot-based sampling.

2. **Where noise lives in the data model.** Golden rule 4 requires the IR and
   backends to stay backend-agnostic and noise-free. A per-gate noise field in
   the IR (or a noise `Instruction`) would violate that and leak channel concepts
   into every backend.

## Decision

- **Trajectories on the state-vector backend.** A Monte-Carlo driver runs the
  circuit `shots` times; each channel application collapses to one Kraus branch
  via quantum-jump (pᵢ = ‖Kᵢ|ψ〉‖², sample, apply, renormalize). Pauli channels
  take a fast path (state-independent weights, unitary kernels, no renorm).
  Density-matrix simulation is **deferred** until a user needs exact ρ.

- **Noise is a runtime `NoiseModel` config applied by a SV-specific driver, not
  IR.** `NoiseModel` maps `(gate_kind, qubits) → channels` (Aer-style
  attachment) and is consumed by `aleph_sv::noise::run_noisy`. The IR, the
  `Backend` trait, and the noiseless `run()` path are untouched. The driver
  works directly on `CpuState` because trajectory channels need
  amplitude/norm/arbitrary-Kraus primitives the generic `Backend` trait does not
  expose.

- **Aer-compatible API.** Python `NoiseModel` + `*_error` factories mirror Qiskit
  Aer names, making the byte-identical-model oracle straightforward and lowering
  the learning curve for Qiskit users.

## Consequences

- Noise is SV-only in v1. DM/MPS noise would reuse the `NoiseModel` types (hoist
  to a shared crate then) but need their own channel application — accepted, not
  built.
- Noisy runs are sampling-based (a distribution, per-shot cost), not exact ρ.
  Acceptable for the shot-sampling use case; exact-ρ users get DM later.
- The noiseless path's performance is structurally unchanged (separate
  `run_noisy` entry point), so the "noiseless unchanged" guard is by
  construction, not luck.
- The Pauli-channel representation is chosen so the P4.6-02 Pauli-frame sampler
  can later absorb Clifford + Pauli-only noise nearly for free (out of scope v1).
- Mid-circuit measurement/reset under noise is a documented v1.1 follow-up; v1
  fixtures use terminal measurement (the `counts` sampling shape).
