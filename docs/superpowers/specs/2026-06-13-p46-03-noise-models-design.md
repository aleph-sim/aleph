# P4.6-03 — Noise models: design spec

**Status:** accepted 2026-06-13. Implemented by P4.6-04 (#167, SV engine) and
P4.6-05 (#168, Python/CLI). ADR: `docs/decisions/0014-noise-trajectories.md`.

## Goal

Give aleph a noise-simulation capability that matches Qiskit Aer's measurement
distributions to 1e-5 at 100k shots under a byte-identical noise model, while
keeping the backend-agnostic IR completely noise-free (golden rule 4). This spec
fixes the representation *before* any implementation; it answers the five
questions the ticket poses and defines the oracle protocol that becomes
P4.6-04's testing section.

## 1. Simulation strategy — stochastic Kraus trajectories (quantum-jump)

**Decision: trajectory (Monte-Carlo wavefunction) method on the state-vector
backend. Density-matrix backend deferred.**

A noisy circuit is sampled by running it `shots` times. Within one shot, each
noise channel application stochastically collapses to one Kraus branch (the
quantum-jump / Monte-Carlo wavefunction method, Nielsen & Chuang §8; Dalibard–
Castin–Mølmer 1992). The ensemble of `shots` final measurements reproduces the
noisy density matrix's measurement distribution to Monte-Carlo accuracy.

- **Memory:** O(2ⁿ) — one statevector per shot, the same as the noiseless SV
  backend. (A density-matrix backend would be O(4ⁿ), ~14-qubit ceiling.)
- **Time:** ∝ shots × circuit. Embarrassingly parallel across shots.
- **Trade-off vs density matrix:** trajectories are sampling-based (a
  distribution, not the exact ρ) and pay per-shot cost, but stay at the SV
  memory ceiling (~28+ qubits) and need no new O(4ⁿ) backend. Aer ships both and
  defaults to trajectories for shot-based sampling; we follow. DM is revisited
  only if a user needs exact ρ / expectation values under noise at small n.

**Determinism:** the per-shot RNG is seeded `seed_shot = hash(seed, shot_index)`,
so shots are reproducible regardless of parallel scheduling and `seed` →
identical counts.

**Parallelism:** rayon over shots (each shot owns its statevector + RNG). The
noiseless `run()` path is **untouched** — noise is a distinct entry point
(`run_noisy`), so the "noiseless performance unchanged" guard is structural, not
a measured coincidence.

## 2. Architecture — `aleph_sv::noise`, IR stays noise-free

**Decision: a SV-specific noise driver in a new `aleph_sv::noise` module. No new
crate (YAGNI); no `Backend` trait change.**

The driver operates directly on `CpuState` because trajectory channels need
amplitude-level primitives the generic `Backend` trait doesn't expose: applying
an arbitrary (possibly non-unitary) 2×2/4×4 Kraus operator and reading the
post-application norm. v1 noise is SV-only, so a SV-specific driver is the
simplest correct home; if DM/MPS noise is ever added, the `NoiseModel` types
hoist to a shared crate at that point.

Module contents:
- `NoiseModel`, `QuantumError`, `ReadoutError`, channel constructors (data;
  config objects, **not** IR).
- `run_noisy(circuit, &NoiseModel, shots, seed) -> Counts` — the Monte-Carlo
  driver.
- `apply_channel(state, &QuantumError, qubits, rng)` — quantum-jump application.

`NoiseModel` is a **runtime configuration** consumed by the driver. It is never
an `aleph_ir::Instruction`; the IR and all existing backends remain noise-free.
The noiseless `run()`/`run_optimized()` drivers are unchanged.

Driver loop (one shot):
```
state ← |0…0⟩;  rng ← seed_shot
for inst in circuit.instructions():
    apply inst to state (unitary, via existing kernels)
    for err in noise_model.errors_for(inst.gate_kind, inst.qubits):
        apply_channel(state, err, inst.qubits, rng)
measure each qubit in Z; apply readout error per qubit (rng)
record bitstring
```
(`Reset`/mid-circuit `Measure` instructions: see §3 "measurement & reset".)

## 3. Noise IR — channel set v1 and attachment

### Channel set v1
| channel | arity | Kraus / action |
| --- | --- | --- |
| depolarizing | 1q, 2q | ρ→(1−p)ρ + p·I/d, d=2ⁿ — exactly Aer's `depolarizing_error(p, n)` parameterization (the impl matches Aer's convention so the oracle's model is byte-identical); as a Pauli mixture, identity with weight 1−p·(d²−1)/d² and each of the d²−1 non-identity Paulis with weight p/d² |
| bit-flip (X), phase-flip (Z), Y-flip | 1q | with prob p apply the named Pauli |
| amplitude damping | 1q | K₀=diag(1,√(1−γ)), K₁=√γ·|0⟩⟨1| |
| phase damping | 1q | K₀=diag(1,√(1−λ)), K₁=diag(0,√λ) |
| readout error | per measured qubit | flip the classical outcome: asymmetric P(1\|0), P(0\|1) |

All gate-attached channels are general `QuantumError` = a list of `(Kraus
operator, ...)` defining a CPTP map; the constructors above produce them.
Readout error is a separate measurement-time object.

### Application (quantum-jump)
For a `QuantumError` with Kraus set {Kᵢ} on the target qubits:
1. Compute pᵢ = ‖Kᵢ|ψ〉‖² for each branch (Σpᵢ = 1 by CPTP).
2. Sample branch i with probability pᵢ (one RNG draw).
3. Apply Kᵢ to |ψ〉 and renormalize by 1/√pᵢ.

**Pauli fast path:** when every Kᵢ is √(probᵢ)·(Pauli), pᵢ = probᵢ is
state-independent, so the branch is sampled directly from the fixed weights and
the chosen Pauli is applied with the existing unitary 1q/2q kernels — no norm
computation, no renormalization. Depolarizing and the flips take this path;
amplitude/phase damping take the general path.

### Attachment model (Aer-style)
`NoiseModel` holds:
- a map `(gate_kind, qubit-tuple) → Vec<QuantumError>` for qubit-specific
  attachments, plus a `(gate_kind) → Vec<QuantumError>` map for "all-qubit"
  attachments (applied to whichever qubits the gate acts on);
- a per-qubit `ReadoutError` map.

`errors_for(gate_kind, qubits)` returns the concatenation of the all-qubit and
the qubit-specific lists, applied in insertion order **after** the gate. This
mirrors Aer's `add_quantum_error` / `add_all_qubit_quantum_error`.

### Measurement & reset
v1 targets the shot-sampling path: gates then a terminal Z-measurement of all
qubits with readout error (the `Backend::sample` shape). Mid-circuit `Measure`
and `Reset` instructions in the trajectory driver are a documented v1.1 follow-up
(they require per-shot collapse + classical record); the v1 oracle fixtures use
terminal measurement, matching Aer's `counts` sampling.

## 4. API surface — Aer-compatible

### Python (pyo3)
```python
import aleph
nm = aleph.NoiseModel()
nm.add_all_qubit_quantum_error(aleph.depolarizing_error(0.01, 1), ["h", "x"])
nm.add_quantum_error(aleph.depolarizing_error(0.02, 2), ["cx"], [0, 1])
nm.add_quantum_error(aleph.amplitude_damping_error(0.05), ["id"], [0])
nm.add_readout_error([[0.98, 0.02], [0.03, 0.97]], [0])   # [P(0|0),P(1|0)],[P(0|1),P(1|1)]
counts = aleph.run(circuit, shots=100_000, noise=nm, seed=7)
```
Error factories mirror Aer names: `depolarizing_error(p, num_qubits)`,
`amplitude_damping_error(gamma)`, `phase_damping_error(lam)`,
`pauli_error([("X", px), ("I", 1-px)])`. `aleph.run(..., noise=nm)` dispatches to
`run_noisy`; without `noise=` it is the existing noiseless path.

### CLI
`aleph run circuit.qasm --shots N --noise depolarizing:0.01` — preset flags for
the common single-parameter channels (depolarizing, readout). Full `NoiseModel`
construction stays in the Python API; the CLI exposes presets only.

## 5. Frame-sampler integration (P4.6-02)

Pauli channels (depolarizing, bit/phase flip) are exactly what a Pauli-frame
simulator absorbs nearly for free: a Pauli error on qubit q is an X/Z frame
injection at that point in the circuit, propagated word-parallel across shots
with no per-shot statevector. For **Clifford** circuits with **Pauli-only**
noise, `run_noisy` could route to the P4.6-02 frame sampler instead of the SV
trajectory loop — a large speedup for QEC noise sweeps. This is **out of scope
for v1** (SV trajectory engine first) but the channel representation is chosen to
make it a clean later addition: `QuantumError`'s Pauli fast-path already exposes
"sample a Pauli per application", which the frame sampler consumes directly. The
`measure_word` seam noted in P4.6-02 is where the frame injection hooks.

## Oracle protocol (→ P4.6-04 testing section)

For each fixture `(circuit, NoiseModel)`:
1. Build the **byte-identical** model in Qiskit Aer (same channel params, same
   gate/qubit attachment, same readout matrices).
2. Run aleph `run_noisy` and Aer at **100k shots**, same logical seed where Aer
   allows.
3. Compare the two count distributions over 2ⁿ outcomes with the calibrated 5σ
   band (`aleph_oracle::assert_distribution_close`, P3-16) at the agreed 1e-5
   tolerance.

Fixture set (small n so 100k shots resolves the distribution):
- depolarizing on `H` (1q) and on `CX` (2q);
- amplitude damping and phase damping on an idle/`id` qubit after `H`;
- readout error (asymmetric) on a deterministic state;
- a combined fixture (depol + readout) on a 3-qubit GHZ.

Property tests (no Aer needed):
- **CPTP sanity:** each channel's Σ pᵢ = 1 (within 1e-12) on random input states;
- **trace preservation:** ‖state‖ = 1 after `apply_channel` + renormalize;
- **deterministic seeding:** same seed → identical counts;
- **noiseless guard:** an empty `NoiseModel` reproduces the noiseless
  distribution; the noiseless `run()` criterion benchmark is unchanged.

## Re-spec of downstream tickets

### P4.6-04 (impl, #167)
Build `aleph_sv::noise`: `NoiseModel`/`QuantumError`/`ReadoutError` + the v1
channel constructors; `apply_channel` (quantum-jump + Pauli fast-path);
`run_noisy` (rayon over shots, per-shot seeding); readout error at terminal
sampling. AC: channel set v1 end-to-end on SV; Aer oracle 1e-5 @ 100k on the
fixture set above; deterministic seeding; noiseless criterion guard unchanged;
CPTP/trace property tests.

### P4.6-05 (Python/CLI, #168)
pyo3 `NoiseModel` + error factories (Aer names) + `aleph.run(..., noise=)`; CLI
`--noise <preset>:<p>`; `scripts/python/test_aleph.py` coverage; README + crate-
README examples; release-notes entry. AC: Python API per this spec with tests;
CLI depolarizing preset; docs updated.

## Non-goals (v1)
- Density-matrix backend; exact ρ / noisy expectation values.
- Mid-circuit measurement/reset under noise (v1.1).
- Frame-sampler routing for Clifford+Pauli noise (later; representation ready).
- Coherent/parametric error calibration, cross-talk, non-Markovian noise.
