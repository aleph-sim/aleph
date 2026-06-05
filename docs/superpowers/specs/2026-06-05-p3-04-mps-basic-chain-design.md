# P3-04 — MPS backend: basic 1D chain

**Issue:** #35 (`area:backend-mps`, `type:feature`, `priority:high`, `research`, XL)
**Milestone:** Phase 3
**Date:** 2026-06-05
**Status:** Approved (brainstorming)

## Goal

Implement a Matrix Product State (MPS) backend in `aleph-mps` for shallow /
structured circuits with bounded entanglement. This is the first of the MPS
chain (P3-04 → P3-05 → P3-06); it delivers the **basic 1D chain** with
**fixed bond dimension χ** truncation. It enables 100+ qubit simulation for
nearest-neighbor VQE/QAOA where the state vector cannot fit.

## Scope

### In scope (P3-04)

- Mixed-canonical MPS state with an orthogonality center.
- 1q gates (local contraction, no SVD).
- 2q **nearest-neighbor** gates (two-site contraction → 4×4 gate → SVD →
  fixed-χ truncation).
- **Fixed-χ truncation** only, but **accumulate and report** the discarded
  Schmidt weight (`trunc_error`).
- Full `Backend` trait impl: `allocate`, `apply_gate`, `measure` (collapse),
  `sample` (perfect sampling), `expectation_value`.
- CLI: `--backend mps` + `--max-bond <χ>`, with a dedicated `run_mps` path.

### Deferred (later tickets)

- **P3-05:** error-bounded truncation *mode* (choose χ from ε) + full error
  budgeting. P3-04 only tracks/reports discarded weight; it does not select χ
  from an error target.
- **P3-06:** non-adjacent 2q gates via SWAP networks. P3-04 **rejects**
  non-adjacent 2q gates with a clear error.
- **P3-07:** automatic backend selection.
- 3q+ gates (Toffoli/CCZ): rejected as unsupported (would need 3-site
  tensors; out of scope for "basic chain"). Tier-2 circuits in scope
  (VQE-H2, nearest-neighbor QAOA) contain only 1q + 2q gates.

## Key decisions (from brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Linalg / SVD library | **nalgebra** (pure Rust) | No system LAPACK dependency (vs `ndarray-linalg`); avoids the cross-platform / EPYC build pain documented for the project; aligns with the "minimize dependencies, no system deps" golden rule. Slower than LAPACK but correctness-first; revisit LAPACK in P3-05 if perf demands. |
| Truncation scope | **fixed-χ + report discarded weight** | Clean split with P3-05 (error-bounded *mode*), but `trunc_error` accumulator is cheap and useful now. |
| Non-adjacent 2q | **reject**; QAOA AC via nearest-neighbor topology | Keeps P3-04 inside "basic 1D chain"; SWAP networks are P3-06. |
| Canonical form | **mixed-canonical + orthogonality center** | Standard DMRG (Schollwöck 2011, the P3-05 reference); SVD truncation is locally optimal; avoids Vidal Γ-Λ division-by-small-Schmidt instability. |
| `measure` / `sample` | `measure` = marginal+collapse; `sample` = **perfect sampling (Ferris–Vidal 2012)** | `sample` takes `&State` (immutable) and avoids per-shot clones; `measure` (collapse) still required by the trait. |
| `probabilities` | **Implemented** — exact joint marginal over the requested subset via a doubled transfer-matrix sweep, with a subset-size cap | User-requested. Exact for any contiguous or non-contiguous subset; bounded by capping the subset size (output is 2^k). |

## Architecture

New dependencies for `aleph-mps`: `aleph-core`, `aleph-backend`, `aleph-ir`
(types referenced by trait default methods), `nalgebra`, `rand`.
`nalgebra` is added to `[workspace.dependencies]` (it re-exports
`num_complex::Complex` 0.4, identical to `aleph_core::Complex` — no bridging
conversions needed).

### Module layout

```
crates/aleph-mps/src/
  lib.rs       — crate doc, re-exports, MpsError enum
  tensor.rs    — Site (rank-3 tensor) + reshape ↔ nalgebra DMatrix helpers,
                 SVD-split, QR-move primitives
  mps.rs       — MpsState: canonicalization, gate application, expectation,
                 perfect sampling, measure+collapse
  gate.rs      — GateInstance → nalgebra 2×2 / 4×4 unitary extraction
  backend.rs   — MpsBackend impl Backend + MpsError → BackendError mapping
```

Files stay focused; each unit has one purpose and a clear interface.

### State representation

```rust
struct Site {            // rank-3 tensor (χ_L, 2, χ_R)
    left: usize,
    right: usize,
    data: Vec<Complex>,  // row-major: data[(l*2 + p)*right + r]
}

struct MpsState {
    sites: Vec<Site>,    // sites[i] shape (χ_i, 2, χ_{i+1}); χ_0 = χ_n = 1
    center: usize,       // orthogonality center site index
    max_bond: usize,     // χ cap
    trunc_error: f64,    // Σ over all truncations of discarded weight Σ_{k>χ} s_k²
}
```

**Canonical invariant:** sites left of `center` are left-canonical
(Σ_{l,p} A†A = I over the right bond), sites right of `center` are
right-canonical; the center site carries the norm.

**Initial state |0…0⟩:** every site is (1, 2, 1) = `[1, 0]`; all bonds χ = 1;
`center = 0`; `trunc_error = 0`. Trivially canonical.

We store flat `Vec<Complex>` with an explicit row-major layout that **we**
control, converting to/from `nalgebra::DMatrix` (column-major) only at SVD/QR
boundaries — this avoids column-major reshape confusion.

### Gate application

- **1q gate on site i:** `A'[l,p',r] = Σ_p U[p',p] · A[l,p,r]`. A unitary on
  the physical index preserves left/right-canonicality
  (Σ A'†A' = U† (Σ A†A) U = I), so **no SVD and no center move** are needed.

- **Center move (i → i±1):** QR decomposition (cheaper than SVD, no
  truncation). Moving right: group site i as (χ_L·2, χ_R) = Q·R, set site i =
  Q (left-canonical), contract R into site i+1's left bond. Move left is
  symmetric (RQ / QR on the transpose).

- **2q nearest-neighbor gate on (i, i+1):**
  1. Move center to i (so the two-site block has identity environment).
  2. Contract sites i, i+1 into Θ of shape (χ_L, 2, 2, χ_R).
  3. Apply the 4×4 gate: `Θ'[l,a',b',r] = Σ_{a,b} U[(a'b'),(ab)] Θ[l,a,b,r]`.
  4. Reshape Θ' to matrix M of shape (χ_L·2, 2·χ_R); SVD: M = U S Vᴴ.
  5. Keep top χ = min(rank, max_bond) singular values;
     `trunc_error += Σ_{k≥χ} s_k²`; renormalize the kept singular values so
     the state stays normalized.
  6. New site i = reshape(U[:, :χ]) → (χ_L, 2, χ), left-canonical;
     new site i+1 = reshape(S·Vᴴ) → (χ, 2, χ_R), now the center.

  When χ is large enough that nothing is discarded, this is exact (norm and
  amplitudes preserved to machine precision).

### Backend trait implementation

| Method | Behavior |
|---|---|
| `allocate(n)` | |0…0⟩ MPS; cap at 1024 qubits (`TooManyQubits` above). |
| `apply_gate` | Dispatch 1q / 2q-adjacent; `MpsError → BackendError` via `map_mps_err` (mirrors stabilizer `map_stab_err`). |
| `measure(q) -> bool` | Move center to q; single-site reduced ρ from the center contraction; `p(0)=ρ₀₀`, `p(1)=ρ₁₁`; sample with rng; project site q onto \|outcome⟩ and renormalize by `1/√p`. Both probs ≈ 0 → `DegenerateMeasurement`. |
| `sample(shots)` | **Perfect sampling**: clone once, canonicalize to right-canonical (center = 0), then for each shot sweep left→right with a left-boundary vector computing conditional probs (right environment is identity by right-canonicality) — no per-shot collapse/clone. Pack `1u64 << q` (matches SV/stabilizer); n ≤ 64 else `TooManyQubits`. |
| `expectation_value(P)` | Copy ψ′, apply each 1q Pauli to the relevant site's physical index, overlap ⟨ψ\|ψ′⟩ via a transfer-matrix sweep, multiply by the PauliString sign, return the real part (Hermitian ⇒ real). Closes the machine-precision AC. |
| `probabilities(qubits)` | Exact joint marginal over `qubits` (length 2^k). Validation mirrors the SV backend: empty subset → `[1.0]`, duplicate → `DuplicateQubit`, out-of-range → `QubitOutOfRange`. Output index bit `pos` corresponds to `qubits[pos]` (slice order, LSB-first) — identical contract to `aleph-sv`. Subset size capped (`MAX_PROB_QUBITS`, e.g. 20); larger → `TooManyQubits`. |
| `apply_diagonal_phase` / `apply_tiled_block` | Inherit trait defaults (MPS only sees raw, unoptimized circuits). |

**`probabilities` algorithm (doubled transfer-matrix sweep).** The state must
be normalized (it is, after truncation renormalization). The joint marginal
over subset S is the diagonal of ρ_S = Tr_{∉S}|ψ⟩⟨ψ|. Sweep sites left→right
maintaining a map from partial bit-pattern → boundary matrix E (χ_bra × χ_ket),
starting from the 1×1 scalar `[1]`:

- Site i ∉ S: contract over the physical index for every current env —
  `E' = Σ_p A[i]_pᴴ · E · A[i]_p`.
- Site i ∈ S: split each env into two — for p ∈ {0,1},
  `E_{pattern·p} = A[i]_pᴴ · E_pattern · A[i]_p` (appends one output bit).

At the end every env is 1×1; its scalar is the probability of that pattern.
Patterns are mapped to output indices using each S-site's position in the
`qubits` slice (slice order, not site order). Cost O(2^k · n · χ³); the cap
keeps it bounded.

### Error type

`aleph-mps` defines a crate-local `MpsError` (`thiserror`):
`NonNearestNeighbor`, `UnsupportedGate { kind }`, `QubitOutOfRange`,
`NonFiniteParam`, `TooManyQubits`, etc. `backend.rs` maps it to the shared
`BackendError`. Non-adjacent 2q maps to
`BackendError::InvalidState { reason: "non-adjacent 2q gate requires a SWAP network (see P3-06)" }`
(no new shared-enum variant needed; `reason` is `&'static str`).

### CLI

- `--backend {statevector, stabilizer, mps}` (extend the clap `ValueEnum`).
- New `--max-bond <χ>` flag (default 128).
- Dedicated `run_mps` path (analogous to `run_stabilizer`): `run_with_backend`
  requires `B::State: AmpsF64`, which `MpsState` cannot satisfy. The path:
  allocate → apply raw gates → `sample` → counts. Reject `--statevector` and
  other incompatible flags.

## Testing

Per `docs/testing.md` and CLAUDE.md:

1. **Unit:** 1q gates on |0⟩, Bell (H+CNOT), GHZ — reconstruct the dense state
   vector from the MPS (contract the chain, small n) and compare to textbook
   amplitudes.
2. **Oracle vs `NaiveSvBackend`** (in-process, as in P3-03): fixtures with only
   1q + nearest-neighbor 2q gates, χ large (no truncation) → amplitudes and
   `expectation_value` match SV to 1e-10.
3. **Property (`proptest`):**
   - Canonical invariant (left/right isometry) holds after random gate
     sequences.
   - Norm ≈ 1 when no truncation occurs.
   - χ = ∞ (large) reproduces the SV state exactly.
   - Weakly-entangled (low-depth nearest-neighbor) state with small χ is
     near-exact (small `trunc_error`); trivial truncation (χ ≥ full rank)
     preserves the state exactly.
4. **VQE H₂ @ 4 qubits:** matches the state vector to machine precision
   (χ unbounded).
5. **QAOA depth-3 @ 50 qubits, nearest-neighbor ring** (`#[ignore]` if > 30 s):
   runs to completion; "reasonable results" = norm preserved, `trunc_error`
   bounded, sampled distribution non-degenerate.
6. **Sampling:** perfect-sampling distribution agrees with measure-all and with
   SV probabilities for small n (total-variation distance within 1e-5 at
   100k shots).
7. **`probabilities`:** exact joint marginals match `NaiveSvBackend::probabilities`
   to 1e-10 for small n across single-qubit, contiguous, and non-contiguous
   subsets; empty subset → `[1.0]`; full subset sums to 1; the perfect-sampling
   empirical distribution converges to `probabilities` (cross-check); validation
   errors (duplicate, out-of-range, oversized subset) match the SV contract.

## Performance

P3-04 has no hard ratio AC (the AC is "produces reasonable results"). A
nearest-neighbor QAOA scaling bench (n vs wall-clock) is recorded for the
protocol but does not gate.

**Note:** nalgebra is pure Rust with no `is_x86_feature_detected!` SIMD-detect
path, so local aarch64 runs the *same* code as EPYC — unlike the SV kernels,
local testing is representative and EPYC validation is not required for
correctness.

## Decomposition (XL → ~12 tasks)

The implementation plan (writing-plans) will break this into roughly:

1. `Site` tensor type + reshape ↔ DMatrix helpers.
2. QR-based center-move canonicalization.
3. 1q gate application.
4. 2q nearest-neighbor application + SVD fixed-χ truncation + `trunc_error`.
5. Dense-statevector reconstruction helper (test support).
6. `expectation_value` (overlap sweep).
7. `measure` (marginal + collapse).
8. `sample` (perfect sampling) + `probabilities` (doubled transfer-matrix sweep).
9. `MpsBackend` impl `Backend` + `MpsError → BackendError`.
10. CLI `--backend mps` + `--max-bond` + `run_mps`.
11. Oracle tests + proptests.
12. VQE-H2 / NN-QAOA acceptance tests, bench, docs.

## References

- Vidal, "Efficient Classical Simulation of Slightly Entangled Quantum
  Computations" (2003) — MPS gate application & truncation.
- Schollwöck, "The density-matrix renormalization group in the age of matrix
  product states" (2011) — canonical forms, mixed-canonical, SVD truncation.
- Ferris & Vidal, "Perfect sampling with unitary tensor networks" (2012) —
  the `sample` algorithm.
- White, "Density Matrix Formulation for Quantum Renormalization Groups"
  (1992).
