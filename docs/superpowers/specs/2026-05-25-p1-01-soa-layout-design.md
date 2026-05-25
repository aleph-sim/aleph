# P1-01 — Struct-of-Arrays memory layout for state-vector amplitudes

**Issue:** P1-01 (see `BACKLOG.md`, GitHub #13)
**Depends on:** P0-09 (Backend trait + NaiveSvBackend), P0-10 (oracle harness), P0-11 (measurement primitives)
**Date:** 2026-05-25

---

## 1. Goal

Ship a second state-vector backend, `SoaSvBackend`, that stores
amplitudes as two parallel `Vec<f64>` (real, imaginary) rather than
`Vec<num_complex::Complex<f64>>` (array-of-structs).  Both backends
live in `aleph-sv`; the existing `NaiveSvBackend` is preserved as
the reference oracle for correctness comparisons.  All four
measurement primitives are ported; all 28 oracle fixtures pass on
both backends; the workhorse equivalence test pins SoA ≡ AoS to
within `1e-12` on every Phase-0 fixture.

SoA is groundwork for the rest of Phase 1: P1-02 (bit-manipulation
indexing), P1-03/P1-04 (AVX2/AVX-512 SIMD), P1-05/P1-06 (specialised
Pauli-X and diagonal kernels) all expect the SoA layout because
their per-lane vectorisation requires sequential `f64` reads, not
strided 16-byte Complex reads.

This ticket targets the **layout-only** win — naive port, no SIMD,
no bit-manipulation trickery.  Acceptance is `~1.5–2×` on QFT-20
purely from improved cache behaviour (two sequential prefetch
streams instead of one strided one).  If the layout-only port lands
below 1.5× on the canonical bench server, this is documented in
the PR description and addressed by P1-02 / P1-03 rather than
blocking the merge — correctness gates always come first.

---

## 2. Scope

### In scope

- New `crates/aleph-sv/src/soa_state.rs`: `SoaState { num_qubits, re: Vec<f64>, im: Vec<f64> }`.
- New `crates/aleph-sv/src/soa_backend.rs`: `SoaSvBackend` implementing the full `Backend` trait.
- Split `crates/aleph-sv/src/kernels.rs` into a module
  `crates/aleph-sv/src/kernels/{mod.rs, aos.rs, soa.rs}`. `aos.rs`
  is a verbatim move of the current file; `soa.rs` is new.
- New `crates/aleph-sv/src/measure_soa.rs` with SoA versions of
  `validate_state_soa`, `measure_impl_soa`, `sample_impl_soa`,
  `expectation_value_impl_soa`, `probabilities_impl_soa`.
- `HasAmplitudes` trait in `aleph-oracle/src/state.rs`: signature
  changes from `fn amplitudes(&self) -> &[Complex]` to
  `fn amplitudes(&self) -> Vec<Complex>` (owned).  AoS impl
  becomes `self.amps.clone()`; SoA impl uses `to_aos()`.
- `aleph-oracle/build.rs`: emit `naive_state`, `naive_distribution`,
  `soa_state`, `soa_distribution` per fixture (4 tests × 28 = 112).
- `crates/aleph-sv/tests/soa_vs_naive.rs`: workhorse equivalence
  test over every committed oracle circuit.
- `crates/aleph-sv/benches/soa_vs_naive.rs`: side-by-side criterion
  benches (`qft/n10`, `qft/n15`, `qft/n20`, `ghz/n20`) with
  `BenchmarkId::new(...)` taking backend name so bencher.dev shows
  paired bars.
- `docs/testing.md`: new "SoA backend (P1-01)" subsection.

### Out of scope (deferred)

- **Bit-manipulation indexing** (loop over `(i_high, i_low)` rather
  than `for i in 0..dim { let j = i ^ (1<<t); … }`). P1-02.
- **SIMD: AVX2 / AVX-512**. P1-03, P1-04.
- **Specialised Pauli-X / diagonal kernels**. P1-05, P1-06.
- **Replacing AoS NaiveSvBackend**. BACKLOG explicitly says "kept
  for naive backend (reference)".
- **`#[cfg(feature = "soa")]` flag**. Both backends always
  compile.
- **Generic-over-storage abstraction** (`Backend<Storage = …>`).
  YAGNI — two impls is simpler than a generic that has to satisfy
  both, and a third in-memory layout doesn't appear on the roadmap
  before P5 (GPU).
- **Cache-blocking of 2q / 3q kernels.** Revisit if QFT-20 falls
  short of the 1.5× AC.
- **Multi-threading.** Phase 2.

---

## 3. Architecture

```
crates/aleph-sv/src/
├── lib.rs              re-export NaiveSvBackend, SoaSvBackend, CpuState, SoaState
├── state.rs            UNCHANGED — CpuState (AoS)
├── soa_state.rs        NEW — SoaState { num_qubits, re, im }
├── backend.rs          UNCHANGED — NaiveSvBackend
├── soa_backend.rs      NEW — SoaSvBackend (Backend impl)
├── kernels/            NEW module dir (was kernels.rs)
│   ├── mod.rs          `pub mod aos; pub mod soa;` + shared helpers
│   ├── aos.rs          moved from kernels.rs verbatim
│   └── soa.rs          NEW — apply_1q/2q/3q on (&mut [f64], &mut [f64])
├── measure.rs          UNCHANGED — measure/sample/etc. on CpuState
├── measure_soa.rs      NEW — *_impl_soa on SoaState
└── sampling.rs         UNCHANGED — AliasTable is layout-agnostic
```

`AliasTable::build(&[f64])` accepts a probability slice regardless
of source layout, so `sample_impl_soa` reuses it directly.

Dep graph: no new crates.  `aleph-oracle` gains a new
`HasAmplitudes` impl alongside its existing one.

---

## 4. `SoaState`

```rust
// crates/aleph-sv/src/soa_state.rs

#[derive(Debug, Clone)]
pub struct SoaState {
    pub(crate) num_qubits: u32,
    pub(crate) re: Vec<f64>,
    pub(crate) im: Vec<f64>,
}

impl SoaState {
    pub fn num_qubits(&self) -> u32 { self.num_qubits }
    pub fn re(&self) -> &[f64] { &self.re }
    pub fn im(&self) -> &[f64] { &self.im }

    /// Materialise as `Vec<Complex>` for oracle / interop paths.
    /// Allocates `2^num_qubits` Complexes.  NOT for hot paths —
    /// the SoA backend's primitives operate on `re`/`im` directly.
    pub fn to_aos(&self) -> Vec<Complex> {
        self.re.iter().zip(self.im.iter())
            .map(|(&r, &i)| Complex::new(r, i))
            .collect()
    }
}
```

### Invariants (enforced at the validation boundary)

`validate_state_soa(state) -> Result<Vec<f64>, BackendError>` is
the parallel of `measure.rs::validate_state`:

1. `state.re.len() == state.im.len() == 1usize << state.num_qubits`
   (structural; catches direct field-literal corruption — see ADR
   0005 trust boundary).
2. `state.num_qubits` fits in `usize::BITS` (handled by `checked_shl`).
3. Every `re[i]` and `im[i]` is finite (ADR 0006: explicit
   `is_finite()` guard before any FP comparison).
4. `Σ (re[i]² + im[i]²)` within `√n · AMPLITUDE_TOL` of 1.0.

Returns the per-amp probability vector for the caller so SoA's
measure / sample / probabilities / expectation_value primitives
share a single pass — same pattern as the AoS path.

### Memory budget

Two `Vec<f64>` of `1 << num_qubits` each: total `2 · 8 = 16` bytes
per amplitude, identical to `Vec<Complex>`.  Soft cap
`MAX_SOA_QUBITS = 28` mirrors `MAX_NAIVE_QUBITS = 28` (4 GiB per
backend).

---

## 5. SoA kernels (`kernels/soa.rs`)

Naïve port of the AoS algorithm — no SIMD, no bit-manipulation
trickery (those land in P1-02 / P1-03).

### 5.1 `apply_1q`

```rust
pub fn apply_1q(
    re: &mut [f64],
    im: &mut [f64],
    target: u32,
    controls: &[u32],
    matrix: &[[Complex; 2]; 2],
)
```

Inner loop walks pairs `(i0, i1)` where `i1 = i0 ^ (1 << target)`
and `i0 < i1` (so each pair is visited once).  Control mask: skip
the pair unless every control bit is set in `i0`.  For each kept
pair:

```text
let m00 = matrix[0][0]; let m01 = matrix[0][1];
let m10 = matrix[1][0]; let m11 = matrix[1][1];
let a0_re = re[i0]; let a0_im = im[i0];
let a1_re = re[i1]; let a1_im = im[i1];
// new amplitudes:
re[i0] = m00.re*a0_re - m00.im*a0_im + m01.re*a1_re - m01.im*a1_im;
im[i0] = m00.re*a0_im + m00.im*a0_re + m01.re*a1_im + m01.im*a1_re;
re[i1] = m10.re*a0_re - m10.im*a0_im + m11.re*a1_re - m11.im*a1_im;
im[i1] = m10.re*a0_im + m10.im*a0_re + m11.re*a1_im + m11.im*a1_re;
```

Eight real f64 multiplies + four adds per pair, same as the AoS
version but writing two streams.  The compiler can later
auto-vectorise this shape; P1-03's explicit AVX2 work targets it.

### 5.2 `apply_2q` and `apply_3q`

Same pattern, groups of 4 (`apply_2q`) and 8 (`apply_3q`) indices
per iteration.  Index sets are constructed from the AoS code's
`gen_indices_2q` / `gen_indices_3q` shape (qubit ordering per ADR
0004 MSB convention: target₀ is LSB of the index group).

4×4 / 8×8 matrix multiplication expanded to real arithmetic — no
`Complex` arithmetic in the hot loop.

### 5.3 Control-qubit mask helper

`crate::kernels::control_mask(controls: &[u32]) -> u64` lives in
`kernels/mod.rs` because it's layout-agnostic (returns the
bitwise-OR of `1<<q` over controls).  Used by both `aos.rs` and
`soa.rs`.

### 5.4 Tests

Per `kernels/soa.rs`:

- Single-target 1q gates on `|0⟩`, `|1⟩`, `|+⟩`, `|−⟩`, `|i⟩`,
  `|−i⟩` for H, X, Y, Z, S, T, Sdg, Tdg, Rx(π/2), Ry(π/2),
  Rz(π/2), Phase(π/4) — direct amplitude assertions vs hand-
  derived values.
- CNOT on `|+0⟩` produces the Bell state `(|00⟩ + |11⟩)/√2`.
- 2-qubit Cz on `|11⟩` flips the phase to `−|11⟩`.
- 3-qubit Toffoli on `|110⟩` becomes `|111⟩` and vice versa.
- Equivalence proptest (the workhorse for kernel correctness):
  for any random 1q/2q/3q gate from
  `aleph_test::gate::arb_*_gate` × random qubit selection ×
  random normalised initial state from
  `aleph_test::state::arb_state_vector`, applying via SoA kernel
  matches AoS kernel within `1e-12` after `to_aos()`.

---

## 6. `SoaSvBackend` + measurement primitives

```rust
// crates/aleph-sv/src/soa_backend.rs

pub struct SoaSvBackend {
    pub(crate) rng: StdRng,
}

impl SoaSvBackend {
    pub fn new() -> Self { Self { rng: StdRng::from_entropy() } }
    pub fn with_seed(seed: u64) -> Self { Self { rng: StdRng::seed_from_u64(seed) } }
}

impl Default for SoaSvBackend {
    fn default() -> Self { Self::new() }
}

pub(crate) const MAX_SOA_QUBITS: u32 = 28;

impl Backend for SoaSvBackend {
    type State = SoaState;

    fn allocate(&mut self, num_qubits: u32) -> Result<SoaState, BackendError> {
        if num_qubits > MAX_SOA_QUBITS {
            return Err(BackendError::TooManyQubits { requested: num_qubits, limit: MAX_SOA_QUBITS });
        }
        let dim = 1usize << num_qubits;
        let mut re = vec![0.0; dim];
        let im = vec![0.0; dim];
        re[0] = 1.0;
        Ok(SoaState { num_qubits, re, im })
    }

    fn apply_gate(&mut self, state: &mut SoaState, gate: &GateInstance)
        -> Result<(), BackendError>
    {
        // Identical validation block to NaiveSvBackend:
        //   - arity check
        //   - bounds + duplicate check across qubits ∪ controls
        //   - materialise matrix, route GateError variants
        //   - unitarity check (defense-in-depth, ADR 0006)
        // Then dispatch to kernels::soa::apply_{1q,2q,3q}.
    }

    fn measure(&mut self, state: &mut SoaState, qubit: u32) -> Result<bool, BackendError> {
        crate::measure_soa::measure_impl_soa(&mut self.rng, state, qubit)
    }
    fn sample(&mut self, state: &SoaState, shots: u32) -> Result<Vec<u64>, BackendError> {
        crate::measure_soa::sample_impl_soa(&mut self.rng, state, shots)
    }
    fn expectation_value(&mut self, state: &SoaState, pauli: &PauliString)
        -> Result<f64, BackendError>
    {
        crate::measure_soa::expectation_value_impl_soa(state, pauli)
    }
    fn probabilities(&mut self, state: &SoaState, qubits: &[u32])
        -> Result<Vec<f64>, BackendError>
    {
        crate::measure_soa::probabilities_impl_soa(state, qubits)
    }
}
```

### 6.1 `measure_soa.rs` primitives

Each mirrors the AoS implementation 1-to-1:

| AoS | SoA | Notes |
|---|---|---|
| `validate_state` | `validate_state_soa` | Same shape, computes per-amp prob from `re[i]² + im[i]²`. Same drift budget, same NaN discipline. |
| `measure_impl` | `measure_impl_soa` | On collapse: write `re[i] = 0.0; im[i] = 0.0` for rejected branch; multiply kept branch by `1/√p`. Same RNG / degenerate-branch handling. |
| `sample_impl` | `sample_impl_soa` | `AliasTable::build(&probs)` (layout-agnostic), draw loop unchanged. |
| `expectation_value_impl` | `expectation_value_impl_soa` | Z fast path uses `probs` (same as AoS post-review-fix). Slow path: clone `re`+`im`, apply each non-I Pauli matrix via SoA 1q kernel, accumulate `Σ (re[i]·new_re[i] + im[i]·new_im[i])` (real part of conjugate product). |
| `probabilities_impl` | `probabilities_impl_soa` | Marginal sum over qubit subset; same indexing logic as AoS. |

### 6.2 Q-bit ordering

ADR 0004 MSB convention applies unchanged: `amps[i]` (or
equivalently `(re[i], im[i])`) corresponds to the basis state
where qubit `q` has value `(i >> q) & 1`.

---

## 7. Oracle integration

### 7.1 `HasAmplitudes` trait signature change

`aleph-oracle/src/state.rs` changes from:
```rust
pub trait HasAmplitudes {
    fn amplitudes(&self) -> &[Complex];
}
```
to:
```rust
pub trait HasAmplitudes {
    /// Returns the state as an owned `Vec<Complex>`.  AoS backends
    /// clone the buffer; SoA backends materialise via `to_aos()`.
    /// Only oracle-path code calls this; hot paths use the
    /// backend's native layout.
    fn amplitudes(&self) -> Vec<Complex>;
}
```

Impls:
```rust
impl HasAmplitudes for aleph_sv::CpuState {
    fn amplitudes(&self) -> Vec<Complex> { self.amps.clone() }
}
impl HasAmplitudes for aleph_sv::SoaState {
    fn amplitudes(&self) -> Vec<Complex> { self.to_aos() }
}
```

Sole consumer (`harness::assert_state_close`) is updated to take
`Vec<Complex>` by value — one-line change, no behavior delta.

**Clone cost on oracle path:** AoS clone is `1024 × 16 B = 16 KB`
at the largest existing fixture (`ghz_10`).  Negligible.

### 7.2 `build.rs` codegen upgrade

Per fixture, emit:
```rust
mod <stem> {
    #[test] fn naive_state() {
        let fx = aleph_oracle::load_fixture(...).expect(...);
        let qasm = aleph_oracle::load_qasm(...).expect(...);
        let mut backend = aleph_sv::NaiveSvBackend::with_seed(0);
        aleph_oracle::run_state_oracle(&mut backend, &fx, &qasm).expect("oracle");
    }
    #[test] fn naive_distribution() {
        // same, with run_distribution_oracle
    }
    #[test] fn soa_state() {
        let fx = aleph_oracle::load_fixture(...).expect(...);
        let qasm = aleph_oracle::load_qasm(...).expect(...);
        let mut backend = aleph_sv::SoaSvBackend::with_seed(0);
        aleph_oracle::run_state_oracle(&mut backend, &fx, &qasm).expect("oracle");
    }
    #[test] fn soa_distribution() {
        // same, with run_distribution_oracle on SoaSvBackend
    }
}
```

28 fixtures × 4 = 112 generated tests (up from 56).  The
`naive_state` / `naive_distribution` names break the previous
`state` / `distribution` naming — minor test-id churn, but a
follow-up of P0-12's "Closes #<issue>" lesson: anyone depending
on the old test IDs gets a clean compile-time / cargo-test
failure rather than a silent skip.

### 7.3 `run_distribution_oracle` generalisation

Currently the function has `B::State = aleph_sv::CpuState` as a
hard bound.  Generalise to:
```rust
pub fn run_distribution_oracle<B>(...)
where
    B: Backend,
    B::State: HasAmplitudes,
```

This lets both backends use the same harness.  P0-11 spec § 6.4
noted this generalisation as deferred until a second backend
landed — P1-01 *is* that second backend.

---

## 8. Bench strategy

`crates/aleph-sv/benches/soa_vs_naive.rs` (NEW):

```rust
use criterion::{BenchmarkId, Criterion};

fn bench_qft_per_backend(c: &mut Criterion) {
    let mut group = c.benchmark_group("qft");
    for &n in &[10u32, 15, 20] {
        let circuit = qft_circuit(n);  // shared helper in tests dir or lib
        group.bench_with_input(BenchmarkId::new(format!("n{n}"), "naive"), &n, |b, _| {
            b.iter_with_setup(
                || NaiveSvBackend::with_seed(0),
                |mut backend| { aleph_backend::run(&mut backend, &circuit).unwrap(); },
            );
        });
        group.bench_with_input(BenchmarkId::new(format!("n{n}"), "soa"), &n, |b, _| {
            b.iter_with_setup(
                || SoaSvBackend::with_seed(0),
                |mut backend| { aleph_backend::run(&mut backend, &circuit).unwrap(); },
            );
        });
    }
    group.finish();
}
```

`BenchmarkId::new(group, parameter)` shape produces bencher.dev
side-by-side bars for `qft/n10/naive` vs `qft/n10/soa`, etc.
Same shape for `ghz/n20`.

**Acceptance:** `qft/n20/soa` ≥ 1.5× faster than `qft/n20/naive`
on the canonical bench server (per BACKLOG AC).  Layout-only
naive port may land at 1.2–1.5× on local M-series; the EPYC
server is the source of truth via bencher.dev.

PR description includes both local-dev and CI-server numbers if
available at merge time.

---

## 9. Testing strategy

### 9.1 Workhorse equivalence

`crates/aleph-sv/tests/soa_vs_naive.rs` (NEW):

```rust
#[test]
fn all_fixtures_match_naive() {
    for fx_name in [/* all 28 fixture stems */] {
        let qasm = aleph_oracle::load_qasm(
            &aleph_oracle::workspace_path(&format!("oracle/circuits/{fx_name}.qasm"))
        ).unwrap();
        let circuit = aleph_parser::parse(&qasm).unwrap();

        let mut naive = NaiveSvBackend::with_seed(0);
        let naive_state = aleph_backend::run(&mut naive, &circuit).unwrap();

        let mut soa = SoaSvBackend::with_seed(0);
        let soa_state = aleph_backend::run(&mut soa, &circuit).unwrap();

        let naive_amps = naive_state.amplitudes();
        assert_eq!(naive_amps.len(), soa_state.re().len());
        assert_eq!(naive_amps.len(), soa_state.im().len());
        for i in 0..naive_amps.len() {
            let a = naive_amps[i];
            let r = soa_state.re()[i];
            let im = soa_state.im()[i];
            let dr = a.re - r;
            let di = a.im - im;
            let delta = (dr * dr + di * di).sqrt();
            assert!(
                delta < 1e-12,
                "fixture {fx_name} amp[{i}]: naive ({}, {}) vs soa ({r}, {im}); |Δ| = {delta:.3e}",
                a.re, a.im
            );
        }
    }
}
```

This is the workhorse — catches kernel regressions without needing
the Qiskit oracle.  Runs in ~1 s for all 28 fixtures (max n=10).

### 9.2 Proptests (`kernels/soa.rs::tests`)

- `apply_1q_soa_matches_aos(gate in arb_1q_gate(), q in 0u32..5, amps in arb_state_vector(5))`
- `apply_2q_soa_matches_aos(gate in arb_2q_gate(), (t0, t1) in distinct_pair(5), amps in arb_state_vector(5))`
- (3q analogue is optional — Toffoli/Ccz are the only 3q variants
  and they're already exercised by the full-circuit equivalence
  test in §9.1)
- `random_circuit_soa_matches_aos(c in arb_circuit_full(5, 0, 12), amps in arb_state_vector(5))`

### 9.3 SoA-specific unit tests (`soa_backend.rs::tests`)

Parallel each existing `NaiveSvBackend` test:
- `allocate_initialises_zero_ket`: `re[0] = 1.0`, others 0; `im` all 0
- `allocate_rejects_too_many_qubits`: 29 → TooManyQubits
- `apply_h_on_zero_yields_plus`
- `apply_cnot_creates_bell`
- `measure_zero_state_returns_false` + `measure_plus_state_collapses_to_basis`
- `sample_zero_state_only_returns_zero` + `sample_bell_state_only_returns_00_or_11`
- `expectation_z_on_zero_is_plus_one` + Z-chain on GHZ
- `expectation_x_on_plus_is_plus_one` (slow-path coverage)
- `probabilities_zero_state`
- `apply_gate_arity_mismatch_rejected` + `apply_gate_non_unitary_user_matrix_rejected` (validation parity)
- NaN-rejection parity: `measure_rejects_nan_amplitude_state` etc.

### 9.4 Integration

The 112 generated oracle tests (§7.2) and the workhorse
equivalence test (§9.1) together cover every Phase-0 fixture
through both backends.  Wall-time impact: oracle ~0.55 s → ~1.1 s
on the workspace `cargo test` run.  Negligible.

---

## 10. Documentation

### `docs/testing.md` new section:

```markdown
## SoA backend (P1-01)

`aleph-sv` ships two state-vector backends:

* `NaiveSvBackend` — reference, array-of-structs (`Vec<Complex<f64>>`).
  Stays as the correctness yardstick.
* `SoaSvBackend` — Phase-1 perf backend, struct-of-arrays
  (`Vec<f64>` × 2).  Same algorithms, layout chosen for SIMD-
  friendly memory access (P1-03 / P1-04 will add the explicit
  vectorisation).

Equivalence is pinned three ways:

1. `crates/aleph-sv/tests/soa_vs_naive.rs::all_fixtures_match_naive`
   — every committed oracle circuit produces the same state
   vector on both backends within 1e-12.
2. Proptest equivalence in `crates/aleph-sv/src/kernels/soa.rs`
   over `arb_1q_gate / arb_2q_gate / arb_circuit_full`.
3. Both backends pass the full oracle suite vs Qiskit Aer (the
   `build.rs` codegen emits `naive_*` and `soa_*` test variants
   per fixture).

When introducing a new state-vector backend (e.g. SIMD-specialised
variants in P1-03), add it to the workhorse equivalence test +
build.rs codegen rather than relying on the oracle alone.
```

---

## 11. Acceptance-criteria mapping (BACKLOG P1-01)

| BACKLOG AC | Where satisfied |
|---|---|
| SoA backend produces identical results to naive backend (≤1e-12 difference) | `tests/soa_vs_naive.rs::all_fixtures_match_naive` (28 fixtures) + per-arity equivalence proptests (§9.2) |
| Benchmark: SoA vs. naive on QFT-20 — expect ~1.5–2× improvement just from cache effects | `benches/soa_vs_naive.rs` ships the comparison; PR description reports the measured number on both local-dev and CI server (bencher.dev for canonical timeline) |
| All Phase 0 tests pass against SoA backend | 112 generated oracle tests (28 × {naive, soa} × {state, distribution}); `naive_*` half unchanged from P0-10/P0-11, `soa_*` half is new |

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| `qft/n20/soa` ≥ 1.5× target not met by naive layout-only port | Workhorse equivalence still passes → backend ships correct.  PR notes measured number (likely 1.2–1.5× on M-series, possibly more on EPYC); P1-02 / P1-03 close the gap.  Don't block merge on the speedup AC if correctness gates are met. |
| 4× growth in oracle test suite (56 → 112) slows CI | Current 56 fixtures run in ~0.55 s → ~1.1 s.  Below CI noise floor. |
| `HasAmplitudes` trait signature change breaks downstream consumers | Only consumer today is `aleph_oracle::harness`; refactor `assert_state_close` in the same PR.  External consumers (none yet) compile-fail loudly. |
| `to_aos()` allocation per oracle-path call | Largest fixture is `ghz_10` → 1024 Complexes ≈ 16 KB.  Hot paths use `re`/`im` slices directly and never touch `to_aos`. |
| Two `Vec<f64>` clones in `expectation_value_impl_soa` slow path vs one `Vec<Complex>` in AoS | Total memory identical (16 B/amp).  Two alloc calls instead of one is a measurable constant on tiny states but irrelevant on the n=10–20 bench range. |
| `aleph-cli`'s `--statevector` view assumes `Vec<Complex>` access | The CLI never holds a `SoaState` (`run_circuit` always allocates a `NaiveSvBackend` by name).  P1-01 doesn't add a `--backend soa` flag (spec §2 deferred); CLI is untouched. |
| Test-id churn: `<stem>::state` → `<stem>::naive_state` | Documented in §7.2.  Failure mode is loud (cargo test reports unknown test name) not silent.  P0-12's "Closes #<issue>" lesson applies in reverse: when changing test IDs, prefer breaking change over silent rename. |

---

## 13. Workflow notes

Standard P0-06…P0-12 workflow:

- Branch: `p1-01-soa-layout` (already created during spec-write).
- Implementation order (drives the plan):
  1. `soa_state.rs` + `validate_state_soa` + unit tests.
  2. `kernels/` module split + move `aos.rs` (no behavior change).
  3. `kernels::control_mask` shared helper.
  4. `kernels/soa.rs::apply_1q` + per-gate unit tests + equivalence proptest.
  5. `kernels/soa.rs::apply_2q` + tests + proptest.
  6. `kernels/soa.rs::apply_3q` + tests.
  7. `measure_soa.rs` (validate + measure + sample + expectation_value + probabilities) + unit tests.
  8. `soa_backend.rs` (Backend impl) + unit tests.
  9. `HasAmplitudes` trait signature change in `aleph-oracle` + AoS impl update.
  10. New `HasAmplitudes for SoaState` impl.
  11. `run_distribution_oracle` generalisation (`B::State: HasAmplitudes` bound).
  12. `oracle/build.rs` upgrade: emit 4 tests per fixture.
  13. `tests/soa_vs_naive.rs` workhorse equivalence test.
  14. `benches/soa_vs_naive.rs` perf comparison; capture numbers.
  15. `docs/testing.md` SoA section.
  16. BACKLOG ACs ticked.
  17. Final lint / fmt / workspace test sweep.

- PR title: `[P1-01] SoA memory layout for amplitudes`.
- PR body: `Closes #13` (BACKLOG P1-01 = GitHub issue #13).
  See P0-12 retro: use the issue number, not the PR self-ref.
- Squash-merge.
