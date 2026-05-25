# P0-11 — Measurement, sampling, and probability primitives

**Issue:** P0-11 (see `BACKLOG.md`)
**Depends on:** P0-09 (Backend trait + initial primitive impls), P0-10 (oracle harness + fixtures with `counts`)
**Date:** 2026-05-25

---

## 1. Goal

Complete and verify the four measurement primitives on `NaiveSvBackend`:
projective `measure`, multi-shot `sample`, marginal `probabilities`,
and Pauli `expectation_value`. P0-09 shipped functional implementations
of all four; P0-11 closes the remaining acceptance criteria:

- Switch `sample` from inverse-CDF (`O(log N)` per shot) to **Vose's
  alias method** (`O(1)` per shot).
- Add a **Pauli-Z fast path** to `expectation_value` (no state clone
  when every non-identity term is `Z`). Already TODO-tagged at
  `measure.rs:156`.
- Wire P0-10's dormant `fixture.counts` column to a new
  `run_distribution_oracle` so every committed fixture continuously
  validates aleph's sampling against the analytical
  distribution.
- Add the statistical convergence (`1M-shot Bell`) and `∑ probabilities
  = 1` property tests required by the BACKLOG.
- Land benchmarks that quantify the alias-vs-CDF and Z-vs-clone speedups.

This is materially a **completion ticket**, not a fresh implementation.

---

## 2. Scope

### In scope

- New module `crates/aleph-sv/src/sampling.rs` containing
  `pub(crate) struct AliasTable { prob: Vec<f64>, alias: Vec<u32> }`
  with `build(p: &[f64])` and `draw(&self, rng: &mut StdRng) -> u32`.
- Rewrite of `measure::sample_impl` to use `AliasTable`.
- Pauli-Z early-return branch in `measure::expectation_value_impl` plus
  a `fn expectation_z_diag(amps: &[Complex], z_mask: u64) -> f64`
  helper.
- New public function in `crates/aleph-oracle/src/harness.rs`:
  `pub fn run_distribution_oracle<B>(...) -> Result<(), OracleError>`
  with constants `DISTRIBUTION_SHOTS = 100_000` and
  `DISTRIBUTION_FLOOR = 1e-6`.
- `crates/aleph-oracle/build.rs` extended to emit two `#[test]`s
  per fixture inside a `mod <name>` (`state` and `distribution`)
  instead of a single test.
- Defensive checks added inline because the new oracle needs them
  (PR-69 PLAUSIBLE findings folded in):
  - `load_fixture` asserts `amplitudes.len() == 1usize << num_qubits`.
  - `build.rs` asserts every `oracle/circuits/<stem>.qasm` has a
    matching `oracle/fixtures/<stem>.json`.
- New tests:
  - `crates/aleph-sv/tests/sampling_convergence.rs` — Bell state at
    1M shots, 10σ band per outcome, zero on forbidden basis states.
  - `crates/aleph-sv/src/measure.rs` proptest — `∑ probabilities =
    1` for random Clifford+T circuits at `n ∈ [1, 6]`.
  - `crates/aleph-sv/src/sampling.rs` `#[cfg(test)]` — uniform,
    Bell, degenerate, single-qubit, and near-1 normalization
    cases for the alias table.
  - `expectation_value` tests for `⟨1|Z|1⟩`, the `Z⊗Z` sign table,
    `⟨0|Y|0⟩` (mixed-Pauli fallthrough), plus a proptest
    asserting Z-fast-path ≡ slow path within `1e-12` on random
    states.
- New benches:
  - `crates/aleph-sv/benches/sample.rs` — `uniform_n{4,10,16}` and
    `ghz_n10` at `1k` / `100k` shots.
  - `crates/aleph-sv/benches/expectation.rs` — `exp_z_chain_n10`,
    `exp_x_chain_n10`, `exp_mixed_zx_n10`.

### Out of scope (deferred)

- `measure_all()` convenience — `sample(state, 1)` covers it.
- `sample_qubits(qubits, shots)` marginal sampling — no consumer
  needs it until mid-circuit measurement (P0-13+).
- Rayon-parallel sampling — defer until P1-04 ("Multi-threading").
- `run_expectation_oracle` — defer until expectation values appear
  in a non-test caller (Phase 4 VQE).
- Dropping or asserting `fixture.counts` — kept as human-readable
  triage data; not part of any test.
- Backend coverage expansion (MPS / Stab / GPU) — these backends
  don't exist yet.

---

## 3. Architecture

```
┌─ aleph-sv ────────────────────────────────────────┐
│  src/measure.rs                                   │
│    validate_state              (unchanged)        │
│    measure_impl                (unchanged)        │
│    sample_impl                 rewrite: alias     │
│    expectation_value_impl      add Z fast path    │
│    probabilities_impl          (unchanged)        │
│                                                   │
│  src/sampling.rs               NEW                │
│    AliasTable                  Vose's method      │
└───────────────────────────────────────────────────┘
                          ▲
                          │ Backend::sample(N)
┌─ aleph-oracle ───────────┴────────────────────────┐
│  src/harness.rs                                   │
│    run_state_oracle            (unchanged)        │
│    run_distribution_oracle     NEW                │
│    DISTRIBUTION_SHOTS = 100_000                   │
│    DISTRIBUTION_FLOOR = 1e-6                      │
│                                                   │
│  src/fixture.rs                                   │
│    load_fixture                + 2^n shape check  │
│                                                   │
│  build.rs                                         │
│    + qasm↔fixture symmetry check                  │
│    emits two #[test] per fixture inside mod <name>│
└───────────────────────────────────────────────────┘
                          ▲
                          │
┌─ benches/{sample,expectation}.rs  NEW            ─┐
│  criterion baselines for alias vs CDF             │
│  and Z fast path vs clone                         │
└───────────────────────────────────────────────────┘
```

No new crates. No changes to the public `Backend` trait. The
`fixture.json` schema does not change.

---

## 4. Alias-method sampler (`crates/aleph-sv/src/sampling.rs`)

### 4.1 Public-to-crate API

```rust
pub(crate) struct AliasTable {
    /// `prob[i]` is the threshold at index `i`: draw u ∈ [0,1);
    /// if u < prob[i] return i, else return alias[i].
    prob: Vec<f64>,
    alias: Vec<u32>,
}

impl AliasTable {
    /// Build from a normalised probability vector. `p` must sum to
    /// `1 ± drift_budget`; callers go through `validate_state` first.
    pub(crate) fn build(p: &[f64]) -> Self { ... }

    /// One draw. Consumes two `f64`s of RNG output.
    pub(crate) fn draw(&self, rng: &mut StdRng) -> u32 { ... }
}
```

### 4.2 Build algorithm (Vose 1991)

1. Compute `scaled[i] = (n as f64) * p[i]` once.
2. Two index stacks: `small` (`scaled < 1`) and `large` (`scaled ≥ 1`).
3. While both non-empty: pop one of each. Let `s = small.pop()`,
   `l = large.pop()`. Set `prob[s] = scaled[s]`, `alias[s] = l`.
   Update `scaled[l] = (scaled[l] + scaled[s]) - 1.0`. If the new
   `scaled[l] < 1.0`, push `l` to small, else push to large.
4. Drain remaining stacks: each leftover index `i` gets
   `prob[i] = 1.0`, `alias[i] = i`.

Citation in code comment: `// Vose 1991, "A linear algorithm for
generating random numbers with a given distribution", Algorithm 3.`

### 4.3 Draw

```rust
let i = rng.gen_range(0..self.prob.len() as u32);
let u: f64 = rng.gen();
if u < self.prob[i as usize] { i } else { self.alias[i as usize] }
```

Two RNG calls per draw. The CDF path used one. This *changes* the
sequence of basis indices for a given seed — see §4.5.

### 4.4 Integration into `sample_impl`

```rust
pub(crate) fn sample_impl(
    rng: &mut StdRng,
    state: &CpuState,
    shots: u32,
) -> Result<Vec<u64>, BackendError> {
    let probs = validate_state(state)?;
    let table = crate::sampling::AliasTable::build(&probs);
    let mut out = Vec::with_capacity(shots as usize);
    for _ in 0..shots {
        out.push(table.draw(rng) as u64);
    }
    Ok(out)
}
```

### 4.5 RNG-determinism impact

The existing P0-09 test `sample_bell_state_only_returns_00_or_11`
asserts the *set* of outcomes (∈{0, 3}) and a rough split (each >
100 in 1000 shots). Both inverse-CDF and alias sampling satisfy
this — the test passes unchanged. No other existing test asserts
specific basis indices from `sample`.

The distribution oracle (§6) uses statistical tolerances per
outcome and does not rely on bit-exact RNG behaviour.

### 4.6 Unit tests in `sampling.rs`

- `uniform_8_outcomes_within_5_sigma_at_1m_draws`.
- `bell_only_returns_0_or_3`.
- `degenerate_1_0_0_0_always_returns_0`.
- `single_index_always_returns_0`.
- `near_normalised_1_plus_1e_minus_15_builds_and_draws`.

### 4.7 Expected speedup

At `n = 10`, `shots = 100_000`: inverse-CDF does ~`log₂(1024) = 10`
comparisons per shot; alias does one branch. ≥ 5× wall-time
reduction expected, quantified in §7.

---

## 5. Pauli-Z fast path (`crates/aleph-sv/src/measure.rs`)

### 5.1 Diagonal identity

For a Pauli string whose non-identity terms are all `Z`,

`⟨ψ|⊗ᵢ Zᵢ|ψ⟩ = Σᵢ (-1)^popcount(i & mask) · |aᵢ|²`

where `mask = ⊕_q (1 << q)` over qubits with `Pauli::Z`. No state
clone, no kernel apply, one pass over `amps`.

### 5.2 Branch shape

Insert after the existing `validate_state` + invariant checks in
`expectation_value_impl`:

```rust
let mut z_mask = 0u64;
let mut all_z_or_i = true;
for (q, p) in &pauli.terms {
    match p {
        Pauli::I => {}
        Pauli::Z => z_mask |= 1u64 << q,
        Pauli::X | Pauli::Y => { all_z_or_i = false; break; }
    }
}
if all_z_or_i {
    return Ok(pauli.coefficient * expectation_z_diag(&state.amps, z_mask));
}
// Existing copy-and-rotate path follows.
```

`expectation_z_diag` is a private free function in the same
module:

```rust
fn expectation_z_diag(amps: &[Complex], z_mask: u64) -> f64 {
    let mut acc = 0.0;
    for (i, a) in amps.iter().enumerate() {
        let sign = if (i as u64 & z_mask).count_ones() & 1 == 0 {
            1.0
        } else {
            -1.0
        };
        acc += sign * a.norm_sqr();
    }
    acc
}
```

### 5.3 Numerical and architectural notes

- `i` is at most `2^28` (capped by `MAX_NAIVE_QUBITS`); `i as u64`
  is exact.
- `count_ones()` lowers to `popcnt` on x86-64 (`target-cpu=native`
  default in our bench profile) and to a single instruction on
  aarch64.
- `mask == 0` (all-identity Pauli string) collapses to `Σ |aᵢ|² =
  1` modulo `validate_state`'s drift budget. The existing
  identity-only test path covers this case.

### 5.4 Tests (in `crates/aleph-sv/src/backend.rs`)

- `expectation_z_on_one_is_minus_one` — flip qubit then measure
  `⟨ψ|Z|ψ⟩ = -1`.
- `expectation_zz_sign_table` — for each of `|00⟩, |01⟩, |10⟩,
  |11⟩` confirm `⟨ψ|Z⊗Z|ψ⟩ ∈ {+1, -1, -1, +1}`.
- `expectation_y_on_zero_is_zero` — exercises the mixed-Pauli
  fallthrough (existing X-on-plus test does the same; the new Y
  case widens it).
- Property test in `measure.rs`: random concrete normalised state,
  random Pauli string with `terms` drawn from `{I, Z}`. Assert
  fast-path result equals a reference computation (the existing
  copy-and-rotate path) within `1e-12`.

---

## 6. Distribution oracle (`crates/aleph-oracle/src/harness.rs`)

### 6.1 Public API

```rust
pub const DISTRIBUTION_SHOTS: u32 = 100_000;
pub const DISTRIBUTION_FLOOR: f64 = 1e-6;

pub fn run_distribution_oracle<B>(
    backend: &mut B,
    fixture: &Fixture,
    qasm_source: &str,
) -> Result<(), OracleError>
where
    B: Backend<State = aleph_sv::CpuState>,
{
    let circuit = aleph_parser::parse(qasm_source)?;
    if circuit.num_qubits() != fixture.num_qubits {
        return Err(OracleError::QubitMismatch {
            name: fixture.name.clone(),
            fixture: fixture.num_qubits,
            circuit: circuit.num_qubits(),
        });
    }
    let state = run(backend, &circuit)?;
    let shots = backend.sample(&state, DISTRIBUTION_SHOTS)?;
    let dim = 1usize << fixture.num_qubits;
    let mut empirical = vec![0u64; dim];
    for s in &shots {
        empirical[*s as usize] += 1;
    }
    let exact: Vec<f64> = fixture
        .statevector
        .amplitudes
        .iter()
        .map(|&(re, im)| re * re + im * im)
        .collect();
    assert_distribution_close(
        &fixture.name,
        fixture.num_qubits,
        &empirical,
        &exact,
    );
    Ok(())
}
```

### 6.2 Tolerance

Per-outcome band, in probability units:

`band = 5 · √(p_exact · (1 - p_exact) / N) + DISTRIBUTION_FLOOR`

At `p = 0.5`, `N = 100k`: σ ≈ 1.6e-3, band ≈ 8e-3.
At `p = 0`, σ = 0, band = `DISTRIBUTION_FLOOR = 1e-6` — allows
≤ 0.1 stray counts on a forbidden outcome (effectively zero).

5σ per outcome → false-positive rate per outcome ≈ 5.7e-7. At the
largest fixture (`ghz_10`, 1024 outcomes), per-fixture flake
probability is ≤ 5.8e-4 (0.06%). Across 28 fixtures, ≤ 1.6% per CI
run with a fresh seed — acceptable, and we pin `seed = 0` so it's
deterministic anyway.

### 6.3 `assert_distribution_close` failure message

```
oracle: <name> distribution mismatch
  basis  |0000000000>
  exact  5.000000e-01
  empir  4.917000e-01   (49170 / 100000)
  |Δ|    8.300e-03   >  band 7.946e-03   (5σ + 1e-6)
```

Mirrors the state-vector oracle's structured message. First
out-of-band outcome aborts the test.

### 6.4 Why `B: Backend<State = aleph_sv::CpuState>`, not a generic bound

`run_state_oracle` uses `B::State: HasAmplitudes`. The distribution
oracle only needs `Backend::sample`, which is on the trait. Pinning
to `aleph_sv::CpuState` is unnecessary in principle but **today
only `NaiveSvBackend` exists**. Keeping the type signature concrete
matches the state-vector harness's "pinned to the only impl"
pattern and avoids a `HasSampling` micro-trait that exists for one
implementor. When the second backend lands, the bound generalises
in one place.

### 6.5 Wiring in `build.rs`

The generator currently emits one `#[test] fn <name>()` per
fixture. It will emit:

```rust
mod <name> {
    #[test]
    fn state() {
        let fx = aleph_oracle::load_fixture(&aleph_oracle::workspace_path(
            "oracle/fixtures/<name>.json",
        ))
        .expect("load fixture");
        let qasm = aleph_oracle::load_qasm(&aleph_oracle::workspace_path(
            "oracle/circuits/<name>.qasm",
        ))
        .expect("load qasm");
        let mut backend = aleph_sv::NaiveSvBackend::with_seed(0);
        aleph_oracle::run_state_oracle(&mut backend, &fx, &qasm).expect("oracle");
    }

    #[test]
    fn distribution() {
        let fx = aleph_oracle::load_fixture(&aleph_oracle::workspace_path(
            "oracle/fixtures/<name>.json",
        ))
        .expect("load fixture");
        let qasm = aleph_oracle::load_qasm(&aleph_oracle::workspace_path(
            "oracle/circuits/<name>.qasm",
        ))
        .expect("load qasm");
        let mut backend = aleph_sv::NaiveSvBackend::with_seed(0);
        aleph_oracle::run_distribution_oracle(&mut backend, &fx, &qasm).expect("oracle");
    }
}
```

Failure of either test surfaces as `naive_sv::<name>::state` or
`naive_sv::<name>::distribution` — distinct from the other so
triage is unambiguous.

### 6.6 Cost budget

Per-fixture distribution test: parse + run + sample(100k) +
tally + per-outcome band check. At `n = 10` this is roughly:
~50 µs parse + ~200 µs run + ~5 ms sample + ~10 µs tally + ~10 µs
check ≈ 5 ms wall time. Across 28 fixtures: ~140 ms. Negligible
vs the existing state-vector tests.

---

## 7. Benchmarks

Two new criterion benches under `crates/aleph-sv/benches/`. Both
follow the existing `naive_sv.rs` pattern (criterion default,
no custom harness, throughput in `BatchSize::SmallInput`).

### 7.1 `benches/sample.rs`

| Bench id | n | shots | State |
|---|---|---|---|
| `sample_uniform_n4_shots1k` | 4 | 1 000 | Uniform `|+⟩⊗⁴` |
| `sample_uniform_n10_shots100k` | 10 | 100 000 | Uniform `|+⟩⊗¹⁰` |
| `sample_uniform_n16_shots100k` | 16 | 100 000 | Uniform `|+⟩⊗¹⁶` |
| `sample_ghz_n10_shots100k` | 10 | 100 000 | GHZ (2 non-zero outcomes) |

Hand-constructed normalised state vectors (no `apply_gate`
through the kernel — too noisy at small n). Each bench measures
one `sample(state, shots)` call.

PR description includes before/after via:

```bash
cargo bench --bench sample -- --save-baseline pre-alias    # at HEAD~1
cargo bench --bench sample -- --baseline pre-alias         # after rewrite
```

### 7.2 `benches/expectation.rs`

| Bench id | n | Pauli |
|---|---|---|
| `exp_z_chain_n10` | 10 | `Z⊗Z⊗…⊗Z` (10 terms) |
| `exp_x_chain_n10` | 10 | `X⊗X⊗…⊗X` (10 terms) |
| `exp_mixed_zx_n10` | 10 | `Z⊗X⊗Z⊗X…` (10 terms) |

State: Hadamard wall (`|+⟩⊗¹⁰`) so neither path is trivially
short-circuited.

Acceptance target: `exp_z_chain_n10` ≥ 10× faster than the
pre-fast-path baseline (no clone, no kernel apply, one pass).

---

## 8. Statistical convergence test (BACKLOG AC #2)

New file `crates/aleph-sv/tests/sampling_convergence.rs`:

```rust
#[test]
fn bell_state_1m_shots_converges_to_uniform_on_phi_plus() {
    let mut b = NaiveSvBackend::with_seed(0);
    let mut s = b.allocate(2).unwrap();
    b.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0])).unwrap();
    b.apply_gate(&mut s, &GateInstance::new(Gate::Cnot, smallvec![0, 1])).unwrap();

    const N: u32 = 1_000_000;
    let shots = b.sample(&s, N).unwrap();
    let mut hist = [0u64; 4];
    for v in &shots {
        hist[*v as usize] += 1;
    }
    assert_eq!(hist[1], 0, "Bell |Φ+⟩ produced a |01⟩ sample");
    assert_eq!(hist[2], 0, "Bell |Φ+⟩ produced a |10⟩ sample");
    // 10σ band; σ = √(0.25 · 1e6) = 500, so band = 5000.
    let band = 5000.0;
    for &k in &[0usize, 3] {
        let dev = (hist[k] as f64 - 500_000.0).abs();
        assert!(
            dev <= band,
            "outcome {k}: count {} deviates by {dev} > {band}",
            hist[k]
        );
    }
}
```

Wall time: ~50 ms. Not marked `#[ignore]` — runs in default
`cargo test`.

---

## 9. ∑ probabilities = 1 property test (BACKLOG testing reqs)

New `proptest!` block in `crates/aleph-sv/src/measure.rs`:

```rust
proptest! {
    #[test]
    fn probabilities_full_basis_sums_to_one(
        n in 1u32..=6,
        ops in proptest::collection::vec(any::<RandomOp>(), 0..30),
    ) {
        let mut b = NaiveSvBackend::with_seed(0);
        let mut s = b.allocate(n).unwrap();
        for op in &ops {
            if let Some(gi) = op.realize(n) {
                b.apply_gate(&mut s, &gi).unwrap();
            }
        }
        let qubits: Vec<u32> = (0..n).collect();
        let p = b.probabilities(&s, &qubits).unwrap();
        let sum: f64 = p.iter().sum();
        let drift = (p.len() as f64).sqrt() * aleph_core::AMPLITUDE_TOL;
        prop_assert!((sum - 1.0).abs() <= drift, "sum = {sum}");
    }
}
```

`RandomOp` is a small in-module enum sampling from
`{H, X, Y, Z, S, T, Cnot}` plus a qubit choice. Default 256
proptest cases × up to 30 gates × n ≤ 6 ≈ 50k apply_gate calls
total — well under 1 s wall time.

---

## 10. Inline defensive checks (PR-69 carry-over)

Two of the PLAUSIBLE findings from the PR-69 code review are
folded in because the distribution oracle benefits from them
directly:

### 10.1 `load_fixture` shape check

In `crates/aleph-oracle/src/fixture.rs::load_fixture`, after the
endianness check:

```rust
let expected_dim = 1usize.checked_shl(fx.num_qubits).ok_or_else(|| {
    OracleError::DimensionMismatch {
        name: fx.name.clone(),
        fixture: 0,
        state: usize::MAX,
    }
})?;
if fx.statevector.amplitudes.len() != expected_dim {
    return Err(OracleError::DimensionMismatch {
        name: fx.name,
        fixture: fx.statevector.amplitudes.len(),
        state: expected_dim,
    });
}
```

A corrupt fixture is now rejected at load time with a precise
error, before any backend allocation.

### 10.2 `build.rs` qasm↔fixture symmetry

After enumerating fixtures, also enumerate `oracle/circuits/`:

```rust
let circuits_dir = workspace_root.join("oracle/circuits");
println!("cargo:rerun-if-changed={}", circuits_dir.display());

let qasm_stems: std::collections::BTreeSet<String> = std::fs::read_dir(&circuits_dir)
    .unwrap_or_else(|e| panic!("read_dir {}: {e}", circuits_dir.display()))
    .filter_map(Result::ok)
    .map(|e| e.path())
    .filter(|p| p.extension().is_some_and(|x| x == "qasm"))
    .map(|p| p.file_stem().unwrap().to_string_lossy().into_owned())
    .collect();

let fixture_stems: std::collections::BTreeSet<String> = entries
    .iter()
    .map(|p| p.file_stem().unwrap().to_string_lossy().into_owned())
    .collect();

for stem in qasm_stems.difference(&fixture_stems) {
    panic!(
        "oracle/circuits/{stem}.qasm has no matching oracle/fixtures/{stem}.json. \
         Run scripts/regen-fixtures.sh."
    );
}
```

Forgetting to regenerate now fails the build with an actionable
message instead of silently producing zero tests for the new
circuit.

The other PR-69 PLAUSIBLE findings (`workspace_path` resolver
brittleness, `fixture.name == stem` check, `measure_all` corruption
with pre-existing classical register, build.rs missing-fixtures-dir)
remain deferred — they're orthogonal to P0-11's purpose and adding
them here would expand scope.

---

## 11. Acceptance-criteria mapping

| BACKLOG AC | Where satisfied |
|---|---|
| All four primitives implemented for naive backend | P0-09 (functional) + P0-11 (alias sampler + Z fast path) |
| Sampling distribution converges to \|ψ\|² (statistical test with 1M shots) | §8 `bell_state_1m_shots_converges_to_uniform_on_phi_plus` |
| Expectation value tests vs. analytical results for known states | §5.4 + existing P0-09 tests |
| Test reqs: measure \|0⟩ → always 0, P(0) = 1 | Existing P0-09 `measure_zero_state_returns_false` |
| Test reqs: measure \|+⟩ → 0/1 equal probability | Existing P0-09 `measure_plus_state_collapses_to_basis` |
| Test reqs: ⟨0\|Z\|0⟩ = 1, ⟨+\|X\|+⟩ = 1 | Existing P0-09 cases |
| Test reqs: ∑ probabilities = 1 (property) | §9 |
| Test reqs: 1M shots Bell → 50/50 within tolerance | §8 |

---

## 12. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Alias build numerical drift on subnormal inputs. | `validate_state` (already in the call path) bounds total drift to `√n · AMPLITUDE_TOL`; alias `scaled[i] < 1` comparison is well-defined for subnormals. Unit test in §4.6 covers a `1 + 1e-15` total. |
| Per-fixture distribution test flake at 5σ. | Math: ≤ 5.7e-7 per outcome, ≤ 5.8e-4 per fixture. CI uses fixed seed = 0, so the result is deterministic per machine. If a `gen.py` Qiskit upgrade shifts an exact `\|a\|²` enough to push aleph's seeded sample out of band, the failure is informative (band reports `5σ + 1e-6`). |
| Z fast path inconsistent with slow path on a pathological state. | The property test in §5.4 pins them to within `1e-12`. If a future kernel change perturbs amplitudes, both paths see the same input and the property test catches divergence. |
| `bell_state_1m_shots_converges_to_uniform_on_phi_plus` is slow (≥ 50 ms). | Runs in default `cargo test`. If it grows past 500 ms over future refactors, mark `#[ignore]` and add a nightly-CI job — but not now. |
| Build script panics in `build.rs` if `oracle/circuits/` or `oracle/fixtures/` is missing (sparse checkouts, Docker layers). | Not addressed in P0-11; tracked as a separate PR-69 follow-up. |

---

## 13. Workflow notes

- Spec lives on the feature branch (`p0-11-primitives`), not on
  main. Squashed into the implementation PR following the
  P0-06…P0-10 workflow.
- Implementation order:
  1. Alias table + unit tests + integration into `sample_impl`.
  2. Sampling bench (file + numbers in PR description).
  3. Z fast path + unit tests + property test.
  4. Expectation bench.
  5. `load_fixture` shape check + tests.
  6. `run_distribution_oracle` + harness tests.
  7. `build.rs` two-tests-per-fixture + qasm↔fixture symmetry check.
  8. 1M-shot Bell convergence test.
  9. `∑ probabilities = 1` proptest.
  10. Documentation in `docs/testing.md` ("Distribution oracle" section).
  11. Final lint / fmt / workspace test sweep.

Each step ships in its own commit so review can read the diff
linearly.
