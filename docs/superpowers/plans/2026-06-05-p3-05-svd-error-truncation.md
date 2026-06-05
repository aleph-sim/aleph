# P3-05 MPS — SVD truncation with controlled error — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an error-bounded SVD-truncation mode to the MPS backend (`aleph-mps`) alongside the existing fixed-χ mode, via a `TruncationPolicy` enum, plumb it through `MpsState`/`MpsBackend`/CLI, and report accumulated truncation error + max bond reached.

**Architecture:** P3-04 already does fixed-χ truncation in `tensor::truncated_svd` (Hermitian Gram-matrix method) and accumulates discarded weight into `MpsState::trunc_error`. This adds a `TruncationPolicy { FixedBond(χ) | ErrorBounded { epsilon, max_bond } }`, branches the χ-selection inside `truncated_svd`, threads the policy through the state/backend/CLI, and tracks the running max bond dimension.

**Tech Stack:** Rust 2021, `nalgebra` (Gram eigendecomposition from P3-04), `clap`, `proptest`, `assert_cmd`, `criterion`.

**Spec:** `docs/superpowers/specs/2026-06-05-p3-05-svd-error-truncation-design.md`

**Branch:** `p3-05-svd-error-truncation` (already created off `main` = `42081af`).

**Conventions:** no `unwrap`/`expect` in library code (tests OK); `cargo clippy --workspace --all-targets -- -D warnings`; **`cargo fmt --all --check`** (workspace form — the per-crate form misses files and CI runs `--all`). End every commit body with: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

## Current code (ground truth, as merged in P3-04)

`crates/aleph-mps/src/tensor.rs` — `truncated_svd`:
```rust
pub fn truncated_svd(
    m: &DMatrix<Complex>,
    max_bond: usize,
) -> (DMatrix<Complex>, Vec<f64>, DMatrix<Complex>, f64) {
    let rows = m.nrows();
    let cols = m.ncols();
    let g = m.adjoint() * m;
    let eig = nalgebra::linalg::SymmetricEigen::new(g);
    let mut pairs: Vec<(f64, usize)> = (0..cols)
        .map(|k| (eig.eigenvalues[k].max(0.0).sqrt(), k))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let s_max = pairs.first().map(|p| p.0).unwrap_or(0.0);
    let eps = 1e-7 * s_max.max(f64::MIN_POSITIVE);
    let significant = pairs.iter().filter(|p| p.0 > eps).count().max(1);
    let chi = significant.min(max_bond.max(1));
    let discarded: f64 = pairs[chi..].iter().map(|p| p.0 * p.0).sum();
    let kept_weight: f64 = pairs[..chi].iter().map(|p| p.0 * p.0).sum();
    let scale = if kept_weight > 0.0 { (1.0 / kept_weight).sqrt() } else { 1.0 };
    let mut u_kept = DMatrix::<Complex>::zeros(rows, chi);
    let mut vt_kept = DMatrix::<Complex>::zeros(chi, cols);
    let mut s_kept = vec![0.0_f64; chi];
    for (new_k, &(sigma, eig_k)) in pairs[..chi].iter().enumerate() {
        let vk = eig.eigenvectors.column(eig_k);
        for c in 0..cols { vt_kept[(new_k, c)] = vk[c].conj(); }
        let mvk = m * vk;
        let inv = if sigma > eps { 1.0 / sigma } else { 0.0 };
        for r in 0..rows { u_kept[(r, new_k)] = mvk[r] * Complex::new(inv, 0.0); }
        s_kept[new_k] = sigma * scale;
    }
    (u_kept, s_kept, vt_kept, discarded)
}
```
`crates/aleph-mps/src/mps.rs` — `MpsState { sites: Vec<Site>, center: usize, max_bond: usize, trunc_error: f64 }`; `pub fn new(n, max_bond) { …, max_bond: max_bond.max(1), trunc_error: 0.0 }`; in `apply_2q`: `let (u_s, s_kept, vt_s, discarded) = truncated_svd(&m, self.max_bond); self.trunc_error += discarded; let chi = s_kept.len();`.
`crates/aleph-mps/src/backend.rs` — `MpsBackend { rng: StdRng, max_bond: usize }`; `DEFAULT_MAX_BOND = 128`; `with_max_bond`; `allocate` → `MpsState::new(num_qubits as usize, self.max_bond)`.
`crates/aleph-cli`: `cli.rs` `Run { …, backend, max_bond: usize }` (`--max-bond` default 128); `exec.rs` `run_circuit(…, max_bond, out)` dispatches `run_mps(…, max_bond, …)` which builds `MpsBackend::…with_max_bond(max_bond)`.

---

## Task 1: `TruncationPolicy` enum + error-bounded selection in `truncated_svd`

**Files:** Modify `crates/aleph-mps/src/tensor.rs`, `crates/aleph-mps/src/lib.rs`.

- [ ] **Step 1: Add the enum + failing unit tests** to `tensor.rs`.

Add above `truncated_svd`:
```rust
/// How `truncated_svd` chooses how many singular values to keep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TruncationPolicy {
    /// Keep at most `χ` singular values (the largest).
    FixedBond(usize),
    /// Keep the fewest singular values whose discarded squared weight is `≤ ε`,
    /// never exceeding `max_bond`.
    ErrorBounded { epsilon: f64, max_bond: usize },
}
```

Add tests to `tensor.rs`'s `mod tests` (it already imports `super::*`, `aleph_core::Complex`, and uses `DMatrix` via `super::*`):
```rust
    // A 4×4 matrix whose singular values are exactly [1, 0.1, 0.01, 0.001]
    // (diagonal embeds them directly).
    fn diag_sigma() -> DMatrix<Complex> {
        let s = [1.0, 0.1, 0.01, 0.001];
        DMatrix::from_fn(4, 4, |i, j| {
            if i == j { Complex::new(s[i], 0.0) } else { Complex::new(0.0, 0.0) }
        })
    }

    #[test]
    fn error_bounded_keeps_minimal_chi() {
        let m = diag_sigma();
        // ε between 0.01²+0.001²=1.01e-4 and 0.1²+...=1.0101e-2 → drop the two
        // smallest, keep χ=2.
        let (_, s, _, disc) = truncated_svd(&m, &TruncationPolicy::ErrorBounded { epsilon: 1e-3, max_bond: 64 });
        assert_eq!(s.len(), 2, "expected χ=2");
        assert!(disc <= 1e-3 + 1e-15, "discarded {disc} exceeds ε");
    }

    #[test]
    fn error_bounded_tiny_eps_keeps_all() {
        let m = diag_sigma();
        let (_, s, _, disc) = truncated_svd(&m, &TruncationPolicy::ErrorBounded { epsilon: 0.0, max_bond: 64 });
        assert_eq!(s.len(), 4, "ε=0 must keep full rank");
        assert!(disc < 1e-12);
    }

    #[test]
    fn error_bounded_cap_overrides_eps() {
        let m = diag_sigma();
        // ε large enough to drop everything, but cap forces χ=1.
        let (_, s, _, _) = truncated_svd(&m, &TruncationPolicy::ErrorBounded { epsilon: 10.0, max_bond: 1 });
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn fixed_bond_matches_legacy() {
        let m = diag_sigma();
        let (_, s, _, _) = truncated_svd(&m, &TruncationPolicy::FixedBond(2));
        assert_eq!(s.len(), 2);
    }
```

- [ ] **Step 2: Run** `cargo test -p aleph-mps tensor` → FAIL (signature mismatch / `TruncationPolicy` unknown).

- [ ] **Step 3: Change `truncated_svd` signature + branch χ selection.** Replace the `max_bond: usize` parameter with `policy: &TruncationPolicy`, and replace the two lines
```rust
    let significant = pairs.iter().filter(|p| p.0 > eps).count().max(1);
    let chi = significant.min(max_bond.max(1));
```
with:
```rust
    let significant = pairs.iter().filter(|p| p.0 > eps).count().max(1);
    // Suffix sums of σ²: suffix_sq[k] = Σ_{j≥k} σ_j² (non-increasing in k).
    let mut suffix_sq = vec![0.0_f64; pairs.len() + 1];
    for k in (0..pairs.len()).rev() {
        suffix_sq[k] = suffix_sq[k + 1] + pairs[k].0 * pairs[k].0;
    }
    let chi = match *policy {
        TruncationPolicy::FixedBond(max_bond) => significant.min(max_bond.max(1)),
        TruncationPolicy::ErrorBounded { epsilon, max_bond } => {
            let cap = significant.min(max_bond.max(1));
            // Smallest keep ∈ [1, cap] whose discarded tail Σ_{j≥keep} σ_j² ≤ ε.
            // suffix_sq is non-increasing, so the first satisfying keep is minimal.
            let mut chosen = cap;
            for keep in 1..=cap {
                if suffix_sq[keep] <= epsilon {
                    chosen = keep;
                    break;
                }
            }
            chosen
        }
    };
```
The line `let discarded: f64 = pairs[chi..].iter().map(|p| p.0 * p.0).sum();` stays (equals `suffix_sq[chi]`; leaving it as-is is fine, or replace with `let discarded = suffix_sq[chi];` — either works, prefer `suffix_sq[chi]` to avoid recomputation). The rest of the function is unchanged.

- [ ] **Step 4: Update the doc comment** on `truncated_svd`: change the first line from "truncated to at most `max_bond` singular values" to "truncated according to `policy` (fixed-χ or error-bounded)". Keep the "# Why not nalgebra's SVD" section.

- [ ] **Step 5: Re-export from lib.rs.** In `crates/aleph-mps/src/lib.rs`, add `pub use tensor::TruncationPolicy;` next to the existing `pub use` lines (the `tensor` module is currently private `mod tensor;` — change to keep it private but re-export the one public type: add `pub use tensor::TruncationPolicy;`). Note `truncated_svd` itself stays crate-internal (it's `pub` within the crate but `tensor` is a private module, so it isn't part of the public API — only `TruncationPolicy` is re-exported).

- [ ] **Step 6: Run** `cargo test -p aleph-mps tensor` → PASS. Then `cargo clippy -p aleph-mps --all-targets -- -D warnings` and `cargo fmt --all -- --check`. (This task changes `truncated_svd`'s only caller, `apply_2q` — it will NOT compile until Task 2 updates the call. To keep this task self-contained, temporarily update the single call site in `mps.rs` `apply_2q` from `truncated_svd(&m, self.max_bond)` to `truncated_svd(&m, &crate::tensor::TruncationPolicy::FixedBond(self.max_bond))` so the crate compiles and tests run. Task 2 then replaces `self.max_bond` with `self.policy` properly.)

- [ ] **Step 7: Commit.**
```bash
git add crates/aleph-mps/src/tensor.rs crates/aleph-mps/src/lib.rs crates/aleph-mps/src/mps.rs
git commit
```
subject `[P3-05] TruncationPolicy enum + error-bounded selection in truncated_svd` + trailer.

---

## Task 2: Thread `TruncationPolicy` through `MpsState` + track max bond reached

**Files:** Modify `crates/aleph-mps/src/mps.rs`.

- [ ] **Step 1: Add failing tests** to `mps.rs`'s `mod tests`:
```rust
    #[test]
    fn max_bond_reached_tracks_growth() {
        // GHZ-4 grows the central bond to 2; max_bond_reached ≥ 2.
        let mut s = MpsState::new(4, 64);
        let h = crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        s.apply_1q(0, &h);
        for i in 0..3u32 {
            let g = GateInstance::new(Gate::Cnot, smallvec![i, i + 1]);
            let cnot = crate::gate::matrix_4x4(&g).unwrap();
            s.apply_2q(&g, &cnot).unwrap();
        }
        assert!(s.max_bond_reached() >= 2, "got {}", s.max_bond_reached());
    }

    #[test]
    fn error_bounded_policy_truncates_state() {
        use crate::tensor::TruncationPolicy;
        // Same GHZ-4 but with a loose ε: bond should be capped small, with
        // nonzero accumulated truncation error.
        let mut s = MpsState::with_policy(4, TruncationPolicy::ErrorBounded { epsilon: 0.3, max_bond: 64 });
        let h = crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        s.apply_1q(0, &h);
        for i in 0..3u32 {
            let g = GateInstance::new(Gate::Cnot, smallvec![i, i + 1]);
            let cnot = crate::gate::matrix_4x4(&g).unwrap();
            s.apply_2q(&g, &cnot).unwrap();
        }
        // GHZ Schmidt values are [1/√2, 1/√2]; dropping one discards 0.5 > 0.3,
        // so it keeps both (χ=2) → error stays 0 here. Assert it ran and the
        // bound was respected (no truncation exceeding ε).
        assert!(s.truncation_error() <= 0.3 + 1e-12);
    }
```

- [ ] **Step 2: Run** `cargo test -p aleph-mps max_bond_reached` → FAIL (`with_policy`/`max_bond_reached` undefined).

- [ ] **Step 3: Change the struct + constructors.** In `mps.rs`, add the import `use crate::tensor::{truncated_svd, thin_qr, Site, TruncationPolicy};` (extend the existing `use crate::tensor::{...}` line to include `TruncationPolicy`). Replace the struct:
```rust
#[derive(Debug, Clone)]
pub struct MpsState {
    pub(crate) sites: Vec<Site>,
    pub(crate) center: usize,
    pub(crate) policy: TruncationPolicy,
    pub(crate) trunc_error: f64,
    pub(crate) max_bond_seen: usize,
}
```
Replace `new` and add `with_policy` + `max_bond_reached`:
```rust
    /// Allocate |0…0⟩ on `n` qubits with a fixed bond cap `max_bond`.
    pub fn new(n: usize, max_bond: usize) -> Self {
        Self::with_policy(n, TruncationPolicy::FixedBond(max_bond.max(1)))
    }

    /// Allocate |0…0⟩ on `n` qubits with an explicit truncation policy.
    pub fn with_policy(n: usize, policy: TruncationPolicy) -> Self {
        let sites = (0..n).map(|_| Site::ket0()).collect();
        MpsState { sites, center: 0, policy, trunc_error: 0.0, max_bond_seen: 1 }
    }

    /// The largest bond dimension reached by any 2q truncation so far.
    pub fn max_bond_reached(&self) -> usize {
        self.max_bond_seen
    }
```
(Keep `num_qubits`, `truncation_error`, `dense_statevector` unchanged.)

- [ ] **Step 4: Update `apply_2q`.** Replace
```rust
        let (u_s, s_kept, vt_s, discarded) = truncated_svd(&m, self.max_bond);
        self.trunc_error += discarded;
        let chi = s_kept.len();
```
with
```rust
        let (u_s, s_kept, vt_s, discarded) = truncated_svd(&m, &self.policy);
        self.trunc_error += discarded;
        let chi = s_kept.len();
        self.max_bond_seen = self.max_bond_seen.max(chi);
```
(Also revert the temporary call-site change from Task 1 Step 6 — this is the proper version.)

- [ ] **Step 5: Run** `cargo test -p aleph-mps` → PASS. clippy `-D warnings` + `cargo fmt --all -- --check` clean.

- [ ] **Step 6: Commit.**
```bash
git add crates/aleph-mps/src/mps.rs
git commit
```
subject `[P3-05] MpsState: TruncationPolicy field + max_bond_reached tracking` + trailer.

---

## Task 3: `MpsBackend` policy field + `with_truncation`

**Files:** Modify `crates/aleph-mps/src/backend.rs`.

- [ ] **Step 1: Add failing tests** to `backend.rs`'s `mod tests`:
```rust
    #[test]
    fn with_truncation_error_bounded_runs() {
        use crate::TruncationPolicy;
        let mut be = MpsBackend::with_seed(0)
            .with_truncation(TruncationPolicy::ErrorBounded { epsilon: 1e-8, max_bond: 32 });
        let mut s = be.allocate(3).unwrap();
        be.apply_gate(&mut s, &GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        be.apply_gate(&mut s, &GateInstance::new(Gate::Cnot, smallvec![0u32, 1u32])).unwrap();
        be.apply_gate(&mut s, &GateInstance::new(Gate::Cnot, smallvec![1u32, 2u32])).unwrap();
        // Sampling still works under the policy.
        for sh in be.sample(&s, 100).unwrap() { assert!(sh == 0b000 || sh == 0b111); }
    }

    #[test]
    fn with_max_bond_is_fixed_bond_sugar() {
        let _ = MpsBackend::with_seed(0).with_max_bond(16); // compiles + runs
    }
```
(Import note: `backend.rs` is inside the crate, so the test uses `crate::TruncationPolicy`, which resolves to the `pub use tensor::TruncationPolicy` re-export from `lib.rs`.)

- [ ] **Step 2: Run** `cargo test -p aleph-mps --lib backend` → FAIL.

- [ ] **Step 3: Change the backend.** In `backend.rs` add `use crate::TruncationPolicy;`. Replace the struct + constructors:
```rust
pub struct MpsBackend {
    rng: StdRng,
    policy: TruncationPolicy,
}

const MAX_QUBITS: u32 = 1024;
const DEFAULT_MAX_BOND: usize = 128;

impl MpsBackend {
    pub fn new() -> Self {
        Self { rng: StdRng::from_entropy(), policy: TruncationPolicy::FixedBond(DEFAULT_MAX_BOND) }
    }
    pub fn with_seed(seed: u64) -> Self {
        Self { rng: StdRng::seed_from_u64(seed), policy: TruncationPolicy::FixedBond(DEFAULT_MAX_BOND) }
    }
    /// Fixed bond-dimension truncation (sugar for `with_truncation(FixedBond(χ))`).
    pub fn with_max_bond(mut self, chi: usize) -> Self {
        self.policy = TruncationPolicy::FixedBond(chi.max(1));
        self
    }
    /// Set an explicit truncation policy (fixed-χ or error-bounded).
    pub fn with_truncation(mut self, policy: TruncationPolicy) -> Self {
        self.policy = policy;
        self
    }
}
```
Replace `allocate`'s body line `Ok(MpsState::new(num_qubits as usize, self.max_bond))` with `Ok(MpsState::with_policy(num_qubits as usize, self.policy))`. (Keep the `MAX_QUBITS` guard above it unchanged.)

- [ ] **Step 4: Run** `cargo test -p aleph-mps` → PASS. clippy `-D warnings` + `cargo fmt --all -- --check`.

- [ ] **Step 5: Commit.**
```bash
git add crates/aleph-mps/src/backend.rs
git commit
```
subject `[P3-05] MpsBackend: TruncationPolicy field + with_truncation` + trailer.

---

## Task 4: CLI `--max-error` + truncation reporting

**Files:** Modify `crates/aleph-cli/src/cli.rs`, `crates/aleph-cli/src/exec.rs`, `crates/aleph-cli/src/main.rs`, `crates/aleph-cli/tests/cli.rs`.

- [ ] **Step 1: Add `--max-error` to the `Run` subcommand** (`cli.rs`), after the `max_bond` field:
```rust
        /// MPS error-bounded truncation: keep the discarded weight per bond
        /// below ε (only used by `--backend mps`; overrides fixed-χ, with
        /// `--max-bond` as a safety cap).
        #[arg(long)]
        max_error: Option<f64>,
```

- [ ] **Step 2: Thread `max_error` to `run_circuit`.** In `main.rs`, add `max_error` to the `Cmd::Run { … }` destructure and pass it to `run_circuit(…, max_bond, max_error, &mut out)`. In `exec.rs`, add `max_error: Option<f64>` to `run_circuit`'s signature (after `max_bond`). Update ALL `run_circuit(…)` call sites in `exec.rs`'s test module to pass `None` after the `128` max_bond argument.

- [ ] **Step 3: Validate + resolve the policy + report, in `run_mps`.** In `exec.rs`:
  1. At the top of `run_circuit`, after the existing `--expectation` validation, validate `max_error`:
  ```rust
      if let Some(e) = max_error {
          if !e.is_finite() || e <= 0.0 {
              return Err(anyhow!("--max-error must be a positive finite number, got {e}"));
          }
      }
  ```
  2. Change the `if backend == BackendKind::Mps { return run_mps(…, max_bond, &seed_label, out); }` call to also pass `max_error`.
  3. Change `run_mps`'s signature to take `max_error: Option<f64>` (after `max_bond`), and replace the backend construction + add reporting:
  ```rust
      use aleph_mps::TruncationPolicy;
      let policy = match max_error {
          Some(epsilon) => TruncationPolicy::ErrorBounded { epsilon, max_bond },
          None => TruncationPolicy::FixedBond(max_bond),
      };
      let mut backend = match seed {
          Some(s) => MpsBackend::with_seed(s),
          None => MpsBackend::new(),
      }
      .with_truncation(policy);
      let state = run(&mut backend, circuit).context("running circuit (mps)")?;
  ```
  Then AFTER the sampling/expectation views, before `Ok(())`, add:
  ```rust
      writeln!(
          out,
          "truncation error: {:.3e}; max bond χ: {}",
          state.truncation_error(),
          state.max_bond_reached()
      )?;
  ```
  (`MpsState::truncation_error()` and `max_bond_reached()` are public.)

- [ ] **Step 4: Integration tests** in `crates/aleph-cli/tests/cli.rs` (follow the existing `aleph()` + `bell_path()`/`ghz3_path()` helpers; mirror the existing `mps_backend_runs_bell` test):
```rust
#[test]
fn mps_backend_reports_truncation() {
    aleph()
        .args(["run"])
        .arg(ghz3_path())
        .args(["--backend", "mps", "--shots", "64", "--seed", "0"])
        .assert()
        .success()
        .stdout(contains("truncation error:"))
        .stdout(contains("max bond"));
}

#[test]
fn mps_backend_max_error_runs() {
    aleph()
        .args(["run"])
        .arg(ghz3_path())
        .args(["--backend", "mps", "--max-error", "1e-8", "--shots", "64", "--seed", "0"])
        .assert()
        .success()
        .stdout(contains("truncation error:"));
}

#[test]
fn mps_backend_rejects_nonpositive_max_error() {
    aleph()
        .args(["run"])
        .arg(ghz3_path())
        .args(["--backend", "mps", "--max-error", "0"])
        .assert()
        .failure()
        .stderr(contains("--max-error must be a positive"));
}
```

- [ ] **Step 5: Run** `cargo test -p aleph-cli` → PASS. clippy `-D warnings` + `cargo fmt --all -- --check`.

- [ ] **Step 6: Commit.**
```bash
git add crates/aleph-cli
git commit
```
subject `[P3-05] CLI --max-error + MPS truncation reporting` + trailer.

---

## Task 5: Property + oracle tests for both modes

**Files:** Modify `crates/aleph-mps/tests/sv_equivalence.rs`.

The file already has helpers `g(gate, &[u32])`, `mps_dense(circuit, chi)`, `sv_dense(circuit)`, and imports `aleph_backend::{run, Backend}`, `aleph_core::{Complex, Gate, GateInstance, Param, Pauli, PauliString}`, `aleph_mps::{MpsBackend, MpsState}`, `aleph_sv::NaiveSvBackend`, `smallvec::smallvec`. Add `use aleph_mps::TruncationPolicy;` at the top.

- [ ] **Step 1: Add an error-bounded dense helper + exactness/bound tests:**
```rust
fn mps_dense_policy(circuit: &aleph_ir::Circuit, policy: TruncationPolicy) -> (Vec<Complex>, f64) {
    let mut be = MpsBackend::with_seed(0).with_truncation(policy);
    let st: MpsState = run(&mut be, circuit).unwrap();
    (st.dense_statevector(), st.truncation_error())
}

#[test]
fn error_bounded_eps0_is_exact() {
    // ε=0 must reproduce the state vector exactly (no truncation).
    let n = 5u32;
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n { c.add_gate(g(Gate::H, &[q])).unwrap(); }
    for q in 0..n - 1 { c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap(); }
    for q in 0..n { c.add_gate(g(Gate::Rz(Param::Concrete(0.2 + q as f64 * 0.1)), &[q])).unwrap(); }
    let (a, err) = mps_dense_policy(&c, TruncationPolicy::ErrorBounded { epsilon: 0.0, max_bond: 64 });
    let b = sv_dense(&c);
    for (x, y) in a.iter().zip(b.iter()) { assert!((x - y).norm() < 1e-10); }
    assert!(err < 1e-12, "ε=0 should discard nothing, got {err}");
}

#[test]
fn error_bounded_deviation_within_budget() {
    // A moderately-entangling NN circuit with a moderate ε: the L2 deviation
    // from the exact state vector must not exceed √(accumulated discarded
    // weight) by more than a small constant factor.
    let n = 6u32;
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n { c.add_gate(g(Gate::H, &[q])).unwrap(); }
    for layer in 0..3 {
        for q in 0..n - 1 {
            c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap();
            c.add_gate(g(Gate::Rz(Param::Concrete(0.3 + layer as f64 * 0.1)), &[q + 1])).unwrap();
        }
    }
    let (a, err) = mps_dense_policy(&c, TruncationPolicy::ErrorBounded { epsilon: 1e-4, max_bond: 64 });
    let b = sv_dense(&c);
    let l2: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).norm_sqr()).sum::<f64>().sqrt();
    // First-order: ‖Δψ‖ ≲ √(Σ discarded). Allow a 4× slack for accumulation.
    assert!(l2 <= 4.0 * err.sqrt() + 1e-9, "L2 {l2} vs √err {}", err.sqrt());
}
```

- [ ] **Step 2: Add a proptest that the per-bond bound is honored** at the `truncated_svd` level is already covered by Task 1 unit tests; here add an end-to-end proptest that error-bounded never *increases* deviation beyond fixed-χ-exact on random NN circuits with ε=0:
```rust
use proptest::prelude::*;
proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]
    #[test]
    fn error_bounded_eps0_matches_sv_random(seq in prop::collection::vec(0u8..6, 0..24)) {
        let n = 4u32;
        let mut c = aleph_ir::Circuit::new(n, 0);
        let mut q = 0u32;
        for op in seq {
            q = (q + 1) % n;
            match op {
                0 => { c.add_gate(g(Gate::H, &[q])).unwrap(); }
                1 => { c.add_gate(g(Gate::X, &[q])).unwrap(); }
                2 => { c.add_gate(g(Gate::S, &[q])).unwrap(); }
                3 => { c.add_gate(g(Gate::Y, &[q])).unwrap(); }
                _ => { let lo = q.min(n - 2); c.add_gate(g(Gate::Cnot, &[lo, lo + 1])).unwrap(); }
            }
        }
        if c.is_empty() { return Ok(()); }
        let (a, _) = mps_dense_policy(&c, TruncationPolicy::ErrorBounded { epsilon: 0.0, max_bond: 64 });
        let b = sv_dense(&c);
        for (x, y) in a.iter().zip(b.iter()) { prop_assert!((x - y).norm() < 1e-9); }
    }
}
```
(The file already has a `use proptest::prelude::*;` and a `proptest! { … }` block from P3-04 — add this test INSIDE the existing block rather than re-importing/re-opening if that's cleaner; either compiles.)

- [ ] **Step 3: Run** `cargo test -p aleph-mps --test sv_equivalence` → PASS. clippy `-D warnings` + `cargo fmt --all -- --check`.

- [ ] **Step 4: Commit.**
```bash
git add crates/aleph-mps/tests/sv_equivalence.rs
git commit
```
subject `[P3-05] error-bounded truncation property + oracle tests` + trailer.

---

## Task 6: Benchmark (fixed-χ vs error-bounded) + docs

**Files:** Modify `crates/aleph-mps/benches/nn_qaoa.rs`, `crates/aleph-mps/src/lib.rs`.

- [ ] **Step 1: Extend the bench.** `benches/nn_qaoa.rs` currently benches `MpsBackend::with_seed(0).with_max_bond(64)` over n∈{10,20,30}. Add a second benchmark group comparing fixed-χ vs error-bounded. Add `use aleph_mps::TruncationPolicy;` and a second function:
```rust
fn bench_policies(cr: &mut Criterion) {
    let mut grp = cr.benchmark_group("nn_qaoa_n20_policy");
    let c = qaoa_circuit(20);
    grp.bench_function("fixed_chi64", |b| {
        b.iter(|| {
            let mut be = MpsBackend::with_seed(0).with_max_bond(64);
            run(&mut be, &c).unwrap()
        })
    });
    grp.bench_function("error_1e-8", |b| {
        b.iter(|| {
            let mut be = MpsBackend::with_seed(0)
                .with_truncation(TruncationPolicy::ErrorBounded { epsilon: 1e-8, max_bond: 64 });
            run(&mut be, &c).unwrap()
        })
    });
    grp.finish();
}
```
and add `bench_policies` to the `criterion_group!`: change `criterion_group!(benches, bench);` to `criterion_group!(benches, bench, bench_policies);`.

- [ ] **Step 2: Verify it compiles:** `cargo build -p aleph-mps --benches`. (No need to run the full bench.)

- [ ] **Step 3: Docs.** In `crates/aleph-mps/src/lib.rs`, extend the crate doc: add one sentence noting the two truncation modes and the `1e-14` Gram-floor caveat, e.g. under the existing module doc:
```rust
//! Truncation is configurable via [`TruncationPolicy`]: a fixed bond dimension
//! or an error-bounded mode that keeps the discarded weight per bond below `ε`.
//! Because truncation goes through a Hermitian Gram-matrix eigendecomposition,
//! the smallest reliably-controllable discarded weight is ~1e-14 (a finer
//! threshold would need a higher-precision SVD).
```

- [ ] **Step 4: Full gate.**
```bash
cargo test -p aleph-mps
cargo test -p aleph-cli
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check
cargo build -p aleph-mps --benches
```
All pass/clean.

- [ ] **Step 5: Commit.**
```bash
git add crates/aleph-mps
git commit
```
subject `[P3-05] policy benchmark + docs` + trailer.

---

## Final verification (before PR)
- [ ] `cargo test --workspace` green.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` clean.
- [ ] `cargo fmt --all --check` clean (workspace form — NOT per-crate).
- [ ] Self-review the diff.

## PR
- Title: `[P3-05] MPS — SVD truncation with controlled error`.
- Body: `Closes #36` (verify via `gh issue list`; spec says #36). Summary, test results (both modes; ε=0 exact oracle; bound-honored), bench note (no ratio AC), Gram-floor caveat.

## Notes for the implementer
1. `TruncationPolicy` lives in `tensor.rs`, re-exported as `aleph_mps::TruncationPolicy`. Inside the crate use `crate::TruncationPolicy` or `crate::tensor::TruncationPolicy`; from the integration test / bench / CLI use `aleph_mps::TruncationPolicy`.
2. `MpsState::new(n, max_bond)` stays valid (sugar for `FixedBond`) — existing tests keep working unchanged.
3. Use `cargo fmt --all` (not `-p`) — CI runs `cargo fmt --all --check` and the per-crate form misses files (lesson from P3-04).
4. Verify gate/Circuit API names against the live code if anything fails to compile: `aleph_ir::Circuit::new(n, 0)`, `.add_gate(...)→Result`, `Gate::Rz(Param::Concrete(θ))`, `NaiveSvBackend` State `.amplitudes()`.
