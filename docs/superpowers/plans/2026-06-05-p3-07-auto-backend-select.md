# P3-07 Automatic Backend Selection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a conservative circuit-analysis heuristic that picks the best backend automatically, exposed as the CLI default `--backend auto`.

**Architecture:** A new pure, read-only `select` module in `aleph-backend` computes `CircuitFeatures` from a `Circuit` and maps them to an abstract `BackendKind { Statevector, Stabilizer, Mps }` via an ordered rule. The CLI gains a `BackendChoice` clap enum (default `Auto`) that resolves to the concrete dispatch, honoring the requested output view and warning on too-large circuits.

**Tech Stack:** Rust 2021, `aleph-backend` (consumes `aleph-ir::Circuit`, `aleph-core::Gate::is_clifford`), `aleph-cli` (clap derive, `assert_cmd` integration tests).

**Spec:** `docs/superpowers/specs/2026-06-05-p3-07-auto-backend-select-design.md`

**Decision rule (`select_from`, ordered):**
1. `all_clifford` → Stabilizer
2. `num_qubits <= SV_EXACT_CAP` (28) → Statevector
3. `all_twoq_nearest_neighbor && twoq_depth <= MPS_DEPTH_THRESHOLD` (64) → Mps
4. else → Statevector

---

## Task 1: `select` module skeleton — `BackendKind` + constants

**Files:**
- Create: `crates/aleph-backend/src/select.rs`
- Modify: `crates/aleph-backend/src/lib.rs` (add `pub mod select;` + re-exports)

- [ ] **Step 1: Write the failing test**

Add to the bottom of the new `crates/aleph-backend/src/select.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_kind_display_labels() {
        assert_eq!(BackendKind::Statevector.to_string(), "state vector");
        assert_eq!(BackendKind::Stabilizer.to_string(), "stabilizer");
        assert_eq!(BackendKind::Mps.to_string(), "MPS");
    }

    #[test]
    fn caps_have_expected_values() {
        assert_eq!(SV_EXACT_CAP, 28);
        assert_eq!(MPS_DEPTH_THRESHOLD, 64);
    }
}
```

- [ ] **Step 2: Write the module head + types**

At the top of `crates/aleph-backend/src/select.rs`:

```rust
//! Automatic backend selection (P3-07).
//!
//! A pure, read-only heuristic: scan a [`Circuit`] for structural features and
//! map them to an abstract [`BackendKind`]. This module names backend kinds but
//! does **not** depend on the concrete `aleph-sv` / `aleph-stab` / `aleph-mps`
//! crates (they depend on `aleph-backend`, not the reverse), so the IR stays
//! backend-agnostic while the selection label lives with the `Backend` trait.
//!
//! See `docs/superpowers/specs/2026-06-05-p3-07-auto-backend-select-design.md`.

use aleph_ir::{Circuit, Instruction};

/// State-vector exact-and-fits soft cap (matches `aleph-sv` / `aleph-cli`).
/// At or below this qubit count an exact dense run is preferred over any
/// approximate backend.
pub const SV_EXACT_CAP: u32 = 28;

/// Soft guard against pathological entanglement growth in a nearest-neighbor
/// circuit. The MPS backend bounds memory via χ regardless, so this is a
/// conservative routing threshold (in two-qubit-gate layers), not a hard bound.
pub const MPS_DEPTH_THRESHOLD: usize = 64;

/// Resolved, abstract backend label produced by the heuristic.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// Dense state vector — exact, memory grows as 2^n.
    Statevector,
    /// Stabilizer tableau — Clifford-only, O(n²) memory.
    Stabilizer,
    /// MPS tensor network — bounded-entanglement, approximate beyond χ.
    Mps,
}

impl std::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            BackendKind::Statevector => "state vector",
            BackendKind::Stabilizer => "stabilizer",
            BackendKind::Mps => "MPS",
        })
    }
}
```

- [ ] **Step 3: Wire the module into `lib.rs`**

In `crates/aleph-backend/src/lib.rs`, after the existing `pub use` / module lines near the top (below the crate doc comment), add:

```rust
pub mod select;
pub use select::{
    analyze, select_backend, select_explained, BackendKind, CircuitFeatures, Selection,
    MPS_DEPTH_THRESHOLD, SV_EXACT_CAP,
};
```

> Note: `analyze`, `select_backend`, `select_explained`, `CircuitFeatures`, and
> `Selection` are added in Tasks 2–3; this re-export line compiles only after
> Task 3. If the workspace must build green between tasks, add the `pub use`
> names incrementally. For subagent-driven execution, add the full line now and
> let Task 2/3 fill them in — the crate will not build until Task 3, which is
> acceptable mid-feature.

- [ ] **Step 4: Run the Task-1 tests**

Run: `cargo test -p aleph-backend select::tests::backend_kind_display_labels select::tests::caps_have_expected_values`
Expected: PASS (the `pub use` of not-yet-defined items will fail to compile until Task 3; if running Task 1 in isolation, temporarily comment the `pub use select::{...}` line, keep `pub mod select;`, run, then restore).

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-backend/src/select.rs crates/aleph-backend/src/lib.rs
git commit -m "[P3-07] select module skeleton: BackendKind + caps"
```

---

## Task 2: `CircuitFeatures` + `analyze`

**Files:**
- Modify: `crates/aleph-backend/src/select.rs`

- [ ] **Step 1: Write the failing tests**

Add inside `mod tests` in `select.rs`:

```rust
use aleph_core::{Gate, GateInstance, Param};
use aleph_ir::Circuit;

// Bell pair: H(0); CNOT(0,1) — all Clifford, one nearest-neighbor 2q gate.
fn bell() -> Circuit {
    let mut c = Circuit::new(2, 0);
    c.add_gate(GateInstance::new(Gate::H, vec![0u32])).unwrap();
    c.add_gate(GateInstance::new(Gate::Cnot, vec![0u32, 1u32]))
        .unwrap();
    c
}

#[test]
fn analyze_bell_is_clifford_nn() {
    let f = analyze(&bell());
    assert_eq!(f.num_qubits, 2);
    assert!(f.all_clifford);
    assert!(f.all_twoq_nearest_neighbor);
    assert_eq!(f.twoq_depth, 1);
}

#[test]
fn analyze_t_gate_breaks_clifford() {
    let mut c = Circuit::new(1, 0);
    c.add_gate(GateInstance::new(Gate::T, vec![0u32])).unwrap();
    let f = analyze(&c);
    assert!(!f.all_clifford);
    assert!(f.all_twoq_nearest_neighbor); // vacuously: no 2q gates
    assert_eq!(f.twoq_depth, 0);
}

#[test]
fn analyze_long_range_breaks_nn() {
    let mut c = Circuit::new(4, 0);
    c.add_gate(GateInstance::new(Gate::Cnot, vec![0u32, 3u32]))
        .unwrap();
    let f = analyze(&c);
    assert!(!f.all_twoq_nearest_neighbor);
    assert_eq!(f.twoq_depth, 1);
}

#[test]
fn analyze_counts_only_twoq_layers_in_twoq_depth() {
    // Rz(0); Rz(1) parallel 1q layer, then CNOT(0,1) — depth 2, twoq_depth 1.
    let mut c = Circuit::new(2, 0);
    c.add_gate(GateInstance::new(Gate::Rz(Param::Concrete(0.3)), vec![0u32]))
        .unwrap();
    c.add_gate(GateInstance::new(Gate::Rz(Param::Concrete(0.3)), vec![1u32]))
        .unwrap();
    c.add_gate(GateInstance::new(Gate::Cnot, vec![0u32, 1u32]))
        .unwrap();
    let f = analyze(&c);
    assert_eq!(f.depth, 2);
    assert_eq!(f.twoq_depth, 1);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p aleph-backend select::tests::analyze`
Expected: FAIL — `analyze` / `CircuitFeatures` not defined.

- [ ] **Step 3: Implement `CircuitFeatures` + `analyze`**

Add to `select.rs` (after the `BackendKind` Display impl):

```rust
/// Read-only structural features of a circuit, computed in a single scan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CircuitFeatures {
    /// Number of qubits the circuit declares.
    pub num_qubits: u32,
    /// Total layer count (`circuit.layers().len()`); diagnostics only.
    pub depth: usize,
    /// Number of layers containing at least one two-qubit gate.
    pub twoq_depth: usize,
    /// Every `Gate` instruction is Clifford (`Measure`/`Barrier` allowed).
    pub all_clifford: bool,
    /// Every two-qubit gate acts on adjacent qubits (`|q0 - q1| == 1`).
    pub all_twoq_nearest_neighbor: bool,
}

/// Scan `c` once and extract the [`CircuitFeatures`] the heuristic needs.
///
/// Pure and total: read-only, never panics. Intended to run on a freshly
/// parsed circuit (before optimization passes), so the SV-only
/// `DiagonalPhase` / `TiledBlock` instructions are not expected; if present
/// they conservatively clear `all_clifford` (they are not Clifford-expressible).
pub fn analyze(c: &Circuit) -> CircuitFeatures {
    let insts = c.instructions();

    let mut all_clifford = true;
    let mut all_twoq_nearest_neighbor = true;
    for inst in insts {
        match inst {
            Instruction::Gate(g) => {
                if !g.gate.is_clifford() {
                    all_clifford = false;
                }
                if g.qubits.len() == 2 && g.qubits[0].abs_diff(g.qubits[1]) != 1 {
                    all_twoq_nearest_neighbor = false;
                }
            }
            // Stabilizer supports measurement; barriers are no-ops. Reset is
            // unsupported on every backend (see spec), so it does not affect
            // the viable choice and is intentionally ignored here.
            Instruction::Measure { .. } | Instruction::Barrier(_) | Instruction::Reset(_) => {}
            // SV-only optimization artifacts: not Clifford-expressible.
            Instruction::DiagonalPhase(_) | Instruction::TiledBlock(_) => {
                all_clifford = false;
            }
        }
    }

    let layers = c.layers();
    let depth = layers.len();
    let twoq_depth = layers
        .iter()
        .filter(|layer| {
            layer
                .iter()
                .any(|&i| matches!(&insts[i], Instruction::Gate(g) if g.qubits.len() == 2))
        })
        .count();

    CircuitFeatures {
        num_qubits: c.num_qubits(),
        depth,
        twoq_depth,
        all_clifford,
        all_twoq_nearest_neighbor,
    }
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p aleph-backend select::tests::analyze`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-backend/src/select.rs
git commit -m "[P3-07] analyze: extract CircuitFeatures in one scan"
```

---

## Task 3: Decision rule — `select_from` / `select_backend` / `select_explained`

**Files:**
- Modify: `crates/aleph-backend/src/select.rs`

- [ ] **Step 1: Write the failing tests**

Add inside `mod tests`:

```rust
fn feats(
    num_qubits: u32,
    twoq_depth: usize,
    all_clifford: bool,
    all_twoq_nearest_neighbor: bool,
) -> CircuitFeatures {
    CircuitFeatures {
        num_qubits,
        depth: twoq_depth,
        twoq_depth,
        all_clifford,
        all_twoq_nearest_neighbor,
    }
}

#[test]
fn rule_clifford_picks_stabilizer() {
    // Clifford wins even at huge n.
    let s = select_from(&feats(5000, 100, true, false));
    assert_eq!(s.kind, BackendKind::Stabilizer);
}

#[test]
fn rule_small_nonclifford_picks_statevector() {
    let s = select_from(&feats(20, 50, false, true));
    assert_eq!(s.kind, BackendKind::Statevector);
}

#[test]
fn rule_large_nn_shallow_picks_mps() {
    let s = select_from(&feats(30, 10, false, true));
    assert_eq!(s.kind, BackendKind::Mps);
}

#[test]
fn rule_large_nn_deep_falls_to_statevector() {
    let s = select_from(&feats(30, MPS_DEPTH_THRESHOLD + 1, false, true));
    assert_eq!(s.kind, BackendKind::Statevector);
}

#[test]
fn rule_large_longrange_falls_to_statevector() {
    let s = select_from(&feats(30, 10, false, false));
    assert_eq!(s.kind, BackendKind::Statevector);
}

#[test]
fn select_backend_matches_select_from() {
    let c = bell();
    assert_eq!(select_backend(&c), select_from(&analyze(&c)).kind);
    assert_eq!(select_backend(&c), BackendKind::Stabilizer);
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p aleph-backend select::tests::rule select::tests::select_backend_matches_select_from`
Expected: FAIL — `select_from` / `Selection` / `select_backend` not defined.

- [ ] **Step 3: Implement the rule**

Add to `select.rs` (after `analyze`):

```rust
/// A resolved backend choice plus a one-line human-readable rationale.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Selection {
    /// The chosen backend.
    pub kind: BackendKind,
    /// Why this backend was chosen (for CLI diagnostics).
    pub reason: &'static str,
}

/// Apply the ordered decision rule to pre-computed features. Pure; total.
pub fn select_from(f: &CircuitFeatures) -> Selection {
    if f.all_clifford {
        return Selection {
            kind: BackendKind::Stabilizer,
            reason: "all gates are Clifford",
        };
    }
    if f.num_qubits <= SV_EXACT_CAP {
        return Selection {
            kind: BackendKind::Statevector,
            reason: "exact and fits (n <= 28)",
        };
    }
    if f.all_twoq_nearest_neighbor && f.twoq_depth <= MPS_DEPTH_THRESHOLD {
        return Selection {
            kind: BackendKind::Mps,
            reason: "nearest-neighbor and shallow; too large for exact (n > 28)",
        };
    }
    Selection {
        kind: BackendKind::Statevector,
        reason: "too large for exact and not MPS-suitable",
    }
}

/// Analyze `c` and apply the decision rule, returning the kind + rationale.
pub fn select_explained(c: &Circuit) -> Selection {
    select_from(&analyze(c))
}

/// Analyze `c` and return the chosen backend kind (AC-exact signature).
pub fn select_backend(c: &Circuit) -> BackendKind {
    select_explained(c).kind
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p aleph-backend select`
Expected: PASS (all `select::tests::*`).

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-backend/src/select.rs
git commit -m "[P3-07] select_from/select_backend: ordered decision rule"
```

---

## Task 4: Integration corpus test (real circuits, one per category)

**Files:**
- Create: `crates/aleph-backend/tests/select_corpus.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/aleph-backend/tests/select_corpus.rs`:

```rust
//! P3-07 acceptance corpus: a representative circuit per category selects the
//! expected backend.

use aleph_backend::{select_backend, BackendKind};
use aleph_core::{Gate, GateInstance, Param};
use aleph_ir::Circuit;

/// GHZ on 6 qubits via H + nearest-neighbor CNOTs — all Clifford.
#[test]
fn clifford_ghz_selects_stabilizer() {
    let mut c = Circuit::new(6, 0);
    c.add_gate(GateInstance::new(Gate::H, vec![0u32])).unwrap();
    for q in 0u32..5 {
        c.add_gate(GateInstance::new(Gate::Cnot, vec![q, q + 1]))
            .unwrap();
    }
    assert_eq!(select_backend(&c), BackendKind::Stabilizer);
}

/// Small (n <= 28) non-Clifford circuit — exact state vector.
#[test]
fn small_nonclifford_selects_statevector() {
    let mut c = Circuit::new(10, 0);
    c.add_gate(GateInstance::new(Gate::T, vec![0u32])).unwrap();
    for q in 0u32..9 {
        c.add_gate(GateInstance::new(Gate::Cnot, vec![q, q + 1]))
            .unwrap();
    }
    assert_eq!(select_backend(&c), BackendKind::Statevector);
}

/// 30-qubit nearest-neighbor shallow non-Clifford brickwork — MPS.
#[test]
fn large_nn_shallow_selects_mps() {
    let mut c = Circuit::new(30, 0);
    // 4 shallow layers of NN gates with a non-Clifford rotation to defeat
    // the Clifford rule.
    for _ in 0..4 {
        for q in (0u32..29).step_by(2) {
            c.add_gate(GateInstance::new(Gate::Cnot, vec![q, q + 1]))
                .unwrap();
        }
    }
    c.add_gate(GateInstance::new(Gate::Rz(Param::Concrete(0.3)), vec![0u32]))
        .unwrap();
    assert_eq!(select_backend(&c), BackendKind::Mps);
}

/// 30-qubit non-Clifford circuit with a long-range gate — state vector
/// (with a too-large warning at the CLI layer).
#[test]
fn large_longrange_selects_statevector() {
    let mut c = Circuit::new(30, 0);
    c.add_gate(GateInstance::new(Gate::Rz(Param::Concrete(0.3)), vec![0u32]))
        .unwrap();
    c.add_gate(GateInstance::new(Gate::Cnot, vec![0u32, 29u32]))
        .unwrap();
    assert_eq!(select_backend(&c), BackendKind::Statevector);
}
```

- [ ] **Step 2: Run to verify pass**

Run: `cargo test -p aleph-backend --test select_corpus`
Expected: PASS (4 tests). These exercise the full `analyze` + rule path on real circuits.

- [ ] **Step 3: Commit**

```bash
git add crates/aleph-backend/tests/select_corpus.rs
git commit -m "[P3-07] acceptance corpus: one circuit per backend category"
```

---

## Task 5: CLI `BackendChoice` enum (default `Auto`) + `resolve`

**Files:**
- Modify: `crates/aleph-cli/src/cli.rs` (rename `BackendKind` → `BackendChoice`, add `Auto`, add `resolve`)

> **Rename note:** the CLI's clap enum is currently `BackendKind`. Rename it to
> `BackendChoice` everywhere in the CLI crate so it does not collide with the
> new `aleph_backend::BackendKind`. After this task the CLI references both:
> `BackendChoice` (user-facing, with `Auto`) and `aleph_backend::BackendKind`
> (resolved). Task 6 updates `exec.rs`/`main.rs` call sites.

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` in `crates/aleph-cli/src/cli.rs` (create the module if absent):

```rust
#[cfg(test)]
mod backend_choice_tests {
    use super::*;
    use aleph_backend::BackendKind;
    use aleph_core::{Gate, GateInstance};
    use aleph_ir::Circuit;

    fn clifford() -> Circuit {
        let mut c = Circuit::new(2, 0);
        c.add_gate(GateInstance::new(Gate::H, vec![0u32])).unwrap();
        c.add_gate(GateInstance::new(Gate::Cnot, vec![0u32, 1u32]))
            .unwrap();
        c
    }

    #[test]
    fn explicit_choice_overrides_without_analysis() {
        let c = clifford();
        assert_eq!(
            BackendChoice::Mps.resolve(&c, false),
            BackendKind::Mps
        );
        assert_eq!(
            BackendChoice::Statevector.resolve(&c, false),
            BackendKind::Statevector
        );
    }

    #[test]
    fn auto_picks_stabilizer_for_clifford() {
        assert_eq!(
            BackendChoice::Auto.resolve(&clifford(), false),
            BackendKind::Stabilizer
        );
    }

    #[test]
    fn auto_downgrades_to_sv_when_amplitudes_requested() {
        // Clifford would be stabilizer, but --statevector needs amplitudes.
        assert_eq!(
            BackendChoice::Auto.resolve(&clifford(), true),
            BackendKind::Statevector
        );
    }

    #[test]
    fn default_choice_is_auto() {
        assert_eq!(BackendChoice::default(), BackendChoice::Auto);
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p aleph-cli backend_choice_tests`
Expected: FAIL — `BackendChoice` / `Auto` / `resolve` not defined.

- [ ] **Step 3: Rename + extend the enum, add `resolve`**

In `crates/aleph-cli/src/cli.rs`, replace the `BackendKind` enum (the clap one) with:

```rust
/// Simulation backend selector for `aleph run`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, clap::ValueEnum, Default)]
pub enum BackendChoice {
    /// Pick automatically from circuit structure (default). Clifford →
    /// stabilizer; large nearest-neighbor + shallow → MPS; else state vector.
    #[default]
    Auto,
    /// Dense state vector. Exact; memory grows as 2^n.
    Statevector,
    /// Stabilizer (Clifford-only). O(n²) memory; thousands of qubits.
    Stabilizer,
    /// MPS tensor network (bounded entanglement). χ via --max-bond.
    Mps,
}

impl BackendChoice {
    /// Resolve a user choice into a concrete [`aleph_backend::BackendKind`].
    ///
    /// `Auto` runs the [`aleph_backend::select_explained`] heuristic; an auto
    /// pick of `Stabilizer` is downgraded to `Statevector` when
    /// `wants_amplitudes` is set (`--statevector`/`--force-statevector`),
    /// because the stabilizer backend has no dense state vector. Explicit
    /// choices are returned verbatim (manual override). Diagnostic notes go to
    /// stderr so stdout stays pipeable.
    pub fn resolve(
        self,
        circuit: &aleph_ir::Circuit,
        wants_amplitudes: bool,
    ) -> aleph_backend::BackendKind {
        use aleph_backend::BackendKind as Bk;
        match self {
            BackendChoice::Statevector => Bk::Statevector,
            BackendChoice::Stabilizer => Bk::Stabilizer,
            BackendChoice::Mps => Bk::Mps,
            BackendChoice::Auto => {
                let sel = aleph_backend::select_explained(circuit);
                if sel.kind == Bk::Stabilizer && wants_amplitudes {
                    eprintln!(
                        "auto-selected backend: state vector \
                         (downgraded from stabilizer: --statevector needs amplitudes \
                         the stabilizer backend cannot provide)"
                    );
                    Bk::Statevector
                } else {
                    eprintln!("auto-selected backend: {} ({})", sel.kind, sel.reason);
                    sel.kind
                }
            }
        }
    }
}
```

Then update the `Run` variant's `backend` field type and default:

```rust
        /// Simulation backend: `auto` (default — picks from circuit
        /// structure), `statevector`, `stabilizer` (Clifford-only; rejects
        /// non-Clifford gates and --statevector), or `mps` (tensor network;
        /// bounded entanglement, rejects --statevector).
        #[arg(long, value_enum, default_value_t = BackendChoice::Auto)]
        backend: BackendChoice,
```

- [ ] **Step 4: Ensure `aleph-cli` depends on `aleph-ir`**

Check `crates/aleph-cli/Cargo.toml` `[dependencies]`. It already uses `aleph_ir` indirectly; add an explicit dependency if not present:

Run: `grep -n 'aleph-ir\|aleph-backend\|aleph-core' crates/aleph-cli/Cargo.toml`
Expected: `aleph-backend` and `aleph-core` present. If `aleph-ir` is absent, add under `[dependencies]`:

```toml
aleph-ir = { path = "../aleph-ir" }
```

- [ ] **Step 5: Run to verify pass**

Run: `cargo test -p aleph-cli backend_choice_tests`
Expected: PASS (4 tests). (The `exec.rs`/`main.rs` call sites still reference the old name and will fail to compile in Task 6's scope — if the crate does not build yet because of those, proceed to Task 6 which fixes them, then run this test together with Task 6's.)

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-cli/src/cli.rs crates/aleph-cli/Cargo.toml
git commit -m "[P3-07] CLI: BackendChoice enum (default auto) + resolve"
```

---

## Task 6: Wire resolution into `run_circuit` + too-large warning

**Files:**
- Modify: `crates/aleph-cli/src/exec.rs` (param type, resolve call, dispatch, warning)
- Modify: `crates/aleph-cli/src/main.rs` (pass the renamed type through)

- [ ] **Step 1: Update `main.rs` import + call**

In `crates/aleph-cli/src/main.rs`, the `Cmd::Run { .. backend .. }` destructure already binds `backend`; no change needed to the destructure. The type flows from `cli.rs`. Confirm it still compiles after the rename by building at Step 4.

- [ ] **Step 2: Update `run_circuit` signature + resolution**

In `crates/aleph-cli/src/exec.rs`:

1. Change the import line `use crate::cli::{BackendKind, Precision};` to:

```rust
use crate::cli::{BackendChoice, Precision};
```

2. Change the `backend` parameter type in `run_circuit`'s signature from
   `backend: BackendKind,` to:

```rust
    backend: BackendChoice,
```

3. Immediately after `let n = circuit.num_qubits();` (line ~70), and after the
   expectation/`max_error` validation blocks (so a bad arg still errors before
   any backend work), insert the resolution + too-large warning. Place it just
   before the existing "3. Statevector cap check" block:

```rust
    // 2b. Resolve the backend (runs the auto heuristic for `--backend auto`).
    //     The requested output view gates an auto stabilizer pick: a
    //     state-vector view needs amplitudes the stabilizer backend lacks.
    let wants_amplitudes = print_statevector || force_statevector;
    let resolved = backend.resolve(&circuit, wants_amplitudes);

    // Too-large soft warning: an exact dense run past the soft cap may exhaust
    // memory. We warn and proceed (the user stayed in control by not narrowing
    // the backend); this mirrors the SV soft-cap-warns-not-refuses convention.
    if resolved == aleph_backend::BackendKind::Statevector && n > aleph_backend::SV_EXACT_CAP {
        eprintln!(
            "warning: n={n} exceeds the {}-qubit state-vector soft cap; \
             this run may exhaust memory (override with a different --backend)",
            aleph_backend::SV_EXACT_CAP
        );
    }
```

4. Replace the three dispatch conditionals that compare against the old
   `BackendKind`:

- The cap check guard `if backend == BackendKind::Statevector` →
  `if resolved == aleph_backend::BackendKind::Statevector`
- `if backend == BackendKind::Stabilizer {` →
  `if resolved == aleph_backend::BackendKind::Stabilizer {`
- `if backend == BackendKind::Mps {` →
  `if resolved == aleph_backend::BackendKind::Mps {`

(The final `match precision { … }` fallthrough remains the state-vector path.)

- [ ] **Step 3: Run to verify failure first, then pass**

Run: `cargo build -p aleph-cli`
Expected: compiles clean (no remaining references to the old CLI `BackendKind`).

Run: `cargo test -p aleph-cli`
Expected: PASS — including Task 5's `backend_choice_tests`. Existing tests that
pass an explicit `--backend statevector` still behave identically.

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-cli/src/exec.rs crates/aleph-cli/src/main.rs
git commit -m "[P3-07] CLI: resolve backend in run_circuit + too-large warning"
```

---

## Task 7: CLI integration tests (`assert_cmd`)

**Files:**
- Modify: `crates/aleph-cli/tests/cli.rs`

- [ ] **Step 1: Inspect the existing test helpers**

Run: `sed -n '1,60p' crates/aleph-cli/tests/cli.rs`
Expected: note the helper that writes a temp `.qasm` file and the `Command::cargo_bin("aleph")` pattern. Reuse them; match the existing style for temp-file creation and assertions.

- [ ] **Step 2: Write the integration tests**

Append to `crates/aleph-cli/tests/cli.rs` (adapt `write_temp_qasm` / binary-invocation
helpers to the names already used in the file):

```rust
// --- P3-07 auto backend selection ---

/// A small Clifford program: H + CNOT (Bell pair). `--backend auto` should
/// route to the stabilizer backend and announce it on stderr, while stdout
/// carries the default 1024-shot counts (which the stabilizer backend supports).
#[test]
fn auto_selects_stabilizer_for_clifford() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bell.qasm");
    std::fs::write(
        &path,
        "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\ncx q[0], q[1];\n",
    )
    .unwrap();

    let mut cmd = assert_cmd::Command::cargo_bin("aleph").unwrap();
    cmd.args(["run", path.to_str().unwrap(), "--backend", "auto", "--seed", "1"]);
    cmd.assert()
        .success()
        .stderr(predicates::str::contains("auto-selected backend: stabilizer"));
}

/// `--backend auto --statevector` on a Clifford circuit downgrades to state
/// vector (stabilizer has no amplitudes) and says so on stderr.
#[test]
fn auto_downgrades_to_sv_for_statevector_view() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bell.qasm");
    std::fs::write(
        &path,
        "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\ncx q[0], q[1];\n",
    )
    .unwrap();

    let mut cmd = assert_cmd::Command::cargo_bin("aleph").unwrap();
    cmd.args([
        "run",
        path.to_str().unwrap(),
        "--backend",
        "auto",
        "--statevector",
    ]);
    cmd.assert()
        .success()
        .stderr(predicates::str::contains("downgraded from stabilizer"));
}

/// An explicit `--backend statevector` does NOT print an auto-select line
/// (manual override path bypasses the heuristic).
#[test]
fn explicit_backend_has_no_auto_line() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bell.qasm");
    std::fs::write(
        &path,
        "OPENQASM 3.0;\nqubit[2] q;\nh q[0];\ncx q[0], q[1];\n",
    )
    .unwrap();

    let mut cmd = assert_cmd::Command::cargo_bin("aleph").unwrap();
    cmd.args([
        "run",
        path.to_str().unwrap(),
        "--backend",
        "statevector",
        "--seed",
        "1",
    ]);
    cmd.assert()
        .success()
        .stderr(predicates::str::contains("auto-selected").not());
}
```

> If the file's existing tests use a local `write_qasm(...)` helper and a
> shared `tempdir`, use those instead of inlining `tempfile`/`std::fs::write`.
> Confirm `tempfile` and `predicates` are dev-dependencies (Step 3).

- [ ] **Step 3: Confirm dev-dependencies**

Run: `grep -n 'assert_cmd\|predicates\|tempfile' crates/aleph-cli/Cargo.toml`
Expected: `assert_cmd` and `predicates` present (used by existing tests). If
`tempfile` is absent but existing tests use a hand-rolled temp dir, follow that
pattern instead of adding `tempfile`.

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p aleph-cli --test cli`
Expected: PASS — new auto-selection tests plus the existing suite.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-cli/tests/cli.rs
git commit -m "[P3-07] CLI integration tests: auto-select / downgrade / override"
```

---

## Task 8: Docs — CLAUDE.md backend note + BACKLOG acceptance checkboxes

**Files:**
- Modify: `BACKLOG.md` (tick P3-07 acceptance criteria)
- Modify: `crates/aleph-cli/src/cli.rs` top-level doc / README only if a backend
  list is enumerated there (skip if not present)

- [ ] **Step 1: Tick the acceptance criteria in `BACKLOG.md`**

Find the `[P3-07]` block and change its acceptance checkboxes from `[ ]` to `[x]`:

```
- [x] Heuristic implemented as `select_backend(circuit) -> BackendKind`
- [x] Manual override available
- [x] Test corpus selects expected backend in each category
```

- [ ] **Step 2: Run the full gate locally**

Run:
```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```
Expected: all green. If `cargo fmt --all --check` reports diffs, run
`cargo fmt --all` (NOT `-p`) and re-commit.

- [ ] **Step 3: Commit**

```bash
git add BACKLOG.md
git commit -m "[P3-07] docs: tick acceptance criteria"
```

---

## Task 9: Open the PR

- [ ] **Step 1: Push and open the PR**

```bash
git push -u origin p3-07-auto-backend-select
gh pr create --title "[P3-07] Automatic backend selection heuristic" --body "$(cat <<'EOF'
Closes #38

## Summary
Adds a conservative circuit-analysis heuristic that picks the backend
automatically, exposed as the new CLI default `--backend auto`.

- New `aleph-backend/src/select.rs`: `analyze(&Circuit) -> CircuitFeatures`,
  `select_backend(&Circuit) -> BackendKind` (AC-exact), `select_explained`
  (kind + reason). Pure, read-only, no concrete-backend deps (IR stays
  backend-agnostic).
- Ordered rule: all-Clifford → stabilizer; n≤28 → state vector; large
  nearest-neighbor + shallow → MPS; else → state vector.
- CLI: `BackendChoice { Auto, Statevector, Stabilizer, Mps }` (default `Auto`).
  Explicit choices are manual overrides. An auto stabilizer pick is downgraded
  to state vector when `--statevector` is requested (stabilizer has no
  amplitudes). Too-large (n>28, exact) prints a stderr warning and proceeds.

## Conservative MPS rule
MPS (bounded-χ, approximate) is only chosen when the state vector cannot fit
(n>28) AND the circuit is all-nearest-neighbor AND shallow — never when an exact
backend works, avoiding silent approximation.

## Test results
- Unit (`select_from`): one test per rule arm.
- Corpus (`tests/select_corpus.rs`): real circuit per category selects the
  expected backend (the AC corpus).
- CLI (`assert_cmd`): `--backend auto` announces the pick on stderr; the
  `--statevector` downgrade note fires; explicit `--backend` skips the heuristic.
- `cargo test --workspace`, `clippy -D warnings`, `cargo fmt --all --check` green.

## Notes
- `auto` is now the default backend. Default output is sampling counts (every
  backend supports it), so routing a Clifford circuit to stabilizer by default
  is output-compatible.
- Reset is unsupported on every backend, so it is intentionally not a selection
  feature.
- Last Phase-3 ticket.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 2: Watch gating CI**

Run: `gh pr checks <PR#>`
Gating = rustfmt / clippy / test linux+macos. Self-hosted `bench` is non-gating.
On format failure: `cargo fmt --all`, re-push.

---

## Self-Review notes

- **Spec coverage:** AC `select_backend(circuit) -> BackendKind` → Task 3;
  manual override → Task 5 (`resolve` explicit arms) + Task 7 test; test corpus
  per category → Task 4 + Task 7. CLI default `auto` → Task 5. Too-large
  warn+proceed → Task 6. Conservative MPS rule → Task 3 ordering. View-downgrade
  → Task 5/6/7.
- **Type consistency:** `BackendKind` (aleph-backend) vs `BackendChoice`
  (aleph-cli) are distinct by design; `resolve` returns `aleph_backend::BackendKind`.
  `Selection { kind, reason }`, `CircuitFeatures { num_qubits, depth, twoq_depth,
  all_clifford, all_twoq_nearest_neighbor }` used identically across Tasks 2/3/5.
  Constants `SV_EXACT_CAP` (u32, 28) and `MPS_DEPTH_THRESHOLD` (usize, 64) used
  in Tasks 1/3/6.
- **No placeholders:** every code step is concrete.
