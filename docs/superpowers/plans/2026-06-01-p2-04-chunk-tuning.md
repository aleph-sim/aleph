# P2-04 Chunked Parallelism Tuning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the SV parallel-chunk knobs (sequential cutoff + rayon grain) tunable per gate-type and target-qubit position via a CPU-model-selected table, threaded explicitly into `par_blocks`/`par_units`, with an honest cross-box benchmark.

**Architecture:** A new pure `kernels::tuning` module owns `ChunkPolicy { min_amps, grain }`, runtime CPU detection (`RefCpu`), a `GateClass × PosClass` lookup table, and `resolve_policy()`. Each **leaf** kernel computes its own policy one line before its `par_blocks`/`par_units` call (it already *is* a gate class and has `target`+`len`). `RefCpu::Generic` returns today's defaults, so unknown hardware and untuned cells are byte-for-byte unchanged. Env vars `ALEPH_PAR_MIN_AMPS`/`ALEPH_PAR_GRAIN` override per-field and double as the sweep instrument.

**Tech Stack:** Rust 2021, rayon, criterion (`internal-bench` feature), `std::arch::x86_64::__cpuid` for brand detection. Bench boxes: EPYC 8124P (AVX-512) + Ryzen 9 3900 (scalar).

**Spec:** `docs/superpowers/specs/2026-06-01-p2-04-chunk-tuning-design.md`

---

## File Structure

- **Create** `crates/aleph-sv/src/kernels/tuning.rs` — all policy types, CPU detection, table, `resolve_policy`. Pure + unit-tested. Single responsibility: "given a gate class + position, what chunk policy?"
- **Modify** `crates/aleph-sv/src/kernels/mod.rs` — add `mod tuning;`, change `par_blocks`/`par_units` to take `ChunkPolicy`, delete `par_min_amps()`, update `par_tests`.
- **Modify** `crates/aleph-sv/src/kernels/aos.rs` — add one `resolve_policy(...)` line before each `par_blocks`/`par_units` call.
- **Modify** `crates/aleph-sv/src/kernels/soa.rs` — same.
- **Create** `crates/aleph-sv/tests/policy_invariance.rs` — end-to-end "varying policy ⇒ bit-identical amplitudes" oracle.
- **Create** `crates/aleph-sv/benches/chunk_tune.rs` — per-(gate,target) micro-bench reading the env knobs (the sweep instrument).
- **Create** `scripts/tune-chunks.sh` — drives the grid sweep, parses criterion medians, prints best-per-cell.
- **Create** `docs/perf/phase2-p2-04.md` — the honest before/after report.

---

## Task 1: `tuning` module — types, CPU detection, table, resolve

**Files:**
- Create: `crates/aleph-sv/src/kernels/tuning.rs`
- Modify: `crates/aleph-sv/src/kernels/mod.rs` (add `mod tuning;` near the other `mod` decls, ~line 25)

- [ ] **Step 1: Write the failing tests** (append to `tuning.rs` as you create it, but write them first mentally / top of file `#[cfg(test)] mod tests`)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_cell_is_the_legacy_default() {
        for class in [GateClass::OneQGeneric, GateClass::OneQDiag, GateClass::TwoQCnot,
                      GateClass::TwoQDiag, GateClass::ThreeQ] {
            for pos in [PosClass::Low, PosClass::Mid, PosClass::High] {
                assert_eq!(chunk_policy(RefCpu::Generic, class, pos), DEFAULT_POLICY);
            }
        }
    }

    #[test]
    fn pos_class_boundaries() {
        // n = 25: Low if target < 2; High if target + 2 >= 25 (i.e. >= 23); else Mid.
        assert_eq!(pos_class(0, 25), PosClass::Low);
        assert_eq!(pos_class(1, 25), PosClass::Low);
        assert_eq!(pos_class(2, 25), PosClass::Mid);
        assert_eq!(pos_class(22, 25), PosClass::Mid);
        assert_eq!(pos_class(23, 25), PosClass::High);
        assert_eq!(pos_class(24, 25), PosClass::High);
    }

    #[test]
    fn pos_class_small_n_does_not_underflow() {
        // n = 2 (Bell state). target in {0,1} → Low; never panics.
        assert_eq!(pos_class(0, 2), PosClass::Low);
        assert_eq!(pos_class(1, 2), PosClass::Low);
    }

    #[test]
    fn cpu_model_env_override() {
        // detect_cpu is the uncached worker so tests don't fight the OnceLock.
        assert_eq!(detect_cpu_from(Some("epyc"), None), RefCpu::Epyc8124P);
        assert_eq!(detect_cpu_from(Some("ryzen"), None), RefCpu::Ryzen3900);
        assert_eq!(detect_cpu_from(Some("generic"), None), RefCpu::Generic);
        assert_eq!(detect_cpu_from(Some("nonsense"), None), RefCpu::Generic);
    }

    #[test]
    fn cpu_model_brand_match() {
        assert_eq!(detect_cpu_from(None, Some("AMD EPYC 8124P 16-Core Processor")),
                   RefCpu::Epyc8124P);
        assert_eq!(detect_cpu_from(None, Some("AMD Ryzen 9 3900 12-Core Processor")),
                   RefCpu::Ryzen3900);
        assert_eq!(detect_cpu_from(None, Some("Intel(R) Xeon(R) Silver 4114")),
                   RefCpu::Generic);
    }
}
```

- [ ] **Step 2: Write the module implementation**

```rust
//! P2-04: per-(gate, qubit-position) chunk-size policy.
//!
//! The SV kernels parallelise via `par_blocks`/`par_units`, whose two
//! knobs — the sequential cutoff (`min_amps`) and the rayon grain
//! (`grain`, i.e. `with_min_len`) — depend on the gate (work per
//! amplitude) and the target qubit (stride / `par_units` regime). This
//! module maps `(cpu_model, gate_class, position) -> ChunkPolicy`.
//!
//! Design: `docs/superpowers/specs/2026-06-01-p2-04-chunk-tuning-design.md`.
//!
//! No-regression contract: `RefCpu::Generic` (and every untuned cell)
//! returns `DEFAULT_POLICY` == the pre-P2-04 hardcoded values, so unknown
//! hardware behaves exactly as before. Results are bit-identical for ANY
//! policy: the knobs only re-partition disjoint-write tasks, never reorder
//! a floating-point reduction (see `par_blocks` doc).

use std::sync::OnceLock;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct ChunkPolicy {
    pub(crate) min_amps: usize,
    pub(crate) grain: usize,
}

/// The pre-P2-04 hardcoded values: sequential below 2^18 amplitudes,
/// rayon `with_min_len(64)`.
pub(crate) const DEFAULT_POLICY: ChunkPolicy = ChunkPolicy {
    min_amps: 1 << 18,
    grain: 64,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum GateClass {
    OneQGeneric,
    OneQDiag,
    OneQAntidiag,
    TwoQDense,
    TwoQCnot,
    TwoQCz,
    TwoQSwap,
    TwoQDiag,
    ThreeQ,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum PosClass {
    Low,
    Mid,
    High,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RefCpu {
    Epyc8124P,
    Ryzen3900,
    Generic,
}

/// Target-position buckets. Constants are design choices, not tuned
/// (see spec §"Out of scope"). `+ HIGH_BAND` (not `n - HIGH_BAND`)
/// avoids underflow for small `n`.
const LOW_BAND: u32 = 2;
const HIGH_BAND: u32 = 2;

/// Classify by the *dominant* (maximum) target index — that governs the
/// outer stride and whether `par_units` flattening is in play.
pub(crate) fn pos_class(max_target: u32, n: u32) -> PosClass {
    if max_target < LOW_BAND {
        PosClass::Low
    } else if max_target + HIGH_BAND >= n {
        PosClass::High
    } else {
        PosClass::Mid
    }
}

/// The tuned table. Populated for high-traffic cells in Task 5; every
/// other cell (and all of `Generic`) returns `DEFAULT_POLICY`.
pub(crate) fn chunk_policy(cpu: RefCpu, _class: GateClass, _pos: PosClass) -> ChunkPolicy {
    match cpu {
        RefCpu::Generic => DEFAULT_POLICY,
        // Task 5 replaces these arms with measured per-(class,pos) values.
        RefCpu::Epyc8124P => DEFAULT_POLICY,
        RefCpu::Ryzen3900 => DEFAULT_POLICY,
    }
}

/// Resolve the effective policy for a kernel invocation. Precedence:
/// test override → env per-field override → table.
#[inline]
pub(crate) fn resolve_policy(class: GateClass, pos: PosClass) -> ChunkPolicy {
    #[cfg(test)]
    {
        if let Some(p) = test_override::get() {
            return p;
        }
    }
    let mut p = chunk_policy(cpu_model(), class, pos);
    if let Some(v) = env_min_amps() {
        p.min_amps = v;
    }
    if let Some(v) = env_grain() {
        p.grain = v;
    }
    p
}

fn env_min_amps() -> Option<usize> {
    static V: OnceLock<Option<usize>> = OnceLock::new();
    *V.get_or_init(|| std::env::var("ALEPH_PAR_MIN_AMPS").ok().and_then(|s| s.parse().ok()))
}

fn env_grain() -> Option<usize> {
    static V: OnceLock<Option<usize>> = OnceLock::new();
    *V.get_or_init(|| std::env::var("ALEPH_PAR_GRAIN").ok().and_then(|s| s.parse().ok()))
}

pub(crate) fn cpu_model() -> RefCpu {
    static M: OnceLock<RefCpu> = OnceLock::new();
    *M.get_or_init(|| {
        let env = std::env::var("ALEPH_CPU_MODEL").ok();
        detect_cpu_from(env.as_deref(), cpu_brand_string().as_deref())
    })
}

/// Pure detection worker (testable without touching the `OnceLock`s or
/// real CPUID). `env` is `ALEPH_CPU_MODEL`; `brand` is the CPUID brand
/// string. Env wins.
fn detect_cpu_from(env: Option<&str>, brand: Option<&str>) -> RefCpu {
    if let Some(e) = env {
        return match e.to_ascii_lowercase().as_str() {
            "epyc" => RefCpu::Epyc8124P,
            "ryzen" => RefCpu::Ryzen3900,
            _ => RefCpu::Generic,
        };
    }
    if let Some(b) = brand {
        if b.contains("EPYC 8124P") {
            return RefCpu::Epyc8124P;
        }
        if b.contains("Ryzen 9 3900") {
            return RefCpu::Ryzen3900;
        }
    }
    RefCpu::Generic
}

#[cfg(target_arch = "x86_64")]
fn cpu_brand_string() -> Option<String> {
    use std::arch::x86_64::__cpuid;
    // SAFETY: `__cpuid` is always callable on x86_64. We first read the
    // max extended leaf (0x8000_0000); the brand-string leaves
    // 0x8000_0002..=0x8000_0004 are valid only if it is >= 0x8000_0004.
    // No memory is touched; the intrinsic only reads CPU registers.
    unsafe {
        if __cpuid(0x8000_0000).eax < 0x8000_0004 {
            return None;
        }
        let mut bytes = Vec::with_capacity(48);
        for leaf in [0x8000_0002u32, 0x8000_0003, 0x8000_0004] {
            let r = __cpuid(leaf);
            for reg in [r.eax, r.ebx, r.ecx, r.edx] {
                bytes.extend_from_slice(&reg.to_le_bytes());
            }
        }
        // Trim trailing NULs/spaces.
        let s = String::from_utf8_lossy(&bytes);
        Some(s.trim_end_matches(['\0', ' ']).to_string())
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn cpu_brand_string() -> Option<String> {
    None
}

/// Test-only policy override (thread-local), so the invariance test can
/// force several policies in one process without fighting the env
/// `OnceLock`s.
#[cfg(test)]
pub(crate) mod test_override {
    use super::ChunkPolicy;
    use std::cell::Cell;
    thread_local! {
        static OVERRIDE: Cell<Option<ChunkPolicy>> = const { Cell::new(None) };
    }
    pub(crate) fn set(p: Option<ChunkPolicy>) {
        OVERRIDE.with(|c| c.set(p));
    }
    pub(crate) fn get() -> Option<ChunkPolicy> {
        OVERRIDE.with(|c| c.get())
    }
}
```

- [ ] **Step 3: Wire the module into `mod.rs`**

In `crates/aleph-sv/src/kernels/mod.rs`, add alongside the other `mod` declarations (after the `soa` block, ~line 25):

```rust
mod tuning;
```

- [ ] **Step 4: Run the tuning tests**

Run: `cargo test -p aleph-sv tuning -- --nocapture`
Expected: all 6 tests PASS. (The crate as a whole won't fully build yet only if `mod.rs` already references things — it shouldn't; `tuning` is self-contained. If `cargo test -p aleph-sv` fails to compile for unrelated reasons, scope to the lib: `cargo test -p aleph-sv --lib tuning`.)

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/src/kernels/tuning.rs crates/aleph-sv/src/kernels/mod.rs
git commit -m "[P2-04] Add kernels::tuning (ChunkPolicy table + CPU detect)"
```

---

## Task 2: Thread `ChunkPolicy` through `par_blocks`/`par_units` + all call sites

This is one atomic refactor (the signature change breaks every caller; the commit must compile). The transformation is uniform — apply the recipe at every call site.

**Files:**
- Modify: `crates/aleph-sv/src/kernels/mod.rs` (`par_blocks`, `par_units`, delete `par_min_amps`, fix `par_tests`)
- Modify: `crates/aleph-sv/src/kernels/aos.rs` (every `par_blocks`/`par_units` call)
- Modify: `crates/aleph-sv/src/kernels/soa.rs` (every `par_blocks`/`par_units` call)

- [ ] **Step 1: Change the helper signatures in `mod.rs`**

Replace `par_blocks` (currently ~line 127) and delete `par_min_amps` (~line 103):

```rust
use crate::kernels::tuning::ChunkPolicy;

pub(crate) fn par_blocks(
    policy: ChunkPolicy,
    count: usize,
    len: usize,
    block_of: impl Fn(usize) -> usize + Sync,
    body: impl Fn(usize) + Sync,
) {
    if len < policy.min_amps {
        for k in 0..count {
            body(block_of(k));
        }
    } else {
        use rayon::prelude::*;
        (0..count)
            .into_par_iter()
            .with_min_len(policy.grain.max(1)) // grain==0 would panic in rayon
            .for_each(|k| body(block_of(k)));
    }
}
```

Replace `par_units` (~line 170) to take and forward the policy:

```rust
pub(crate) fn par_units(
    policy: ChunkPolicy,
    outer_count: usize,
    units_per_block: usize,
    stride: usize,
    len: usize,
    base_of: impl Fn(usize) -> usize + Sync,
    body: impl Fn(usize) + Sync,
) {
    debug_assert!(units_per_block.is_power_of_two());
    let total = outer_count * units_per_block;
    let unit_bits = units_per_block.trailing_zeros();
    let unit_mask = units_per_block - 1;
    par_blocks(
        policy,
        total,
        len,
        move |u| base_of(u >> unit_bits) + (u & unit_mask) * stride,
        body,
    );
}
```

Delete the now-unused `par_min_amps` fn and update the `par_blocks` doc comment's reference to it (say "sequential when `len < policy.min_amps`"). Also update the doc comment on the old `MIN`/`par_min_amps` block (~lines 93-111) — remove it; the env knob now lives in `tuning::resolve_policy`.

- [ ] **Step 2: Fix `par_tests` in `mod.rs`**

`coverage` must pass a policy. Toggle the branch via `min_amps`, and also sweep grains to prove partitioning is exhaustive+disjoint for any grain:

```rust
#[cfg(test)]
mod par_tests {
    use super::par_blocks;
    use crate::kernels::tuning::ChunkPolicy;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn coverage(count: usize, policy: ChunkPolicy, len: usize) -> Vec<usize> {
        let hits: Vec<AtomicUsize> = (0..count).map(|_| AtomicUsize::new(0)).collect();
        par_blocks(policy, count, len, |k| k, |slot| {
            hits[slot].fetch_add(1, Ordering::Relaxed);
        });
        hits.iter().map(|a| a.load(Ordering::Relaxed)).collect()
    }

    #[test]
    fn par_blocks_visits_each_block_once_sequential() {
        let p = ChunkPolicy { min_amps: usize::MAX, grain: 64 };
        assert!(coverage(1000, p, 0).iter().all(|&h| h == 1));
    }

    #[test]
    fn par_blocks_visits_each_block_once_parallel_across_grains() {
        for grain in [1usize, 16, 64, 1024] {
            let p = ChunkPolicy { min_amps: 0, grain };
            assert!(
                coverage(1000, p, usize::MAX).iter().all(|&h| h == 1),
                "grain={grain} dropped/duplicated a block"
            );
        }
    }
}
```

- [ ] **Step 3: Apply the call-site recipe in `aos.rs` and `soa.rs`**

For **every** `par_blocks(...)` / `par_units(...)` call, insert one line directly above it:

```rust
let policy = crate::kernels::tuning::resolve_policy(
    crate::kernels::tuning::GateClass::<CLASS>,
    crate::kernels::tuning::pos_class(<MAX_TARGET>, (len.trailing_zeros())),
);
```

then pass `policy` as the **new first argument**. Use `len` if in scope, else derive `n` from the buffer: AoS `amps.len().trailing_zeros()`, SoA `re.len().trailing_zeros()`.

`<CLASS>` is fixed by the enclosing fn (the fn *is* that gate class):

| Enclosing fn (matches both `_scalar` and `_avx512*` variants) | `<CLASS>` | `<MAX_TARGET>` |
|---|---|---|
| `apply_1q_avx512` (generic 2×2) | `OneQGeneric` | `target` |
| `apply_1q_diagonal_*` | `OneQDiag` | `target` |
| `apply_1q_x_*`, `apply_1q_y_*`, `apply_1q_antidiag_*` (incl. `_lowbit`, and SoA `*_soa*`) | `OneQAntidiag` | `target` |
| `apply_2q_dense_scalar`, `apply_2q_avx512` | `TwoQDense` | `targets[0].max(targets[1])` |
| `apply_2q_cnot_*` (incl. `_tier_b/_c`) | `TwoQCnot` | `control.max(target)` (or `targets[0].max(targets[1])`) |
| `apply_2q_cz_*` | `TwoQCz` | `targets[0].max(targets[1])` |
| `apply_2q_swap_*` | `TwoQSwap` | `targets[0].max(targets[1])` |
| `apply_2q_diagonal_*` | `TwoQDiag` | `targets[0].max(targets[1])` |
| `apply_3q_generic*`, toffoli/ccz kernels under `dispatch_toffoli*`/`dispatch_ccz*` | `ThreeQ` | `targets.iter().copied().max().unwrap()` |

Add `use crate::kernels::tuning::{self, GateClass, ChunkPolicy};` at the top of `aos.rs` and `soa.rs` to shorten the lines if preferred (then `tuning::resolve_policy`, `GateClass::OneQDiag`, `tuning::pos_class`).

Find every site with:
```bash
grep -nE "par_blocks\(|par_units\(" crates/aleph-sv/src/kernels/aos.rs crates/aleph-sv/src/kernels/soa.rs \
  | grep -v "fn par_"
```
Edit each. The compiler will flag any you miss (wrong arg count).

- [ ] **Step 4: Build**

Run: `cargo build -p aleph-sv --all-targets`
Expected: clean build, no errors. If "this function takes N arguments but N-1 were supplied" appears, you missed a call site — fix it.

- [ ] **Step 5: Run the full SV test suite (default policy ⇒ unchanged behavior)**

Run: `cargo test -p aleph-sv`
Expected: ALL pass, including the existing oracle/equivalence tests (`all_fixtures_match_naive`, SoA≡AoS, thread-sweep). Default policy == old constants, so nothing should move.

- [ ] **Step 6: Clippy + fmt**

Run: `cargo clippy -p aleph-sv --all-targets -- -D warnings && cargo fmt --check`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add crates/aleph-sv/src/kernels/
git commit -m "[P2-04] Thread ChunkPolicy into par_blocks/par_units (leaf-local resolve)"
```

---

## Task 3: End-to-end policy-invariance oracle test

Proves the deliverable's correctness guarantee: changing the chunk policy never changes amplitudes (it only re-partitions disjoint-write tasks).

**Files:**
- Create: `crates/aleph-sv/tests/policy_invariance.rs`

- [ ] **Step 1: Write the test**

```rust
//! P2-04: applying any gate under different ChunkPolicy values must
//! produce bit-identical amplitudes. The knobs only change task
//! partitioning, never which amplitude a body writes and never a
//! cross-thread FP reduction — so equality is exact, not within a
//! tolerance. Guards against a future kernel accidentally letting the
//! policy leak into results.

use aleph_core::Complex;
use aleph_sv::kernels::{self, tuning};

fn seeded_state(n: u32) -> Vec<Complex> {
    (0..(1usize << n))
        .map(|k| {
            let r = ((k as u64).wrapping_mul(2_654_435_761) as f64) * 1e-19;
            Complex::new(r.sin(), r.cos())
        })
        .collect()
}

fn h_matrix() -> [[Complex; 2]; 2] {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    [[Complex::new(s, 0.0), Complex::new(s, 0.0)],
     [Complex::new(s, 0.0), Complex::new(-s, 0.0)]]
}

/// Force `policy`, run `f`, restore. Single-threaded test, so the
/// thread-local override is visible for the whole synchronous call —
/// including inside the rayon fan-out, because `par_blocks` consults the
/// policy passed *as an argument*, not the thread-local. (The override
/// only feeds `resolve_policy` at the leaf, on this same thread.)
fn with_policy<R>(policy: tuning::ChunkPolicy, f: impl FnOnce() -> R) -> R {
    tuning::test_override::set(Some(policy));
    let r = f();
    tuning::test_override::set(None);
    r
}

const POLICIES: &[tuning::ChunkPolicy] = &[
    tuning::ChunkPolicy { min_amps: 0, grain: 1 },           // always-parallel, finest
    tuning::ChunkPolicy { min_amps: 0, grain: 4096 },        // always-parallel, coarse
    tuning::ChunkPolicy { min_amps: usize::MAX, grain: 64 }, // always-sequential
    tuning::ChunkPolicy { min_amps: 1 << 18, grain: 64 },    // default
];

fn assert_invariant(reference: &[Complex], state: &[Complex], label: &str) {
    assert_eq!(reference.len(), state.len());
    for (i, (a, b)) in reference.iter().zip(state).enumerate() {
        assert_eq!(a.re.to_bits(), b.re.to_bits(), "{label}: re mismatch at {i}");
        assert_eq!(a.im.to_bits(), b.im.to_bits(), "{label}: im mismatch at {i}");
    }
}

#[test]
fn one_q_generic_h_is_policy_invariant() {
    let n = 12; // > 0 so the parallel branch is exercised when min_amps==0
    let m = h_matrix();
    for &target in &[0u32, 5, 11] {
        let reference = {
            let mut s = seeded_state(n);
            with_policy(POLICIES[3], || kernels::aos::apply_1q(&mut s, target, &[], &m));
            s
        };
        for p in POLICIES {
            let mut s = seeded_state(n);
            with_policy(*p, || kernels::aos::apply_1q(&mut s, target, &[], &m));
            assert_invariant(&reference, &s, &format!("H target={target} policy={p:?}"));
        }
    }
}

#[test]
fn cnot_is_policy_invariant() {
    let n = 12;
    let m = {
        let z = Complex::new(0.0, 0.0);
        let o = Complex::new(1.0, 0.0);
        [[o, z, z, z], [z, o, z, z], [z, z, z, o], [z, z, o, z]]
    };
    let reference = {
        let mut s = seeded_state(n);
        with_policy(POLICIES[3], || kernels::aos::apply_2q(&mut s, [3, 7], &[], &m));
        s
    };
    for p in POLICIES {
        let mut s = seeded_state(n);
        with_policy(*p, || kernels::aos::apply_2q(&mut s, [3, 7], &[], &m));
        assert_invariant(&reference, &s, &format!("CNOT policy={p:?}"));
    }
}
```

Note: this test needs `kernels`, `kernels::tuning`, and `kernels::tuning::test_override` reachable from an integration test. They are crate-internal. Gate the test crate's access by enabling the `internal-bench` feature (which already makes `kernels::aos`/`soa` public) **and** make `tuning` + `test_override` public under that feature.

- [ ] **Step 2: Expose `tuning` under `internal-bench`**

In `crates/aleph-sv/src/kernels/mod.rs`, change `mod tuning;` to mirror the `aos`/`soa` pattern:

```rust
#[cfg(not(feature = "internal-bench"))]
pub(crate) mod tuning;
#[cfg(feature = "internal-bench")]
pub mod tuning;
```

And in `tuning.rs`, the `test_override` module + `ChunkPolicy` fields are needed by the integration test. Since integration tests don't get `#[cfg(test)]`, expose `test_override` whenever `internal-bench` is on:

```rust
#[cfg(any(test, feature = "internal-bench"))]
pub(crate) mod test_override { /* ... as Task 1 ... */ }
```

and make `resolve_policy` consult it under the same cfg:

```rust
#[cfg(any(test, feature = "internal-bench"))]
{
    if let Some(p) = test_override::get() {
        return p;
    }
}
```

Make `ChunkPolicy`'s fields `pub` (they're already `pub(crate)`; under `internal-bench` the type is reachable, fields stay `pub(crate)` which is visible to the integration test only if it's the same crate — integration tests are a *separate* crate, so promote the fields and the type to `pub` when `internal-bench` is on, or add `pub(crate)`→`pub`). Simplest: declare `ChunkPolicy`, `GateClass`, `PosClass`, `RefCpu`, `resolve_policy`, `pos_class`, `chunk_policy`, `DEFAULT_POLICY` as `pub` (the module itself is only `pub` under `internal-bench`, so this leaks nothing in normal builds).

- [ ] **Step 3: Run the invariance test**

Run: `cargo test -p aleph-sv --features internal-bench --test policy_invariance`
Expected: both tests PASS (exact bit-equality across all policies).

- [ ] **Step 4: Re-run the default suite to confirm no `pub` change broke anything**

Run: `cargo test -p aleph-sv && cargo clippy -p aleph-sv --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-sv/
git commit -m "[P2-04] Policy-invariance oracle: any ChunkPolicy ⇒ identical amplitudes"
```

---

## Task 4: Sweep bench + driver script (the empirical instrument)

**Files:**
- Create: `crates/aleph-sv/benches/chunk_tune.rs`
- Modify: `crates/aleph-sv/Cargo.toml` (register the bench, `required-features = ["internal-bench"]`)
- Create: `scripts/tune-chunks.sh`

- [ ] **Step 1: Write the bench**

```rust
//! P2-04 sweep instrument. Applies ONE gate class at ONE target on a
//! large state, repeatedly, so a driver can vary ALEPH_PAR_MIN_AMPS /
//! ALEPH_PAR_GRAIN (the env override in `tuning::resolve_policy`) and
//! read criterion's median per grid point.
//!
//! Env:
//!   ALEPH_TUNE_GATE   = h|zdiag|x|dense|cnot|cz|swap|cphase|toffoli
//!   ALEPH_TUNE_TARGET = target qubit index (default 12)
//!   ALEPH_TUNE_N      = qubit count (default 25)
//!   ALEPH_PAR_MIN_AMPS / ALEPH_PAR_GRAIN = the knobs under test

use aleph_core::Complex;
use aleph_sv::kernels;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn env_u32(k: &str, d: u32) -> u32 { std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d) }

fn seeded_state(n: u32) -> Vec<Complex> {
    (0..(1usize << n)).map(|k| {
        let r = ((k as u64).wrapping_mul(2_654_435_761) as f64) * 1e-19;
        Complex::new(r.sin(), r.cos())
    }).collect()
}

fn bench(c: &mut Criterion) {
    let gate = std::env::var("ALEPH_TUNE_GATE").unwrap_or_else(|_| "cphase".into());
    let t = env_u32("ALEPH_TUNE_TARGET", 12);
    let n = env_u32("ALEPH_TUNE_N", 25);
    let mut s = seeded_state(n);
    let id = format!("chunk_tune/{gate}/t{t}/n{n}");

    macro_rules! run1 { ($m:expr) => {{
        let m = $m; c.bench_function(&id, |b| b.iter(|| kernels::aos::apply_1q(black_box(&mut s), t, &[], &m)));
    }};}
    macro_rules! run2 { ($m:expr, $q:expr) => {{
        let m = $m; let q = $q; c.bench_function(&id, |b| b.iter(|| kernels::aos::apply_2q(black_box(&mut s), q, &[], &m)));
    }};}

    let z = Complex::new(0.0, 0.0);
    let o = Complex::new(1.0, 0.0);
    let sq = std::f64::consts::FRAC_1_SQRT_2;
    let q2 = [t, t.saturating_sub(1)]; // two distinct qubits near the target

    match gate.as_str() {
        "h"     => run1!([[Complex::new(sq,0.0),Complex::new(sq,0.0)],[Complex::new(sq,0.0),Complex::new(-sq,0.0)]]),
        "zdiag" => run1!([[o,z],[z,Complex::new(-1.0,0.0)]]),
        "x"     => run1!([[z,o],[o,z]]),
        "cphase"=> run2!([[o,z,z,z],[z,o,z,z],[z,z,o,z],[z,z,z,Complex::new(0.0,1.0)]], q2),
        "cnot"  => run2!([[o,z,z,z],[z,o,z,z],[z,z,z,o],[z,z,o,z]], q2),
        "cz"    => run2!([[o,z,z,z],[z,o,z,z],[z,z,o,z],[z,z,z,Complex::new(-1.0,0.0)]], q2),
        "swap"  => run2!([[o,z,z,z],[z,z,o,z],[z,o,z,z],[z,z,z,o]], q2),
        "dense" => run2!([[Complex::new(0.5,0.5),z,z,Complex::new(0.5,-0.5)],[z,o,z,z],[z,z,o,z],[Complex::new(0.5,-0.5),z,z,Complex::new(0.5,0.5)]], q2),
        other   => panic!("unknown ALEPH_TUNE_GATE={other}"),
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
```

(`toffoli` deferred — 3q is a low-traffic cell for the sweep; add later if a cell warrants it.)

- [ ] **Step 2: Register the bench in `Cargo.toml`**

Append to `crates/aleph-sv/Cargo.toml` (mirror the existing `[[bench]]` blocks):

```toml
[[bench]]
name = "chunk_tune"
harness = false
required-features = ["internal-bench"]
```

- [ ] **Step 3: Verify the bench builds and runs one point locally**

Run: `ALEPH_TUNE_GATE=h ALEPH_TUNE_N=12 cargo bench -p aleph-sv --features internal-bench --bench chunk_tune -- --warm-up-time 1 --measurement-time 2`
Expected: builds, runs, prints a `chunk_tune/h/t12/n12` timing. (Small n locally — just a smoke test. Real sweep is n=25 on the boxes.)

- [ ] **Step 4: Write the driver script**

```bash
# scripts/tune-chunks.sh — P2-04 chunk-size grid sweep.
# Usage: ALEPH_CPU_MODEL=epyc ./scripts/tune-chunks.sh 2>&1 | tee tune-$(hostname).log
# Run ONLY on a verified-idle box (uptime ~0; no cargo bench / bencher run).
set -euo pipefail

GATES=("h" "zdiag" "cnot" "cphase")          # high-traffic Tier-1 classes
TARGETS=(1 12 24)                            # Low / Mid / High position buckets
MIN_AMPS=(65536 131072 262144 524288 1048576)
GRAINS=(16 32 64 128 256 512)
N="${ALEPH_TUNE_N:-25}"

echo "# host=$(hostname) cpu_model=${ALEPH_CPU_MODEL:-auto} n=$N"
echo "# gate target min_amps grain median_ns"
for g in "${GATES[@]}"; do
  for t in "${TARGETS[@]}"; do
    for ma in "${MIN_AMPS[@]}"; do
      for gr in "${GRAINS[@]}"; do
        out=$(ALEPH_TUNE_GATE="$g" ALEPH_TUNE_TARGET="$t" ALEPH_TUNE_N="$N" \
              ALEPH_PAR_MIN_AMPS="$ma" ALEPH_PAR_GRAIN="$gr" \
              RUSTFLAGS="-C target-cpu=native" \
              cargo bench -p aleph-sv --features internal-bench --bench chunk_tune \
                -- --warm-up-time 1 --measurement-time 3 --noplot 2>/dev/null)
        # criterion prints e.g. "time:   [12.3 ms 12.4 ms 12.5 ms]" — grab the median.
        med=$(echo "$out" | grep -oE 'time:[[:space:]]*\[[^]]+\]' | head -1 | awk '{print $3, $4}')
        echo "$g $t $ma $gr $med"
      done
    done
  done
done
```

- [ ] **Step 5: `chmod` + smoke-run one cell of the driver locally**

Run: `chmod +x scripts/tune-chunks.sh && ALEPH_TUNE_N=12 ALEPH_CPU_MODEL=generic bash -c 'ALEPH_TUNE_GATE=h ALEPH_TUNE_TARGET=5 ALEPH_PAR_MIN_AMPS=0 ALEPH_PAR_GRAIN=64 cargo bench -p aleph-sv --features internal-bench --bench chunk_tune -- --measurement-time 2 --noplot' >/dev/null && echo OK`
Expected: `OK` (validates the env plumbing end-to-end; full sweep happens in Task 5).

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-sv/benches/chunk_tune.rs crates/aleph-sv/Cargo.toml scripts/tune-chunks.sh
git commit -m "[P2-04] Sweep instrument: chunk_tune bench + tune-chunks.sh driver"
```

---

## Task 5: Run the sweep on EPYC + Ryzen, populate the table

Operational — runs on the bench boxes (`[[aleph-bench-server]]` EPYC `root@195.154.249.85`, `[[aleph-bench-server-2]]` Ryzen `root@49.12.173.85`). The Ryzen origin is a local bundle — `scp` a fresh `git bundle` and verify HEAD (per memory `p2-02-merged`).

- [ ] **Step 1: Verify each box is idle (CLAUDE.md gate)**

On each box: `uptime` (load ≈ 0) and `pgrep -af "cargo bench|bencher run|Runner.Worker"` (empty). Do NOT measure otherwise — CI-race contamination understated P2-01 by ~2× (memory `feedback-check-server-clean`).

- [ ] **Step 2: Build release + run the driver on each box**

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release -p aleph-sv --features internal-bench
ALEPH_CPU_MODEL=epyc  ./scripts/tune-chunks.sh 2>&1 | tee tune-epyc.log     # on EPYC
ALEPH_CPU_MODEL=ryzen ./scripts/tune-chunks.sh 2>&1 | tee tune-ryzen.log    # on Ryzen
```

- [ ] **Step 3: Pick best-per-cell**

For each (gate→GateClass, target→PosClass) pick the `(min_amps, grain)` with the lowest median. Map `target 1→Low, 12→Mid, 24→High`; `h→OneQGeneric, zdiag→OneQDiag, cnot→TwoQCnot, cphase→TwoQDiag`. Record the winning `ChunkPolicy` per cell per CPU. If a cell's best is within noise of `DEFAULT_POLICY`, leave it at default (don't encode noise).

- [ ] **Step 4: Populate `chunk_policy` in `tuning.rs`**

Replace the `Epyc8124P`/`Ryzen3900` arms with a `(class, pos)` match using the measured winners, falling through to `DEFAULT_POLICY` for untuned cells. Example shape (values are placeholders — use the sweep's actual winners):

```rust
RefCpu::Epyc8124P => match (_class, _pos) {
    (GateClass::TwoQDiag, PosClass::Mid)  => ChunkPolicy { min_amps: 1 << 17, grain: 128 },
    (GateClass::TwoQCnot, PosClass::High) => ChunkPolicy { min_amps: 1 << 16, grain: 256 },
    // ... other measured winners ...
    _ => DEFAULT_POLICY,
},
RefCpu::Ryzen3900 => match (_class, _pos) {
    // ... measured winners ...
    _ => DEFAULT_POLICY,
},
```

(Rename the `_class`/`_pos` params to `class`/`pos` once they're used.)

- [ ] **Step 5: Update the tuning unit test for populated cells**

Add to `tuning.rs` tests — assert a couple of known populated cells now differ from default and a known untuned cell still equals default:

```rust
#[test]
fn populated_cells_override_default() {
    // Use the actual cells you populated in Step 4.
    assert_ne!(chunk_policy(RefCpu::Epyc8124P, GateClass::TwoQDiag, PosClass::Mid), DEFAULT_POLICY);
    assert_eq!(chunk_policy(RefCpu::Epyc8124P, GateClass::ThreeQ, PosClass::Low), DEFAULT_POLICY);
}
```

- [ ] **Step 6: Test + commit**

Run: `cargo test -p aleph-sv tuning && cargo test -p aleph-sv --features internal-bench --test policy_invariance`
Expected: PASS (invariance still holds — populated cells are just different valid policies).

```bash
git add crates/aleph-sv/src/kernels/tuning.rs tune-epyc.log tune-ryzen.log
git commit -m "[P2-04] Populate chunk table from EPYC+Ryzen sweep"
```

---

## Task 6: Improvement benchmark + perf report

**Files:**
- Create: `docs/perf/phase2-p2-04.md`

- [ ] **Step 1: Measure tuned-vs-default on the reference box**

On the box with the cleaner/larger sweep signal (designated primary), run the Tier-1 workloads at n=25 twice — once with the auto-detected table, once forcing the default — and compare:

```bash
# tuned (auto CPU detect)
RUSTFLAGS="-C target-cpu=native" cargo bench -p aleph-benches --bench qft_scaling -- --save-baseline tuned
# default (force Generic == legacy constants)
ALEPH_CPU_MODEL=generic RUSTFLAGS="-C target-cpu=native" cargo bench -p aleph-benches --bench qft_scaling -- --baseline tuned
```

(Workspace bench crate is `aleph-benches`. Available workload benches: `qft_scaling`, `qft`, `ghz`, `random`, `bell`. Use `qft_scaling` + `random` at minimum. Confirm idle first.)

- [ ] **Step 2: Write the report**

Create `docs/perf/phase2-p2-04.md` with: the sweep grid + winning cells per CPU, the tuned-vs-default Tier-1 numbers (honest — flat is a valid result per P2-02/03 precedent), which box is the **designated primary reference CPU** and why, and a note that the other ~22 cells stay at default (YAGNI). Cross-reference the spec.

- [ ] **Step 3: Commit**

```bash
git add docs/perf/phase2-p2-04.md
git commit -m "[P2-04] Phase 2 chunk-tuning perf report (honest tuned-vs-default)"
```

---

## Task 7: PR

- [ ] **Step 1: Final gates**

Run: `cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --check`
Expected: all green.

- [ ] **Step 2: Push + open PR**

```bash
git push -u origin p2-04-chunk-tuning
gh pr create --title "[P2-04] Chunked parallelism tuning" --body "$(cat <<'EOF'
Closes #30

## Approach
Per-(gate, qubit-position) `ChunkPolicy { min_amps, grain }` selected by CPU
model at runtime (`kernels::tuning`), computed leaf-locally and passed
explicitly into `par_blocks`/`par_units` (Approach A — no hidden state).
`RefCpu::Generic` and every untuned cell return the pre-P2-04 defaults, so
unknown hardware is byte-for-byte unchanged. Env `ALEPH_PAR_MIN_AMPS` /
`ALEPH_PAR_GRAIN` override per-field and drove the sweep.

## Tests
- `policy_invariance`: any ChunkPolicy ⇒ bit-identical amplitudes (exact).
- `par_tests`: exhaustive+disjoint coverage across grains {1,16,64,1024}.
- `tuning`: Generic==default, pos_class boundaries, CPU detect (env + brand).
- Full SV oracle suite green under default policy.

## Benchmark
<tuned-vs-default Tier-1 numbers from docs/perf/phase2-p2-04.md; honest,
flat-is-valid per P2-02/03 precedent>. Primary reference CPU: <EPYC|Ryzen>.

## Notes / follow-ups
- Runtime auto-tune deferred (BACKLOG: "start with table").
- ~22 low-traffic cells left at default (YAGNI).
- LOW_BAND/HIGH_BAND position thresholds are fixed design constants.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Self-review the diff**

Run: `git diff origin/main...HEAD` and re-read with fresh eyes (CLAUDE.md PR Workflow). Confirm `Closes #30` (issue number, not PR), benchmark numbers filled in, no leftover placeholders.

---

## Self-Review (plan vs spec)

- **Spec §Mechanism (leaf-local A)** → Task 2 (recipe + mapping table). ✓
- **Spec §CPU detection** → Task 1 (`detect_cpu_from`, `cpu_brand_string`, env override). ✓
- **Spec §Types & taxonomy** → Task 1 (`ChunkPolicy`/`GateClass`/`PosClass`/`pos_class`). ✓
- **Spec §Table + no-regression** → Task 1 (Generic==default) + Task 5 (populate). ✓
- **Spec §Policy resolution & precedence (per-field, OnceLock-cached)** → Task 1 (`resolve_policy`, `env_min_amps`/`env_grain`). ✓
- **Spec §Sweep harness** → Task 4 (bench + driver) + Task 5 (run). ✓
- **Spec §Correctness & testing** → Task 1 (unit), Task 2 (par_tests across grains), Task 3 (invariance). ✓
- **Spec §AC: tuned table for one reference CPU** → Task 5 (+ designated primary in Task 6). ✓
- **Spec §AC: benchmark improvement over default** → Task 6. ✓
- **Spec §Out of scope** (auto-tune, threshold tuning, 22 cells, NUMA×chunk) → respected; PR notes call them out. ✓

Type consistency: `ChunkPolicy`, `GateClass`, `PosClass`, `RefCpu`, `resolve_policy`, `pos_class`, `chunk_policy`, `DEFAULT_POLICY`, `test_override::{set,get}` used consistently across Tasks 1–6. `par_blocks`/`par_units` take `policy` as the first arg everywhere.
