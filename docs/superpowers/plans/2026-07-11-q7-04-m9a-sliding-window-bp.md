# Q7-04 M9a — Software golden `SlidingWindowBp` + (W,C)×seam×p LER sweep

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The software golden for streaming relay-BP: a residual-carry sliding-window decoder over
multi-round circuit-level BB DEMs with `FixedRelayBp` as the per-window base decoder, plus the LER
sweep that picks the (W, C, seam) configuration M9b bakes into RTL. First of three staged PRs of
issue #455; delivers AC-1.

**Architecture:** `SlidingWindowBp` mirrors the surface-code `SlidingWindowDecoder` (`sliding.rs`)
with two BP-specific deltas (spec § 3): straddling error mechanisms are **truncated** at the
temporal cut (no sink nodes — BP handles hypergraphs), and commit operates on **error-vars**
(earliest in-window detector round < commit boundary). Windows are precompiled at construction
(translation invariance makes interior windows identical — the property M9b's one-baked-header
depends on). Seam state: residual-only, or residual + soft priors (previous window's posterior
LLRs seed shared uncommitted vars).

**Tech Stack:** Rust (aleph-qec crate), rayon (already a dep), criterion not needed. Spec:
`docs/superpowers/specs/2026-07-11-q7-04-streaming-relay-bp-design.md`.

## Global Constraints

- CLAUDE.md: no `unwrap()`/`expect()` in library code (tests OK); `cargo clippy --workspace
  --all-targets -- -D warnings` and `cargo fmt --check` must pass; comments explain WHY; cite
  papers next to algorithm code.
- Frozen operating point (must match RTL/M8 golden): `MSG_BITS=8, FRAC_BITS=3` (Q5.3),
  `LEGS=6, ITERS=10, GAMMA=(-0.3, 0.9), SEED=0x5E1A_4B9C`.
- Bit-exactness anchor: with `W = num_slices` (one window, commit-all) `SlidingWindowBp` must
  reproduce the batch `FixedRelayBp` decode **exactly** — same vars, same order, same disorder.
- Branch: `q7-04-m9a-sliding-window-bp` off local `main` (contains the spec commit `2e49982`).
  PR title `[Q7-04] M9a: …`. PR body references issue #455 but must **NOT** say `Closes #455`
  (two more stages follow) — write "Part 1 of 3 for #455".
- No git worktrees; work directly in `/Users/ex/GitHub/aleph`.

## File Structure

- Modify `crates/aleph-qec/src/fixed_bp.rs` — two small hooks: `lambda_q()` getter,
  `with_lambda_q()` prior override (for the soft-priors seam).
- Create `crates/aleph-qec/src/relay_window.rs` — `SlidingWindowBp`, `SeamMode`,
  `WindowBpExport`, `StreamStats` + inline unit tests.
- Modify `crates/aleph-qec/src/lib.rs` — register module + re-exports.
- Create `crates/aleph-qec/examples/qec_q7_stream_sweep.rs` — the sweep binary.
- Modify `docs/perf/qec-q7-fixed-bp.md` — new M9a section (after the EPYC run).
- Modify `docs/qec/BACKLOG.md` — tick Q7-04 AC-1.

---

### Task 1: Branch + `FixedRelayBp` prior-override hooks

**Files:**
- Modify: `crates/aleph-qec/src/fixed_bp.rs` (after `with_early_exit`, ~line 194)

**Interfaces:**
- Produces: `FixedRelayBp::lambda_q(&self) -> &[i32]`,
  `FixedRelayBp::with_lambda_q(self, Vec<i32>) -> Self` — consumed by Task 4 (soft-priors seam).

- [ ] **Step 1: Create the branch**

```bash
cd /Users/ex/GitHub/aleph
git checkout -b q7-04-m9a-sliding-window-bp main
```

- [ ] **Step 2: Write the failing test** (inside `mod tests` at the bottom of `fixed_bp.rs`; the
  file already has a test module — add to it)

```rust
    /// Q7-04: the sliding-window soft-priors seam re-seeds per-variable priors. Overriding λ_q
    /// must change the decode inputs; an identity override must change nothing.
    #[test]
    fn lambda_q_override_roundtrip() {
        let dem = DetectorErrorModel::parse("error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1\n")
            .unwrap();
        let dec = FixedRelayBp::new(&dem, 8, 3);
        let lam = dec.lambda_q().to_vec();
        assert_eq!(lam.len(), 3, "one prior per variable");
        // Identity override: decode of a fixed syndrome is unchanged.
        let syn = Syndrome::new(2, vec![0]);
        let base = dec.decode_fixed(&syn);
        let same = dec.clone().with_lambda_q(lam.clone()).decode_fixed(&syn);
        assert_eq!(base, same);
        // A hostile override (all strongly "fired") flips the hard decision inputs.
        let forced = dec.clone().with_lambda_q(vec![-127; 3]);
        assert_eq!(forced.lambda_q(), &[-127, -127, -127]);
    }
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cargo test -p aleph-qec lambda_q_override_roundtrip`
Expected: FAIL — `lambda_q`/`with_lambda_q` not found.

- [ ] **Step 4: Implement the two methods** (in `impl FixedRelayBp`, right after
  `with_early_exit`)

```rust
    /// The quantised per-variable prior LLRs `λ_q` (build-time quantisation of `ln((1−p)/p)`).
    pub fn lambda_q(&self) -> &[i32] {
        &self.lambda_q
    }

    /// Replace the per-variable quantised priors. The sliding-window soft-priors seam (Q7-04)
    /// re-decodes buffer-round variables in the next window seeded with the previous window's
    /// posterior LLR; this is the injection point. Values are used as-is — the caller clamps to
    /// the message magnitude range.
    ///
    /// # Panics
    /// If `lambda_q.len()` differs from the number of variables.
    pub fn with_lambda_q(mut self, lambda_q: Vec<i32>) -> Self {
        assert_eq!(lambda_q.len(), self.n_vars, "one prior per variable");
        self.lambda_q = lambda_q;
        self
    }
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cargo test -p aleph-qec lambda_q_override_roundtrip`
Expected: PASS. Also run `cargo test -p aleph-qec --lib` — all pre-existing tests still green.

- [ ] **Step 6: Commit**

```bash
cargo fmt
git add crates/aleph-qec/src/fixed_bp.rs
git commit -m "[Q7-04] fixed_bp: lambda_q getter + with_lambda_q prior override

Injection point for the M9a sliding-window soft-priors seam: the next
window re-decodes buffer-round variables seeded with the previous
window's posterior LLRs."
```

---

### Task 2: `relay_window.rs` — window compilation (truncation + translation invariance)

**Files:**
- Create: `crates/aleph-qec/src/relay_window.rs`
- Modify: `crates/aleph-qec/src/lib.rs` (module + re-export)

**Interfaces:**
- Consumes: `DetectorErrorModel`/`DemError` (dem.rs), `FixedRelayBp::with_budget` (fixed_bp.rs).
- Produces: `SlidingWindowBp::new(dem, detector_round, window, commit) -> Self`,
  `SlidingWindowBp::with_seam(self, SeamMode) -> Self`,
  `SlidingWindowBp::window_dem(&self, s, win_hi) -> WindowBpExport`,
  `SlidingWindowBp::max_window_detectors(&self) -> usize`,
  `WindowBpExport { dem, globals: Vec<usize>, mech_globals: Vec<usize> }`,
  `SeamMode::{ResidualOnly, SoftPriors}` — consumed by Tasks 3–5.

- [ ] **Step 1: Write the module with construction + `window_dem` and two failing tests.**
  Create `crates/aleph-qec/src/relay_window.rs`:

```rust
//! [`SlidingWindowBp`] — sliding-window streaming decoding of multi-round circuit-level BB DEMs
//! with the fixed-point relay-BP as the per-window base decoder (Q7-04, M9a software golden).
//!
//! The window/commit schedule is the surface-code [`SlidingWindowDecoder`](crate::sliding)
//! (Skoric et al., arXiv:2209.09219; Tan et al., arXiv:2209.08552) with two BP-specific deltas
//! (design spec `docs/superpowers/specs/2026-07-11-q7-04-streaming-relay-bp-design.md` § 3):
//!
//! 1. **Hypergraph time cut.** A DEM error mechanism flips an arbitrary detector set, so a
//!    mechanism straddling the window edge is **truncated** to its in-window detectors (open
//!    temporal boundary, standard for sliding-window BP/LDPC decoding) instead of routed to a
//!    temporal-sink node — matching needs sink *nodes*; BP does not.
//! 2. **Commit on error-vars.** A variable whose earliest in-window detector round lies in the
//!    commit region is committed when its hard decision fires: its observable mask XORs into the
//!    running logical, its in-window detectors toggle the residual. The trailing `W − C` buffer
//!    rounds stay lit and are re-decoded by the next window with fresh future context.
//!
//! Interior windows are translation-invariant — they compile to the *identical* local DEM — which
//! is what lets M9b bake ONE window graph header into the RTL. A test pins this property.
//!
//! Seam state across windows is selectable ([`SeamMode`]): residual-only (the UF-streaming
//! discipline, bounded binary state) or residual + **soft priors** (the previous window's
//! posterior LLRs seed the shared uncommitted variables of the next window). The M9a sweep
//! (`examples/qec_q7_stream_sweep.rs`) decides which ships to RTL.

use std::collections::HashMap;

use crate::dem::{DemError, DetectorErrorModel};
use crate::fixed_bp::FixedRelayBp;
use crate::syndrome::{Correction, Syndrome};

/// The frozen M5/M8 relay-BP operating point the RTL implements (`docs/perf/qec-q7-fixed-bp.md`).
/// The golden must decode with exactly these parameters to stay bit-comparable to silicon.
const MSG_BITS: u32 = 8;
const FRAC_BITS: u32 = 3;
const LEGS: usize = 6;
const ITERS_PER_LEG: u32 = 10;
const GAMMA: (f64, f64) = (-0.3, 0.9);
const SEED: u64 = 0x5E1A_4B9C;

/// What carries across the window seam besides the committed corrections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeamMode {
    /// Only the binary residual syndrome (the UF-streaming discipline). Each window is a fresh
    /// relay-BP decode from the DEM priors.
    ResidualOnly,
    /// Residual + soft priors: shared uncommitted variables of the next window start from the
    /// previous window's posterior LLRs (clamped to the message range) instead of the DEM priors.
    SoftPriors,
}

/// One window's compiled BP problem, exported for reuse (software decode now; the M9b RTL
/// window-graph emitter later — the single source of truth, like `sliding::WindowExport`).
#[derive(Clone, Debug)]
pub struct WindowBpExport {
    /// The window DEM: detectors are the in-window ones re-indexed `0..globals.len()`; variables
    /// are the stream mechanisms with in-window support, detector sets truncated at the cut.
    pub dem: DetectorErrorModel,
    /// `globals[l]` = global detector id of local detector `l`.
    pub globals: Vec<usize>,
    /// `mech_globals[v]` = index into the stream DEM's `errors` of local variable `v`.
    pub mech_globals: Vec<usize>,
}

/// Per-stream decode statistics (feeds the Q7-07 non-convergence study).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StreamStats {
    /// Number of windows decoded.
    pub windows: usize,
    /// Windows whose relay-BP found no syndrome-valid decision (committed best-effort + flagged).
    pub nonconverged: usize,
    /// Detectors still lit after the final window. When every window converged this must be 0
    /// (each lit commit-region detector is covered by an odd number of fired mechanisms, all of
    /// which touch the commit region and therefore commit and toggle it).
    pub residual: usize,
}

/// A precompiled window: its span, DEM, decoder, and commit/seam metadata.
///
/// Windows are compiled once at construction — the stream length is known up front here, and
/// interior windows are translation-invariant anyway (the RTL relies on exactly that to bake a
/// single window graph). Memory is `O(num_slices/C)` window slots of `O(W)` size each.
#[derive(Clone, Debug)]
struct WindowSlot {
    s: usize,
    win_hi: usize,
    export: WindowBpExport,
    decoder: FixedRelayBp,
    /// `commit_var[v]`: variable `v` has an in-window detector with round `< commit_hi`.
    commit_var: Vec<bool>,
    /// `(prev_local, this_local)` pairs of mechanisms shared with the PREVIOUS window slot —
    /// the soft-priors injection map. Empty for the first window.
    prev_var_map: Vec<(usize, usize)>,
}

/// Sliding-window streaming relay-BP over a multi-round DEM with per-detector round coordinates.
#[derive(Clone, Debug)]
pub struct SlidingWindowBp {
    dem: DetectorErrorModel,
    detector_round: Vec<usize>,
    num_slices: usize,
    window: usize,
    commit: usize,
    seam: SeamMode,
    slots: Vec<WindowSlot>,
    /// `2^(MSG_BITS−1) − 1`; soft priors clamp here before injection.
    max_mag: i32,
}

impl SlidingWindowBp {
    /// Build the streaming decoder: window length `window` (W rounds), commit region `commit`
    /// (C rounds), over `dem` with `detector_round[d]` the time coordinate of detector `d`
    /// (from [`BBMemoryExperiment::detector_rounds`](crate::BBMemoryExperiment::detector_rounds)).
    /// Uses the frozen M8 operating point (Q5.3, 6×10 schedule); seam defaults to
    /// [`SeamMode::ResidualOnly`].
    ///
    /// # Panics
    /// If `detector_round.len() != dem.detectors` or `commit` is not in `1..=window`.
    pub fn new(
        dem: DetectorErrorModel,
        detector_round: Vec<usize>,
        window: usize,
        commit: usize,
    ) -> Self {
        assert_eq!(
            detector_round.len(),
            dem.detectors,
            "need one round per detector"
        );
        assert!(
            window >= 1 && (1..=window).contains(&commit),
            "need 1 <= commit <= window"
        );
        let num_slices = detector_round
            .iter()
            .copied()
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);

        let mut slots: Vec<WindowSlot> = Vec::new();
        let mut s = 0usize;
        while s < num_slices {
            let win_hi = (s + window).min(num_slices);
            let last = win_hi >= num_slices;
            let commit_hi = if last { num_slices } else { s + commit };

            let export = compile_window(&dem, &detector_round, s, win_hi);
            let decoder = FixedRelayBp::with_budget(
                &export.dem,
                LEGS,
                ITERS_PER_LEG,
                GAMMA,
                SEED,
                MSG_BITS,
                FRAC_BITS,
            );
            let commit_var = export
                .mech_globals
                .iter()
                .map(|&g| {
                    dem.errors[g].dets.iter().any(|&d| {
                        let r = detector_round[d as usize];
                        r >= s && r < commit_hi
                    })
                })
                .collect();
            // Soft-priors map vs the previous slot: shared global mechanism -> both local ids.
            let prev_var_map = match slots.last() {
                None => Vec::new(),
                Some(prev) => {
                    let this_local: HashMap<usize, usize> = export
                        .mech_globals
                        .iter()
                        .enumerate()
                        .map(|(l, &g)| (g, l))
                        .collect();
                    prev.export
                        .mech_globals
                        .iter()
                        .enumerate()
                        .filter_map(|(pl, g)| this_local.get(g).map(|&tl| (pl, tl)))
                        .collect()
                }
            };

            slots.push(WindowSlot {
                s,
                win_hi,
                export,
                decoder,
                commit_var,
                prev_var_map,
            });
            s += commit;
        }

        Self {
            dem,
            detector_round,
            num_slices,
            window,
            commit,
            seam: SeamMode::ResidualOnly,
            slots,
            max_mag: (1i32 << (MSG_BITS - 1)) - 1,
        }
    }

    /// Select the seam state (default [`SeamMode::ResidualOnly`]).
    pub fn with_seam(mut self, seam: SeamMode) -> Self {
        self.seam = seam;
        self
    }

    /// Window length `W`.
    pub fn window(&self) -> usize {
        self.window
    }

    /// Commit-region length `C`.
    pub fn commit(&self) -> usize {
        self.commit
    }

    /// Compile the window covering rounds `[s, win_hi)` — public because it is the single source
    /// of truth the M9b RTL window-graph emitter will consume (mirrors `sliding::window_dem`).
    pub fn window_dem(&self, s: usize, win_hi: usize) -> WindowBpExport {
        compile_window(&self.dem, &self.detector_round, s, win_hi)
    }

    /// The largest number of detectors any single window spans — bounds the per-window working
    /// set independent of stream length (mirrors `sliding::max_window_detectors`).
    pub fn max_window_detectors(&self) -> usize {
        self.slots
            .iter()
            .map(|w| w.export.globals.len())
            .max()
            .unwrap_or(0)
    }
}

/// Build one window's DEM: in-window detectors re-indexed `0..n_active` in ascending global
/// order; mechanisms with any in-window support kept **in stream-DEM order** with their detector
/// sets truncated to the window (probability unchanged — an open temporal boundary). Mechanisms
/// are NOT merged after truncation: the 1:1 `mech_globals` map is what the commit path and the
/// soft-priors seam key on, and the order preservation is what makes the one-window case
/// bit-identical to the batch decode (same vars, same γ disorder indexing).
fn compile_window(
    dem: &DetectorErrorModel,
    detector_round: &[usize],
    s: usize,
    win_hi: usize,
) -> WindowBpExport {
    let mut local_of = vec![u32::MAX; dem.detectors];
    let mut globals: Vec<usize> = Vec::new();
    for (d, &r) in detector_round.iter().enumerate() {
        if r >= s && r < win_hi {
            local_of[d] = globals.len() as u32;
            globals.push(d);
        }
    }

    let mut errors: Vec<DemError> = Vec::new();
    let mut mech_globals: Vec<usize> = Vec::new();
    for (g, e) in dem.errors.iter().enumerate() {
        let loc: Vec<u32> = e
            .dets
            .iter()
            .filter_map(|&d| {
                let l = local_of[d as usize];
                (l != u32::MAX).then_some(l)
            })
            .collect();
        if loc.is_empty() {
            continue; // no in-window support
        }
        errors.push(DemError::new(e.prob, loc, e.obs.clone()));
        mech_globals.push(g);
    }

    WindowBpExport {
        dem: DetectorErrorModel {
            detectors: globals.len(),
            observables: dem.observables,
            errors,
        },
        globals,
        mech_globals,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::CircuitNoise;
    use crate::bivariate_bicycle::BBCode;

    /// Truncation at the temporal cut: a straddling mechanism keeps its probability and its
    /// observables but loses its out-of-window detectors; fully-outside mechanisms vanish.
    #[test]
    fn truncates_straddling_mechanisms() {
        let dem = DetectorErrorModel {
            detectors: 3,
            observables: 1,
            errors: vec![
                DemError::new(0.1, vec![0], vec![]),     // round 0 only
                DemError::new(0.2, vec![1, 2], vec![0]), // straddles rounds 1..2
                DemError::new(0.3, vec![2], vec![]),     // round 2 only
            ],
        };
        let sw = SlidingWindowBp::new(dem, vec![0, 1, 2], 2, 1);
        let w = sw.window_dem(0, 2); // covers rounds [0, 2): global detectors 0, 1
        assert_eq!(w.globals, vec![0, 1]);
        assert_eq!(w.mech_globals, vec![0, 1], "mechanism 2 has no in-window support");
        assert_eq!(w.dem.detectors, 2);
        assert_eq!(w.dem.errors[0], DemError::new(0.1, vec![0], vec![]));
        assert_eq!(
            w.dem.errors[1],
            DemError::new(0.2, vec![1], vec![0]),
            "detector 2 truncated at the cut; probability and observable kept"
        );
    }

    /// Interior windows compile to the IDENTICAL local DEM — the translation invariance the M9b
    /// one-baked-header RTL depends on. (Head windows see prep boundary effects, tail windows the
    /// final readout; interior ones must all match.)
    #[test]
    fn interior_windows_are_translation_invariant() {
        let code = BBCode::gross();
        let rounds = 8;
        let dem = code
            .circuit_level_dem(rounds, CircuitNoise::uniform(0.003))
            .unwrap();
        let dr = code.memory_x_experiment(rounds).detector_rounds();
        let sw = SlidingWindowBp::new(dem, dr, 3, 1);
        let a = sw.window_dem(2, 5);
        let b = sw.window_dem(3, 6);
        assert_eq!(a.globals.len(), b.globals.len());
        assert_eq!(
            a.dem, b.dem,
            "interior windows must compile to the identical local DEM"
        );
    }
}
```

- [ ] **Step 2: Register the module in `crates/aleph-qec/src/lib.rs`.** Add alongside the
  existing `pub mod sliding;` declaration (find the module list) and the re-export block:

```rust
pub mod relay_window;
```

and with the other `pub use` lines (keep alphabetical placement near `pub use relay_bp::…`):

```rust
pub use relay_window::{SeamMode, SlidingWindowBp, StreamStats, WindowBpExport};
```

- [ ] **Step 3: Run the tests**

Run: `cargo test -p aleph-qec relay_window`
Expected: `truncates_straddling_mechanisms` PASS.
`interior_windows_are_translation_invariant` — expected PASS; **if it fails on mechanism
order/probs**, the stream DEM's mechanism enumeration is not translation-consistent: fix by
canonically sorting each window's mechanisms by `(local dets, obs, prob.to_bits())` inside
`compile_window` *and* update the Task 3 batch-equality test to compare corrections through the
same sort (document the sort in the module docs). Do NOT silently weaken the assertion.

- [ ] **Step 4: Commit**

```bash
cargo fmt
git add crates/aleph-qec/src/relay_window.rs crates/aleph-qec/src/lib.rs
git commit -m "[Q7-04] relay_window: window compilation for streaming relay-BP

Truncating hypergraph time cut + translation-invariance pin (the
property the M9b one-baked-header RTL depends on)."
```

---

### Task 3: `decode_stream` (residual-only) + `Decoder` impl + stats

**Files:**
- Modify: `crates/aleph-qec/src/relay_window.rs`

**Interfaces:**
- Consumes: `FixedRelayBp::decode_fixed_soft -> BpSoft {ehat: Vec<u8>, llr: Vec<f64>, converged}`.
- Produces: `SlidingWindowBp::decode_stream(&self, &Syndrome) -> (Correction, StreamStats)`;
  `impl Decoder for SlidingWindowBp` (used by the sweep and any harness).

- [ ] **Step 1: Add the failing tests** to `mod tests` in `relay_window.rs`:

```rust
    use crate::experiment::sample_shots;
    use crate::fixed_bp::FixedRelayBp;

    /// With one window covering the whole stream (W = num_slices, commit-all), the sliding
    /// decoder IS the batch decode: same vars in the same order ⇒ same γ disorder ⇒ bit-exact.
    #[test]
    fn full_window_equals_batch() {
        let code = BBCode::gross();
        let rounds = 3;
        let dem = code
            .circuit_level_dem(rounds, CircuitNoise::uniform(0.003))
            .unwrap();
        let dr = code.memory_x_experiment(rounds).detector_rounds();
        let ns = rounds + 1; // detector rounds run 0..=rounds
        let sw = SlidingWindowBp::new(dem.clone(), dr, ns, ns);
        // One window, and it must keep every mechanism (all BB circuit mechanisms detect).
        assert_eq!(sw.window_dem(0, ns).mech_globals.len(), dem.errors.len());
        let batch = FixedRelayBp::with_budget(&dem, 6, 10, (-0.3, 0.9), 0x5E1A_4B9C, 8, 3);

        let (syndromes, _) = sample_shots(&dem, 50, 7);
        for syn in &syndromes {
            let (corr, stats) = sw.decode_stream(syn);
            let (want, _) = batch.decode_fixed(syn);
            assert_eq!(corr, want, "one-window sliding decode must be bit-exact to batch");
            assert_eq!(stats.windows, 1);
        }
    }

    /// When every window converges, the committed correction must clear every real detector —
    /// the residual drains to zero (the streaming validity property).
    #[test]
    fn converged_stream_drains_residual() {
        let code = BBCode::gross();
        let rounds = 8;
        let dem = code
            .circuit_level_dem(rounds, CircuitNoise::uniform(0.002))
            .unwrap();
        let dr = code.memory_x_experiment(rounds).detector_rounds();
        let sw = SlidingWindowBp::new(dem.clone(), dr, 3, 1);

        let (syndromes, _) = sample_shots(&dem, 30, 11);
        let mut converged_shots = 0;
        for syn in &syndromes {
            let (_, stats) = sw.decode_stream(syn);
            if stats.nonconverged == 0 {
                converged_shots += 1;
                assert_eq!(
                    stats.residual, 0,
                    "all windows converged but the residual did not drain"
                );
            }
        }
        assert!(converged_shots > 0, "expected some fully-converged shots at p=0.002");
    }

    /// The per-window working set is bounded by W, independent of stream length.
    #[test]
    fn working_set_is_bounded() {
        let code = BBCode::gross();
        let mk = |rounds: usize| {
            let dem = code
                .circuit_level_dem(rounds, CircuitNoise::uniform(0.003))
                .unwrap();
            let dr = code.memory_x_experiment(rounds).detector_rounds();
            SlidingWindowBp::new(dem, dr, 3, 1)
        };
        assert_eq!(mk(6).max_window_detectors(), mk(12).max_window_detectors());
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p aleph-qec relay_window`
Expected: FAIL — `decode_stream` not found.

- [ ] **Step 3: Implement `decode_stream` + `Decoder`** (in `impl SlidingWindowBp`, plus a
  trailing trait impl):

```rust
    /// Decode an entire stream syndrome by sliding the window across the rounds, committing each
    /// window's commit-region variables. Returns the committed logical correction and the
    /// per-stream statistics (non-convergence feeds Q7-07; `residual` is the validity probe).
    ///
    /// A window that finds no syndrome-valid decision still commits its best-kept decision —
    /// report-and-flag (spec § 4-M9a); the flag lands in [`StreamStats::nonconverged`].
    pub fn decode_stream(&self, syndrome: &Syndrome) -> (Correction, StreamStats) {
        let nd = self.dem.detectors;
        let mut lit = vec![false; nd];
        for &d in &syndrome.fired {
            if (d as usize) < nd {
                lit[d as usize] = true;
            }
        }

        let mut logical = vec![false; self.dem.observables];
        let mut nonconverged = 0usize;
        // Soft-priors carry: previous window's (posterior λ_q by prev-local var, committed).
        let mut carry: Option<(Vec<i32>, Vec<bool>)> = None;

        for slot in &self.slots {
            let fired: Vec<u32> = (0..slot.export.globals.len() as u32)
                .filter(|&l| lit[slot.export.globals[l as usize]])
                .collect();
            let win_syn = Syndrome::new(slot.export.dem.detectors, fired);

            let soft = match (self.seam, &carry) {
                (SeamMode::SoftPriors, Some((post, committed)))
                    if !slot.prev_var_map.is_empty() =>
                {
                    let mut lam = slot.decoder.lambda_q().to_vec();
                    for &(pl, tl) in &slot.prev_var_map {
                        // Committed mechanisms are already accounted for (obs applied, residual
                        // toggled) — they restart from the DEM prior, not the stale posterior.
                        if !committed[pl] {
                            lam[tl] = post[pl].clamp(-self.max_mag, self.max_mag);
                        }
                    }
                    slot.decoder
                        .clone()
                        .with_lambda_q(lam)
                        .decode_fixed_soft(&win_syn)
                }
                _ => slot.decoder.decode_fixed_soft(&win_syn),
            };
            if !soft.converged {
                nonconverged += 1;
            }

            // Commit: fired vars touching the commit region. XOR obs into the logical, toggle
            // the mechanism's in-window detectors in the residual (out-of-window detectors were
            // truncated from the decode and stay for the next window to explain).
            let mut committed = vec![false; soft.ehat.len()];
            for (v, (&e, &cv)) in soft.ehat.iter().zip(&slot.commit_var).enumerate() {
                if e == 1 && cv {
                    committed[v] = true;
                    let g = slot.export.mech_globals[v];
                    for &o in &self.dem.errors[g].obs {
                        logical[o as usize] ^= true;
                    }
                    for &d in &self.dem.errors[g].dets {
                        let r = self.detector_round[d as usize];
                        if r >= slot.s && r < slot.win_hi {
                            lit[d as usize] ^= true;
                        }
                    }
                }
            }
            // The posterior llr is the quantised i32 the fixed decoder computed, round-tripped
            // through f64 losslessly (magnitudes ≪ 2^53).
            carry = Some((
                soft.llr.iter().map(|&x| x as i32).collect(),
                committed,
            ));
        }

        let residual = lit.iter().filter(|&&x| x).count();
        (
            Correction::new(logical),
            StreamStats {
                windows: self.slots.len(),
                nonconverged,
                residual,
            },
        )
    }
```

and at the bottom of the file (before `mod tests`):

```rust
impl crate::decoder::Decoder for SlidingWindowBp {
    /// Decode a full-stream syndrome via sliding windows (stats dropped; use
    /// [`SlidingWindowBp::decode_stream`] to keep them).
    fn decode(&self, syndrome: &Syndrome) -> Correction {
        self.decode_stream(syndrome).0
    }

    /// Shots are independent; the sweep decodes hundreds of thousands of streams, so the batch
    /// path is rayon-parallel (mirrors the sampler's determinism: output order is input order).
    fn decode_batch(&self, syndromes: &[Syndrome]) -> crate::error::Result<Vec<Correction>> {
        use rayon::prelude::*;
        Ok(syndromes.par_iter().map(|s| self.decode(s)).collect())
    }
}
```

- [ ] **Step 4: Run the tests**

Run: `cargo test -p aleph-qec relay_window`
Expected: all PASS. `full_window_equals_batch` failing on inequality means var order or disorder
indexing diverged — check that `compile_window` kept stream order and that no mechanism was
dropped (the `mech_globals.len()` assert isolates which).

- [ ] **Step 5: Commit**

```bash
cargo fmt
git add crates/aleph-qec/src/relay_window.rs
git commit -m "[Q7-04] relay_window: residual-carry decode_stream + Decoder impl

Commit-on-vars rule (earliest in-window detector < commit boundary);
one-window case pinned bit-exact to the batch FixedRelayBp; converged
streams drain the residual; report-and-flag on non-convergence."
```

---

### Task 4: Soft-priors seam

**Files:**
- Modify: `crates/aleph-qec/src/relay_window.rs` (tests only — the mechanism landed in Task 3's
  `decode_stream` + Task 1's `with_lambda_q`; this task proves it)

**Interfaces:**
- Consumes: `SeamMode::SoftPriors`, `decode_stream`.

- [ ] **Step 1: Add the failing tests** to `mod tests`:

```rust
    /// One window ⇒ no seam ⇒ SoftPriors must be identical to ResidualOnly (and to batch).
    #[test]
    fn soft_priors_single_window_matches_residual_only() {
        let code = BBCode::gross();
        let rounds = 3;
        let dem = code
            .circuit_level_dem(rounds, CircuitNoise::uniform(0.003))
            .unwrap();
        let dr = code.memory_x_experiment(rounds).detector_rounds();
        let ns = rounds + 1;
        let a = SlidingWindowBp::new(dem.clone(), dr.clone(), ns, ns);
        let b = SlidingWindowBp::new(dem.clone(), dr, ns, ns).with_seam(SeamMode::SoftPriors);
        let (syndromes, _) = sample_shots(&dem, 20, 3);
        for syn in &syndromes {
            assert_eq!(a.decode_stream(syn).0, b.decode_stream(syn).0);
        }
    }

    /// Multi-window SoftPriors decodes, drains on convergence, and is deterministic.
    #[test]
    fn soft_priors_decodes_and_is_deterministic() {
        let code = BBCode::gross();
        let rounds = 8;
        let dem = code
            .circuit_level_dem(rounds, CircuitNoise::uniform(0.002))
            .unwrap();
        let dr = code.memory_x_experiment(rounds).detector_rounds();
        let sw = SlidingWindowBp::new(dem.clone(), dr, 3, 1).with_seam(SeamMode::SoftPriors);
        let (syndromes, _) = sample_shots(&dem, 20, 13);
        for syn in &syndromes {
            let (c1, s1) = sw.decode_stream(syn);
            let (c2, s2) = sw.decode_stream(syn);
            assert_eq!((&c1, &s1), (&c2, &s2), "decode_stream must be a pure function");
            if s1.nonconverged == 0 {
                assert_eq!(s1.residual, 0);
            }
        }
    }
```

- [ ] **Step 2: Run the tests**

Run: `cargo test -p aleph-qec relay_window`
Expected: PASS (the machinery exists). If `soft_priors_single_window_matches_residual_only`
fails, the seam is being applied where no previous window exists — check the
`!slot.prev_var_map.is_empty()` guard and that `carry` starts `None`.

- [ ] **Step 3: Commit**

```bash
cargo fmt
git add crates/aleph-qec/src/relay_window.rs
git commit -m "[Q7-04] relay_window: pin the soft-priors seam

Single-window SoftPriors ≡ ResidualOnly ≡ batch; multi-window decode is
deterministic and drains on convergence."
```

---

### Task 5: Sweep example `qec_q7_stream_sweep.rs`

**Files:**
- Create: `crates/aleph-qec/examples/qec_q7_stream_sweep.rs`

**Interfaces:**
- Consumes: `SlidingWindowBp`, `SeamMode`, `FixedRelayBp`, `sample_shots`, `LogicalErrorResult`.
- Produces: the CSV the M9a doc section is written from.

- [ ] **Step 1: Write the example:**

```rust
//! Q7-04 M9a — the (W, C) × seam × p sliding-window LER sweep that picks the streaming
//! configuration M9b bakes into RTL.
//!
//! For each physical error rate the same Monte-Carlo shots (same seed ⇒ same `sample_shots`
//! stream) are decoded by the batch `FixedRelayBp` (the reference — windowing cost, not sampling
//! noise, is what the comparison isolates) and by every (W, C, seam) sliding-window
//! configuration. Reported per cell: windowed LER ± CI, batch LER ± CI, within-CI flag, the
//! fraction of shots with ≥1 non-converged window (feeds Q7-07), and the fraction with a
//! non-zero final residual.
//!
//! Usage:
//!   cargo run --release -p aleph-qec --example qec_q7_stream_sweep -- [rounds] [shots] [seed]
//!   # defaults: rounds=12 shots=20000 seed=2024
//!
//! Decision rule (spec § 4-M9a): pick the smallest (W, C, seam) whose LER stays within the batch
//! CI at every p (or a documented, explicitly-accepted gap). Soft priors ship only on a clear win.

use aleph_qec::{
    sample_shots, BBCode, CircuitNoise, Correction, FixedRelayBp, LogicalErrorResult, SeamMode,
    SlidingWindowBp,
};
use rayon::prelude::*;

const MSG_BITS: u32 = 8;
const FRAC_BITS: u32 = 3;
const LEGS: usize = 6;
const ITERS: u32 = 10;
const GAMMA: (f64, f64) = (-0.3, 0.9);
const SEED: u64 = 0x5E1A_4B9C;

/// Circuit-level per-cycle rates around the relay-BP threshold (~0.3 %).
const PS: &[f64] = &[0.001, 0.003, 0.005];
/// The (W, C) grid from the design spec § 4-M9a.
const WC: &[(usize, usize)] = &[(3, 1), (4, 2), (6, 2), (6, 3)];

fn mispredicted(pred: &Correction, truth: &[bool], observables: usize) -> bool {
    (0..observables).any(|o| {
        pred.observable_flips.get(o).copied().unwrap_or(false)
            != truth.get(o).copied().unwrap_or(false)
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rounds: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(12);
    let shots: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20_000);
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2024);

    let code = BBCode::gross();
    eprintln!(
        "# gross [[144,12,12]] circuit-level rounds={rounds}, shots={shots}, seed={seed}, \
         schedule={LEGS}x{ITERS}, word=Q{}.{}",
        MSG_BITS - 1 - FRAC_BITS,
        FRAC_BITS
    );
    println!("p,W,C,seam,ler_win,ci_win,ler_batch,ci_batch,within_ci,nonconv_frac,resid_frac");

    for &p in PS {
        let dem = code
            .circuit_level_dem(rounds, CircuitNoise::uniform(p))
            .expect("circuit-level DEM");
        let dr = code.memory_x_experiment(rounds).detector_rounds();
        let (syndromes, truths) = sample_shots(&dem, shots, seed);

        // Batch reference on the same shots.
        let batch = FixedRelayBp::with_budget(&dem, LEGS, ITERS, GAMMA, SEED, MSG_BITS, FRAC_BITS);
        let batch_errs = syndromes
            .par_iter()
            .zip(&truths)
            .filter(|(syn, truth)| {
                mispredicted(&batch.decode_fixed(syn).0, truth, dem.observables)
            })
            .count() as u64;
        let rb = LogicalErrorResult::new(shots, batch_errs);
        eprintln!("p={p}: batch LER {:.3e} ± {:.1e}", rb.rate, rb.ci95);

        for &(w, c) in WC {
            for seam in [SeamMode::ResidualOnly, SeamMode::SoftPriors] {
                let sw = SlidingWindowBp::new(dem.clone(), dr.clone(), w, c).with_seam(seam);
                let results: Vec<_> = syndromes
                    .par_iter()
                    .map(|syn| sw.decode_stream(syn))
                    .collect();
                let errs = results
                    .iter()
                    .zip(&truths)
                    .filter(|((corr, _), truth)| mispredicted(corr, truth, dem.observables))
                    .count() as u64;
                let rw = LogicalErrorResult::new(shots, errs);
                let nonconv = results.iter().filter(|(_, s)| s.nonconverged > 0).count();
                let resid = results.iter().filter(|(_, s)| s.residual > 0).count();
                let within = (rw.rate - rb.rate).abs() <= (rw.ci95 + rb.ci95);
                let seam_name = match seam {
                    SeamMode::ResidualOnly => "residual",
                    SeamMode::SoftPriors => "soft",
                };
                println!(
                    "{p},{w},{c},{seam_name},{:.6},{:.6},{:.6},{:.6},{},{:.4},{:.4}",
                    rw.rate,
                    rw.ci95,
                    rb.rate,
                    rb.ci95,
                    within as u8,
                    nonconv as f64 / shots as f64,
                    resid as f64 / shots as f64
                );
                eprintln!(
                    "p={p} W={w} C={c} {seam_name}: LER {:.3e} ± {:.1e} {} | nonconv {:.2}% | resid>0 {:.2}%",
                    rw.rate,
                    rw.ci95,
                    if within { "[within CI]" } else { "[DIFFERS]" },
                    nonconv as f64 / shots as f64 * 100.0,
                    resid as f64 / shots as f64 * 100.0
                );
            }
        }
    }
}
```

- [ ] **Step 2: Smoke-run locally** (tiny budget — correctness of the harness, not statistics)

Run: `cargo run --release -p aleph-qec --example qec_q7_stream_sweep -- 6 200 7`
Expected: CSV header + 3×8 = 24 data rows, no panics; `resid_frac` near the nonconv fraction;
batch LER printed per p. (~a few minutes on the M-series laptop.)

- [ ] **Step 3: Full workspace gate**

Run: `cargo test -p aleph-qec && cargo clippy --workspace --all-targets -- -D warnings && cargo fmt --check`
Expected: all green. Fix anything clippy flags (likely `needless_range_loop` style nits) before
committing.

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-qec/examples/qec_q7_stream_sweep.rs
git commit -m "[Q7-04] qec_q7_stream_sweep: (W,C) x seam x p LER sweep vs batch

Same-shots comparison isolates the windowing cost; reports nonconv and
residual fractions (feeds Q7-07). Decision rule per the M9 design spec."
```

---

### Task 6: EPYC sweep run

**Files:**
- Create: sweep CSVs under `docs/perf/data/` (check the dir exists; if the repo keeps sweep CSVs
  elsewhere — look at how M5/M8 stored theirs in `docs/perf/qec-q7-fixed-bp.md` — follow that
  precedent; if none, inline the table in the doc and keep raw CSVs out of git)

**Interfaces:**
- Produces: the sweep table + chosen (W, C, seam) → Task 7's doc section.

- [ ] **Step 1: Verify the EPYC box is idle** (project memory: CI races silently corrupt
  measurements — this is an LER run, not a latency run, so contention only costs wall time, but
  check anyway)

```bash
ssh root@195.154.249.85 'uptime; pgrep -af "cargo bench|bencher run|Runner.Worker" || echo IDLE'
```

- [ ] **Step 2: Transfer the branch via git bundle** (memory lesson: don't push mid-measurement;
  the runner shares the box)

```bash
git bundle create /tmp/m9a.bundle main..q7-04-m9a-sliding-window-bp
scp /tmp/m9a.bundle root@195.154.249.85:/root/
ssh root@195.154.249.85 'cd /root/aleph && git fetch /root/m9a.bundle q7-04-m9a-sliding-window-bp:q7-04-m9a && git checkout q7-04-m9a && git log --oneline -1'
```

(If `/root/aleph` does not exist, clone from the CI runner's `_work` checkout per the
`p46-02-merged` memory, then apply the bundle. Cargo lives at
`~/.rustup/toolchains/*/bin/cargo` — not on PATH.)

- [ ] **Step 3: Pilot timing run** (sizes the full run; nohup + poll, never block the shell)

```bash
ssh root@195.154.249.85 'cd /root/aleph && nohup ~/.rustup/toolchains/*/bin/cargo run --release -p aleph-qec --example qec_q7_stream_sweep -- 12 2000 2024 > /root/m9a-pilot.csv 2> /root/m9a-pilot.log &'
# poll:
ssh root@195.154.249.85 'tail -5 /root/m9a-pilot.log; ls -la /root/m9a-pilot.csv'
```

Scale the full-run shots so wall time stays under ~6 h: if the pilot (2 k shots) takes T minutes,
full shots ≈ `2000 × 360/T`, capped at 100 k, floored at 20 k. (LER at p=0.003 is expected
O(1e-3..1e-2) at rounds=12, so 20 k shots gives a usable CI; the chosen config gets a
confirmation run at ≥100 k or the largest affordable.)

- [ ] **Step 4: Full sweep + confirmation run**

```bash
ssh root@195.154.249.85 'cd /root/aleph && nohup ~/.rustup/toolchains/*/bin/cargo run --release -p aleph-qec --example qec_q7_stream_sweep -- 12 <SHOTS> 2024 > /root/m9a-sweep.csv 2> /root/m9a-sweep.log &'
```

Then apply the decision rule (smallest (W, C, seam) within batch CI at every p, soft priors only
on a clear win) and re-run just that config mentally — the sweep already includes it; the
confirmation run repeats the full sweep at higher shots ONLY for the chosen (W, C): edit nothing,
just note which rows matter. Copy both CSVs back:

```bash
scp root@195.154.249.85:/root/m9a-{pilot,sweep}.csv /Users/ex/GitHub/aleph/docs/perf/data/ 2>/dev/null || scp root@195.154.249.85:/root/m9a-sweep.csv /private/tmp/claude-501/-Users-ex-GitHub-aleph/0d395d29-de7a-4745-89d0-b09538be7920/scratchpad/
```

- [ ] **Step 5: Sanity-check the numbers before writing the doc**

Expected shape: LER(W=6) ≤ LER(W=4) ≤ LER(W=3) at fixed C (more lookahead never hurts);
full-window would equal batch. If a windowed LER is *below* batch beyond CI, suspect a bug
(windowing cannot beat batch systematically) — stop and debug before documenting.

---

### Task 7: Doc section + BACKLOG tick + PR

**Files:**
- Modify: `docs/perf/qec-q7-fixed-bp.md` (append the M9a section after M8, following the
  existing per-milestone section style — read the M8 section header format first)
- Modify: `docs/qec/BACKLOG.md:1360` (tick AC-1)

- [ ] **Step 1: Write the M9a section** in `docs/perf/qec-q7-fixed-bp.md`. Structure (content
  from the actual sweep results — no invented numbers):

```markdown
## M9a — sliding-window golden + (W, C) sweep (Q7-04, PR #NNN)

**What:** `SlidingWindowBp` (`crates/aleph-qec/src/relay_window.rs`) — residual-carry sliding
window over multi-round circuit-level BB DEMs, per-window base decoder = the frozen `FixedRelayBp`
operating point (Q5.3, 6×10). Hypergraph time cut by truncation; commit on error-vars (earliest
in-window detector < commit boundary); seam = residual-only or + soft priors. One-window case
pinned bit-exact to batch; interior windows pinned translation-invariant (the M9b one-header RTL
property).

**Sweep (EPYC, rounds=12, N shots, seed 2024):**
<the CSV table, formatted>

**Chosen config: (W=?, C=?, seam=?)** — <one-paragraph rationale against the decision rule;
LER cost vs batch at each p; nonconv fraction (→ Q7-07)>.

**AC-1 met:** multi-round (rounds ≥ 3) circuit-level windows generated and decoded; golden
matches batch on the one-window anchor and stays within <the measured band> of batch LER at the
chosen config.
```

- [ ] **Step 2: Tick AC-1** in `docs/qec/BACKLOG.md` (line ~1360): change
  `- [ ] Emitter generates multi-round (rounds ≥ 3) circuit-level windows; golden model matches.`
  to `- [x] …` (keep the text; append a short pointer: `(M9a, PR #NNN — see
  docs/perf/qec-q7-fixed-bp.md § M9a)`).

- [ ] **Step 3: Final gates + self-review**

```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
git diff main --stat   # re-read the whole diff with fresh eyes
```

- [ ] **Step 4: Push + PR**

```bash
git push -u origin q7-04-m9a-sliding-window-bp
gh pr create --title "[Q7-04] M9a: sliding-window streaming relay-BP golden + (W,C) sweep" --body "$(cat <<'EOF'
Part 1 of 3 for #455 (M9a of the design spec
`docs/superpowers/specs/2026-07-11-q7-04-streaming-relay-bp-design.md`). Do not auto-close.

## What
- `SlidingWindowBp`: residual-carry sliding window over multi-round circuit-level BB DEMs,
  base decoder = frozen `FixedRelayBp` (Q5.3, 6×10). Hypergraph truncation cut; commit on
  error-vars; seam residual-only / + soft priors.
- `FixedRelayBp::{lambda_q, with_lambda_q}` prior-override hooks.
- `qec_q7_stream_sweep` example + EPYC sweep → chosen (W, C, seam) for M9b.

## Tests
<test list + counts>

## Sweep results
<table + chosen config + rationale>

## AC
Q7-04 AC-1 ticked; AC-2/AC-3 are M9b/M9c.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 5: Code review + merge** per the standard workflow (`/code-review`, addressed
  findings, squash-merge after CI green).

## Execution notes

- Tests in Tasks 3–4 decode gross-code circuit DEMs with the full 6×10 schedule — locally a few
  minutes total is expected; if a test exceeds ~30 s, cut its shot count (they are correctness
  anchors, not statistics) rather than `#[ignore]`.
- The Task 2 translation-invariance test is the load-bearing one for M9b. If it fails, the
  contingency (canonical window sort) changes the Task 3 batch-equality strategy — surface that
  to the user before proceeding, per the spec's "one baked header" premise.
