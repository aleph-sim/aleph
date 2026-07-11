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
#[allow(unused_imports)] // consumed by decode_stream (Task 3)
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
#[allow(dead_code)] // consumed by decode_stream (Task 3)
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
#[allow(dead_code)] // consumed by decode_stream (Task 3)
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
    use crate::bivariate_bicycle::BBCode;
    use crate::builder::CircuitNoise;

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
        assert_eq!(
            w.mech_globals,
            vec![0, 1],
            "mechanism 2 has no in-window support"
        );
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
