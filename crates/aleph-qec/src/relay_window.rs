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
//!
//! **Tail schedule (M9b co-sim contract):** the slot loop advances `s += C` past the first
//! commit-all window, so a stream decodes `⌈num_slices/C⌉` slots INCLUDING degenerate shrinking
//! commit-all tail windows (rounds=12, W=6, C=2 → 7 slots, of which s=8/10/12 are all
//! commit-all). This mirrors `sliding.rs`; the streaming RTL must replay the same schedule or
//! the bit-exact gate fails at stream end.
//!
//! # Hardware schedule (M9b)
//!
//! [`SlidingWindowBp`] compiles a **fresh window DEM per slot** (`window_dem(s, s+W)`) — exact,
//! but the RTL has exactly one baked window graph, not one per slot. [`HwSlidingWindowBp`] is
//! the golden the RTL streaming decoder is gated bit-exact against: it decodes *every* slot on
//! the **single interior window graph** compiled once at construction (translation invariance,
//! see above, is what makes that graph identical to what an exact per-slot compile would have
//! produced for any non-edge slot), sliding a fixed-size local frame across the stream and
//! **zero-padding** past the real stream end instead of shrinking the window. The commit mask is
//! likewise baked once from the interior graph's local structure, not recomputed per slot. This
//! makes [`HwSlidingWindowBp::decode_stream`] a pure function of (uniform-graph template,
//! baked commit mask, stream) — exactly the state an FPGA/ASIC datapath can hold. `streamvectors`
//! and the RTL co-sim gate key on this struct and its
//! [`WindowTrace`]-producing [`HwSlidingWindowBp::decode_stream_trace`]; [`SlidingWindowBp`]
//! remains the exact-schedule LER reference the (W, C, seam) sweep picked from.
//!
//! **Discard semantics.** Each slide drops the frame's leading `C` rounds unconditionally —
//! commit-region bits still lit after a slot's commit toggle slide off and are *gone*; no later
//! window can re-explain them (unlike [`SlidingWindowBp`], whose residual stays in the global
//! stream). Mid-stream nonconvergence is therefore visible only through the per-slot
//! [`WindowTrace::commit_clean`] flag and the cumulative discarded-bit count in
//! [`StreamStats::residual`]; the logical-error rate remains the true quality metric.

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
    /// Undrained detector count. When every window converged this must be 0 (each lit
    /// commit-region detector is covered by an odd number of fired mechanisms, all of which
    /// touch the commit region and therefore commit and toggle it).
    ///
    /// [`SlidingWindowBp`]: detectors still lit in the global stream after the final window.
    /// [`HwSlidingWindowBp`]: cumulative count of lit commit-region bits **discarded** by the
    /// slides (sampled per slot after the commit toggle, before the slide) — the end-of-stream
    /// local frame is all zero-padded rounds and would always count 0.
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

    /// Total number of round-slices in the stream (`max(detector_round) + 1`).
    pub fn num_slices(&self) -> usize {
        self.num_slices
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
            carry = Some((soft.llr.iter().map(|&x| x as i32).collect(), committed));
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

/// One window slot's decision under the hardware schedule — the unit the RTL must reproduce
/// bit-exactly (see the module's "Hardware schedule" doc section).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WindowTrace {
    /// Per window-var (indexed like [`HwSlidingWindowBp::window_export`]'s `dem.errors`): `1` iff
    /// the decoder chose the var (`ê_v = 1`) AND its baked commit bit
    /// ([`HwSlidingWindowBp::commit_mask`]) is set.
    pub committed: Vec<u8>,
    /// XOR of this slot's committed vars' observable masks — this slot's contribution to the
    /// running logical correction.
    pub obs: u64,
    /// The base decoder's syndrome-valid flag for this slot (`true` iff some relay-BP leg found
    /// an `ê` reproducing this slot's local syndrome).
    pub valid: bool,
    /// True iff the commit region `frame[0, C·dpr)` is all-zero AFTER this slot's commit toggle
    /// (and before the slide). Every var with a det in the commit region has its commit bit set
    /// by construction, so a converged decode always drains the region — the flag is non-vacuous
    /// (nonconverged decodes can leave it dirty) and maps to the RTL result word's
    /// `residual_empty` bit.
    pub commit_clean: bool,
}

/// The hardware-schedule golden: decodes **every** window slot on the single interior window
/// graph baked at construction, sliding a fixed-size local residual frame across the stream and
/// zero-padding past the real stream end. See the module's "Hardware schedule" doc section for
/// why this differs from [`SlidingWindowBp`] (which compiles an exact DEM per slot) and why that
/// difference is exactly what the RTL — one baked graph, no per-slot recompilation — needs a
/// golden for.
#[derive(Clone, Debug)]
pub struct HwSlidingWindowBp {
    /// Total round-slices in the stream this instance was built for (`max(detector_round) + 1`).
    num_slices: usize,
    /// Window length `W` (rounds).
    window: usize,
    /// Commit-region length `C` (rounds).
    commit: usize,
    /// Detectors per round-slice (uniform by construction; 72 for the gross code).
    dpr: usize,
    /// The one interior window graph every slot decodes on.
    export: WindowBpExport,
    /// The base decoder, built once over `export.dem` at the frozen M8 operating point.
    decoder: FixedRelayBp,
    /// Baked per-local-variable commit bit: `export.dem.errors[v]` has some detector at local
    /// round `< commit`.
    commit_mask: Vec<bool>,
    /// Baked per-local-variable observable bitmask (`export.dem.errors[v].obs` as a `u64` mask).
    obs_mask: Vec<u64>,
    /// `⌈num_slices / commit⌉` — the number of slots a full-stream decode replays.
    num_slots: usize,
}

impl HwSlidingWindowBp {
    /// Build the HW-schedule golden: compile the single interior window graph (same offset
    /// formula the M9b RTL window-graph emitter uses: `s0 = ((rounds − W)/2).max(1)`, `rounds =
    /// num_slices − 1`) and bake its decoder + commit mask once.
    ///
    /// # Panics
    /// If `detector_round.len() != dem.detectors`, `commit` is not in `1..=window`, the stream's
    /// detectors are not laid out **round-major with a uniform per-round count** (`detector d` at
    /// global round `r` ⇒ `d == r*dpr + p` for some `p < dpr`, dpr uniform across rounds — the
    /// structural property that lets one baked graph stand in for every slot), or the stream is
    /// too short to leave a trailing round past the interior window.
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
        assert!(num_slices > 0, "need at least one round-slice");

        // Uniform dpr: every round-slice must have the same detector count, and detectors must
        // be laid out round-major (id = round*dpr + position) — the RTL bakes ONE window graph,
        // so every slot's local topology must be identical, which requires this layout globally.
        let dpr = dem.detectors / num_slices;
        assert_eq!(
            dpr * num_slices,
            dem.detectors,
            "detectors must split evenly into round-slices"
        );
        for (d, &r) in detector_round.iter().enumerate() {
            assert_eq!(
                d / dpr,
                r,
                "detectors must be round-major with a uniform per-round count (id = round*dpr + position)"
            );
        }

        let rounds = num_slices - 1;
        assert!(
            rounds >= window,
            "stream must have more rounds than the window to compile an interior graph"
        );
        let s0 = ((rounds - window) / 2).max(1);
        assert!(
            s0 + window <= rounds,
            "interior window must leave a trailing round"
        );

        let export = compile_window(&dem, &detector_round, s0, s0 + window);
        let decoder = FixedRelayBp::with_budget(
            &export.dem,
            LEGS,
            ITERS_PER_LEG,
            GAMMA,
            SEED,
            MSG_BITS,
            FRAC_BITS,
        );
        let commit_mask: Vec<bool> = export
            .dem
            .errors
            .iter()
            .map(|e| e.dets.iter().any(|&d| (d as usize) / dpr < commit))
            .collect();
        let obs_mask: Vec<u64> = export
            .dem
            .errors
            .iter()
            .map(|e| {
                e.obs.iter().fold(0u64, |acc, &o| {
                    // The per-slot obs contribution is a u64 mask (the RTL's BP_OBS_MASK word).
                    assert!(o < 64, "observable index {o} does not fit the u64 obs mask");
                    acc | (1u64 << o)
                })
            })
            .collect();
        let num_slots = num_slices.div_ceil(commit);

        Self {
            num_slices,
            window,
            commit,
            dpr,
            export,
            decoder,
            commit_mask,
            obs_mask,
            num_slots,
        }
    }

    /// Enable/disable first-valid early termination in the per-slot base decode (passthrough to
    /// [`FixedRelayBp::with_early_exit`]; `decode_fixed_soft` honors it). The RTL core's
    /// early-exit mode takes the FIRST syndrome-valid leg while the default keeps the
    /// lowest-weight valid decision over the whole schedule — the decisions genuinely differ
    /// where first-valid ≠ best-kept, so each RTL mode is gated against its own golden (the
    /// M6–M8 `circvectors`/`circvectorsearly` house pattern). Schedule, commit logic, and trace
    /// fields are unchanged.
    pub fn with_early_exit(mut self, on: bool) -> Self {
        self.decoder = self.decoder.with_early_exit(on);
        self
    }

    /// The one interior window graph every slot decodes on (see the module's "Hardware schedule"
    /// doc section) — public because the M9b RTL window-graph emitter consumes it.
    pub fn window_export(&self) -> &WindowBpExport {
        &self.export
    }

    /// Detectors per round-slice (uniform across the whole stream by construction).
    pub fn dpr(&self) -> usize {
        self.dpr
    }

    /// The baked per-local-variable commit bit (`BP_VAR_COMMIT` in the RTL), one per
    /// [`window_export`](Self::window_export)'s `dem.errors`.
    pub fn commit_mask(&self) -> &[bool] {
        &self.commit_mask
    }

    /// `⌈num_slices / commit⌉` — the number of slots a full-stream decode replays, including the
    /// degenerate zero-padded tail slots past the real stream end.
    pub fn num_slots(&self) -> usize {
        self.num_slots
    }

    /// Load `count` local rounds starting at global round `from` into `dst` (`count*dpr` local
    /// detectors), zero for any round `>= num_slices` — the RTL's zero-pad-past-stream-end
    /// contract. `lit` is the dense global syndrome bit-vector (`num_slices*dpr` long).
    fn load_rounds(
        dst: &mut [bool],
        from: usize,
        count: usize,
        lit: &[bool],
        dpr: usize,
        num_slices: usize,
    ) {
        for r in 0..count {
            let round = from + r;
            let (lo, hi) = (r * dpr, (r + 1) * dpr);
            if round < num_slices {
                dst[lo..hi].copy_from_slice(&lit[round * dpr..(round + 1) * dpr]);
            } else {
                dst[lo..hi].fill(false);
            }
        }
    }

    /// Decode an entire stream syndrome under the hardware schedule, discarding the per-slot
    /// trace (use [`decode_stream_trace`](Self::decode_stream_trace) to keep it). Delegates to
    /// `decode_stream_trace` rather than duplicating the slot FSM — the RTL bit-exact gate keys
    /// on that one implementation, so there is exactly one place the schedule logic can drift.
    pub fn decode_stream(&self, syn: &Syndrome) -> (Correction, StreamStats) {
        let (corr, stats, _trace) = self.decode_stream_trace(syn);
        (corr, stats)
    }

    /// Decode an entire stream syndrome under the hardware schedule, returning the committed
    /// logical correction, stream statistics, and the per-slot [`WindowTrace`] the RTL co-sim
    /// gate compares bit-for-bit.
    ///
    /// Every slot decodes on the single interior graph baked at construction: the local residual
    /// frame holds `window` round-slices, slides forward by `commit` rounds each slot, and is
    /// zero-padded for any round past the real stream end (`num_slices`) — the RTL has no way to
    /// shrink its window for a degenerate tail, so neither does this golden.
    pub fn decode_stream_trace(
        &self,
        syn: &Syndrome,
    ) -> (Correction, StreamStats, Vec<WindowTrace>) {
        let total_detectors = self.num_slices * self.dpr;
        let mut lit = vec![false; total_detectors];
        for &d in &syn.fired {
            if (d as usize) < total_detectors {
                lit[d as usize] = true;
            }
        }

        let wlen = self.window * self.dpr;
        let mut frame = vec![false; wlen];
        // Warm: rounds [0, window) of the real stream, zero-padded past the end.
        Self::load_rounds(&mut frame, 0, self.window, &lit, self.dpr, self.num_slices);

        let nvars = self.export.dem.errors.len();
        let clen = self.commit * self.dpr;
        let mut logical = vec![false; self.export.dem.observables];
        let mut nonconverged = 0usize;
        let mut residual = 0usize;
        let mut trace = Vec::with_capacity(self.num_slots);

        for k in 0..self.num_slots {
            let fired: Vec<u32> = (0..wlen as u32).filter(|&l| frame[l as usize]).collect();
            let syn_w = Syndrome::new(wlen, fired);
            let soft = self.decoder.decode_fixed_soft(&syn_w);
            if !soft.converged {
                nonconverged += 1;
            }

            let mut committed = vec![0u8; nvars];
            let mut obs = 0u64;
            for (v, &e) in soft.ehat.iter().enumerate() {
                if e == 1 && self.commit_mask[v] {
                    committed[v] = 1;
                    obs ^= self.obs_mask[v];
                    for &d in &self.export.dem.errors[v].dets {
                        frame[d as usize] ^= true;
                    }
                }
            }
            for (o, flip) in logical.iter_mut().enumerate() {
                *flip ^= (obs >> o) & 1 == 1;
            }
            // Sampled after the commit toggle, before the slide: lit commit-region bits are
            // about to slide off and be DISCARDED — they, summed over slots, are the residual.
            // (The frame after the final slide is all zero-padded rounds, always 0 — useless.)
            let dirty = frame[..clen].iter().filter(|&&x| x).count();
            residual += dirty;
            trace.push(WindowTrace {
                committed,
                obs,
                valid: soft.converged,
                commit_clean: dirty == 0,
            });

            // Slide by C rounds: drop the committed C rounds, pull in C fresh ones (or zero past
            // stream end) — the exact FSM reload the RTL replays every slot.
            frame.copy_within(clen.., 0);
            let new_lo = (k + 1) * self.commit + (self.window - self.commit);
            let tail_lo = wlen - clen;
            Self::load_rounds(
                &mut frame[tail_lo..],
                new_lo,
                self.commit,
                &lit,
                self.dpr,
                self.num_slices,
            );
        }

        (
            Correction::new(logical),
            StreamStats {
                windows: self.num_slots,
                nonconverged,
                residual,
            },
            trace,
        )
    }
}

impl crate::decoder::Decoder for HwSlidingWindowBp {
    /// Decode a full-stream syndrome via the hardware schedule (stats/trace dropped; use
    /// [`HwSlidingWindowBp::decode_stream`] or
    /// [`HwSlidingWindowBp::decode_stream_trace`] to keep them).
    fn decode(&self, syndrome: &Syndrome) -> Correction {
        self.decode_stream(syndrome).0
    }

    /// Shots are independent; mirrors [`SlidingWindowBp`]'s rayon-parallel batch path.
    fn decode_batch(&self, syndromes: &[Syndrome]) -> crate::error::Result<Vec<Correction>> {
        use rayon::prelude::*;
        Ok(syndromes.par_iter().map(|s| self.decode(s)).collect())
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

    use crate::experiment::sample_shots;
    use crate::fixed_bp::FixedRelayBp;

    // --- HwSlidingWindowBp (M9b) -------------------------------------------------------------

    /// Build a gross-code circuit-level DEM + detector-round vector — the fixture the realistic
    /// (RTL-representative) HW-schedule tests build. Its rounds=12 circuit-level DEM + 4824-var
    /// window Tanner graph take ~200 s to construct in the unoptimized debug profile CI uses, so
    /// every test on this fixture is `#[ignore]`-marked (run manually with
    /// `cargo test -p aleph-qec --release -- --ignored`).
    fn gross_stream(rounds: usize, p: f64) -> (DetectorErrorModel, Vec<usize>) {
        let code = BBCode::gross();
        let dem = code
            .circuit_level_dem(rounds, CircuitNoise::uniform(p))
            .unwrap();
        let dr = code.memory_x_experiment(rounds).detector_rounds();
        (dem, dr)
    }

    /// The small `[[72,12,6]]` BB code (ℓ=m=6, same polynomials as gross) — its circuit-level DEM
    /// and window graph are a fraction of gross's, cheap to build even in the debug profile, so
    /// the code-agnostic golden invariants (slot arithmetic, interior-graph match, determinism,
    /// trace aggregation, converged-drain, early-exit divergence) are gated LIVE here. The gross
    /// fixture carries the RTL-representative dense-regime checks in the manual `--ignored` runs.
    fn small_stream(rounds: usize, p: f64) -> (DetectorErrorModel, Vec<usize>, usize) {
        let code = BBCode::new(6, 6, &[(3, 0), (0, 1), (0, 2)], &[(0, 3), (1, 0), (2, 0)]);
        let dem = code
            .circuit_level_dem(rounds, CircuitNoise::uniform(p))
            .unwrap();
        let exp = code.memory_x_experiment(rounds);
        let dr = exp.detector_rounds();
        // Detectors-per-round for this code: the X-check count (dr is round-major, uniform).
        let dpr = dem.detectors / (rounds + 1);
        (dem, dr, dpr)
    }

    /// Build the HW-schedule golden over a gross-code stream, keeping the source DEM alongside
    /// it so callers can sample shots from the same distribution ([`sample_one_shot`]).
    fn hw_gross(
        rounds: usize,
        p: f64,
        window: usize,
        commit: usize,
    ) -> (HwSlidingWindowBp, DetectorErrorModel) {
        let (dem, dr) = gross_stream(rounds, p);
        let hw = HwSlidingWindowBp::new(dem.clone(), dr, window, commit);
        (hw, dem)
    }

    /// Sample one Monte-Carlo shot from `dem` — the same sampler [`sample_shots`] uses, so HW and
    /// exact-schedule LER comparisons decode identical shot streams.
    fn sample_one_shot(dem: &DetectorErrorModel, seed: u64) -> Syndrome {
        sample_shots(dem, 1, seed)
            .0
            .into_iter()
            .next()
            .expect("sample_shots(dem, 1, _) returns exactly one syndrome")
    }

    /// The correction's observable flips packed into a `u64` bitmask, for comparing against
    /// [`WindowTrace::obs`].
    fn obs_from(corr: &Correction) -> u64 {
        corr.observable_flips
            .iter()
            .enumerate()
            .fold(0u64, |acc, (o, &b)| if b { acc | (1u64 << o) } else { acc })
    }

    /// Interior-graph match + slot arithmetic — LIVE on the small code (cheap to build in debug).
    /// The HW golden bakes the ONE interior window graph and must reproduce exactly what the
    /// exact-schedule decoder compiles for that offset; `num_slots`/`dpr` are pure metadata.
    #[test]
    fn hw_interior_graph_matches_window_dem() {
        let (rounds, window, commit) = (6usize, 3usize, 1usize);
        let (dem, dr, dpr) = small_stream(rounds, 0.003);
        let num_slices = rounds + 1;
        let sw = SlidingWindowBp::new(dem.clone(), dr.clone(), window, commit);
        let hw = HwSlidingWindowBp::new(dem, dr, window, commit);
        let s0 = ((rounds - window) / 2).max(1); // the emitter's interior-offset formula
        assert_eq!(hw.window_export().dem, sw.window_dem(s0, s0 + window).dem);
        assert_eq!(hw.dpr(), dpr);
        assert_eq!(hw.num_slots(), num_slices.div_ceil(commit));
        assert_eq!(hw.commit_mask().len(), hw.window_export().dem.errors.len());
        assert!(hw.commit_mask().iter().any(|&b| b), "some var must commit");
    }

    /// Realistic gross-code interior-graph match at the RTL op point (rounds=12, W=6, C=2 ⇒ 7
    /// slots, dpr=72). Heavy: the rounds=12 gross DEM + 4824-var window Tanner build is ~200 s in
    /// the debug profile (construction cost, not decode — no shots are run here), so it is
    /// `#[ignore]`d out of the debug CI gate. Run it manually with
    /// `cargo test -p aleph-qec --release -- --ignored`; the small-code live twin above covers the
    /// invariant, and the emitter's committed gross vectors gate the RTL.
    #[test]
    #[ignore]
    fn hw_interior_graph_matches_window_dem_gross() {
        let (rounds, window) = (12usize, 6usize);
        let (dem, dr) = gross_stream(rounds, 0.003);
        let sw = SlidingWindowBp::new(dem.clone(), dr.clone(), window, 2);
        let hw = HwSlidingWindowBp::new(dem, dr, window, 2);
        let s0 = ((rounds - window) / 2).max(1);
        assert_eq!(hw.window_export().dem, sw.window_dem(s0, s0 + window).dem);
        assert_eq!(hw.dpr(), 72);
        assert_eq!(hw.num_slots(), 7);
    }

    /// Core per-shot decode invariants on the SMALL code (LIVE, cheap in debug): determinism,
    /// `decode_stream`/`decode_stream_trace` agreement, obs-aggregation (committed obs masks XOR
    /// to the returned logical), and the converged-drain contract (a fully-converged stream
    /// discards nothing — `residual == 0` and every slot `commit_clean`). The realistic gross
    /// W=6 twins are the `#[ignore]`d `hw_trace_aggregates_to_stream_decode` and
    /// `hw_converged_stream_drains_residual`.
    #[test]
    fn hw_small_stream_decode_invariants() {
        let (dem, dr, _dpr) = small_stream(6, 0.004);
        let hw = HwSlidingWindowBp::new(dem.clone(), dr, 3, 1);
        let mut converged_checked = 0;
        let mut dirty_shots = 0;
        for seed in 0..8u64 {
            let syn = sample_one_shot(&dem, seed);
            let (corr, stats, trace) = hw.decode_stream_trace(&syn);
            // Pure function of the shot.
            let (corr2, stats2, trace2) = hw.decode_stream_trace(&syn);
            assert_eq!((&corr, &stats, &trace), (&corr2, &stats2, &trace2));
            // decode_stream agrees with the trace-returning path.
            let (corr_s, stats_s) = hw.decode_stream(&syn);
            assert_eq!(corr_s, corr);
            assert_eq!(stats_s.residual, stats.residual);
            // Committed obs contributions XOR to the returned logical correction.
            let obs_xor = trace.iter().fold(0u64, |a, t| a ^ t.obs);
            assert_eq!(obs_from(&corr), obs_xor);
            // residual > 0 ⟺ some slot left its commit region dirty (the two views agree).
            assert_eq!(stats.residual > 0, trace.iter().any(|t| !t.commit_clean));
            if stats.nonconverged == 0 {
                assert_eq!(stats.residual, 0, "converged stream discarded lit bits");
                assert!(trace.iter().all(|t| t.commit_clean));
                converged_checked += 1;
            } else if stats.residual > 0 {
                dirty_shots += 1;
            }
        }
        assert!(
            converged_checked > 0,
            "expected some fully-converged shots on the small code"
        );
        // The dirty branch must actually fire, else the residual⟺commit_clean check above is
        // vacuous (a future p/rounds tweak could silently drain everything).
        assert!(
            dirty_shots > 0,
            "expected some shot to discard lit commit-region bits — the residual metric would \
             be vacuous otherwise; raise p or the seed budget if this regresses"
        );
    }

    /// Early-exit golden invariants on the SMALL code (LIVE, cheap in debug): the first-valid
    /// early-exit trace differs from the best-kept trace on ≥1 slot within the seed budget (the
    /// two goldens are genuinely distinct — the M6–M8 house pattern), and the early-exit decode
    /// is deterministic. This is the only LIVE coverage of `with_early_exit`; the realistic gross
    /// twin is the `#[ignore]`d [`hw_early_exit_differs_and_is_deterministic`].
    #[test]
    fn hw_small_early_exit_differs_and_is_deterministic() {
        let (dem, dr, _dpr) = small_stream(6, 0.006);
        let best = HwSlidingWindowBp::new(dem.clone(), dr.clone(), 3, 1);
        let early = HwSlidingWindowBp::new(dem.clone(), dr, 3, 1).with_early_exit(true);
        let mut differs = false;
        for seed in 0..24u64 {
            let syn = sample_one_shot(&dem, seed);
            // Early-exit decode is a pure function of the shot.
            let (ca, sa, ta) = early.decode_stream_trace(&syn);
            let (cb, sb, tb) = early.decode_stream_trace(&syn);
            assert_eq!(
                (&ca, &sa, &ta),
                (&cb, &sb, &tb),
                "early-exit decode_stream_trace must be deterministic"
            );
            let (_c, _s, t_best) = best.decode_stream_trace(&syn);
            if t_best != ta {
                differs = true;
                break; // divergence pinned; determinism already checked on every seed seen
            }
        }
        assert!(
            differs,
            "first-valid (early-exit) and best-kept traces never differed on the small code — \
             the two goldens would be redundant; raise p or the seed budget if this regresses"
        );
    }

    /// Realistic W=6 dense-regime version of the aggregation invariant. Heavy: each stream decode
    /// of the 4824-var gross window graph is ~115 s in the unoptimized debug profile (~100× the
    /// release cost), so it is `#[ignore]`d out of CI's debug `cargo test --workspace` gate. Run
    /// it manually with `cargo test -p aleph-qec --release -- --ignored`; the per-PR safety net is
    /// the small-code live twin [`hw_small_stream_decode_invariants`] plus the committed RTL
    /// co-sim vectors (`hw/bp_stream_vectors.txt`, regenerated by `make -C hw bpstream`).
    #[test]
    #[ignore]
    fn hw_trace_aggregates_to_stream_decode() {
        let (hw, dem) = hw_gross(12, 0.003, 6, 2);
        for seed in 0..8u64 {
            let syn = sample_one_shot(&dem, seed);
            let (corr, stats) = hw.decode_stream(&syn);
            let (corr_t, stats_t, trace) = hw.decode_stream_trace(&syn);
            assert_eq!(corr, corr_t);
            assert_eq!(stats.residual, stats_t.residual);
            let obs_xor = trace.iter().fold(0u64, |a, t| a ^ t.obs);
            assert_eq!(obs_from(&corr), obs_xor);
        }
    }

    /// The early-exit golden is a genuinely DIFFERENT golden (M6–M8 house pattern): first-valid
    /// and best-kept decisions must differ on at least one slot at p=0.005 within a few seeds,
    /// and the early-exit decode must itself stay a pure function of the shot.
    ///
    /// Heavy: the DENSE p=0.005 W=6 regime (where divergence is common — 25/280 slots at the op
    /// point) with ~115 s-per-decode gross stream decodes, so it is `#[ignore]`d out of the debug
    /// CI gate. Run it manually with `cargo test -p aleph-qec --release -- --ignored`; the per-PR
    /// safety net is the small-code live twin
    /// [`hw_small_early_exit_differs_and_is_deterministic`].
    #[test]
    #[ignore]
    fn hw_early_exit_differs_and_is_deterministic() {
        let (dem, dr) = gross_stream(12, 0.005);
        let best = HwSlidingWindowBp::new(dem.clone(), dr.clone(), 6, 2);
        let early = HwSlidingWindowBp::new(dem.clone(), dr, 6, 2).with_early_exit(true);
        // Divergence is common at p=0.005 (25/280 slots at the op point), so it fires within a
        // few seeds; the loop breaks on the first, keeping the debug-profile cost bounded.
        let mut differs = false;
        for seed in 0..12u64 {
            let syn = sample_one_shot(&dem, seed);
            let (ca, sa, ta) = early.decode_stream_trace(&syn);
            let (cb, sb, tb) = early.decode_stream_trace(&syn);
            assert_eq!(
                (&ca, &sa, &ta),
                (&cb, &sb, &tb),
                "early-exit decode_stream_trace must be deterministic"
            );
            let (_c, _s, t_best) = best.decode_stream_trace(&syn);
            if t_best != ta {
                differs = true;
                break; // mechanism pinned; determinism already checked on every seed seen
            }
        }
        assert!(
            differs,
            "first-valid (early-exit) and best-kept traces never differed in 12 shots at \
             p=0.005 — the two goldens would be redundant"
        );
    }

    /// Mirror of M9a's `converged_stream_drains_residual`, on the HW schedule and its discarded-
    /// bits residual semantics: a fully-converged stream discards nothing (`residual == 0`,
    /// every slot `commit_clean`), and the dirty path is reachable — some nonconverged shot at
    /// p=0.005 leaves lit commit-region bits for the slide to discard (non-vacuous metric).
    ///
    /// Heavy: the dense p=0.005 W=6 regime is required for both the converged AND the dirty shot
    /// to appear in one seed set, and each W=6 stream decode is ~115 s in the debug profile
    /// (~100× release) — 30 seeds is ~1 h in debug, so it is `#[ignore]`d out of CI's debug
    /// `cargo test --workspace` gate. Run it manually with
    /// `cargo test -p aleph-qec --release -- --ignored` (~12 s there); the per-PR safety net is
    /// the small-code live twin [`hw_small_stream_decode_invariants`]. At p=0.005 dirty-discard is
    /// ~51 % and nonconvergence ~96 %, so 30 seeds reliably yield ≥1 converged and ≥1 dirty shot.
    #[test]
    #[ignore]
    fn hw_converged_stream_drains_residual() {
        let (hw, dem) = hw_gross(12, 0.005, 6, 2);
        let mut converged_checked = 0;
        let mut nonconv_shots = 0;
        let mut dirty_shots = 0;
        for seed in 0..30u64 {
            let syn = sample_one_shot(&dem, seed);
            let (_c, stats, trace) = hw.decode_stream_trace(&syn);
            if stats.nonconverged == 0 {
                assert_eq!(
                    stats.residual, 0,
                    "all windows converged but lit bits were discarded"
                );
                assert!(
                    trace.iter().all(|t| t.commit_clean),
                    "all windows converged but some slot left its commit region dirty"
                );
                converged_checked += 1;
            } else {
                nonconv_shots += 1;
                // residual > 0 ⟺ some slot's commit_clean is false (residual is exactly the
                // per-slot dirty popcounts summed); the two views must agree.
                let any_dirty = trace.iter().any(|t| !t.commit_clean);
                assert_eq!(stats.residual > 0, any_dirty);
                if any_dirty {
                    dirty_shots += 1;
                }
            }
        }
        assert!(converged_checked > 0, "expected some fully-converged shots");
        assert!(
            nonconv_shots > 0,
            "expected some nonconverged shots at p=0.005"
        );
        assert!(
            dirty_shots > 0,
            "expected some nonconverged shot to leave a dirty commit region \
             (the residual metric must be non-vacuous)"
        );
    }

    #[test]
    #[ignore] // slow sanity: HW-schedule LER within 2x of exact-schedule LER, same shots
    fn hw_ler_close_to_exact_schedule() {
        // n=2000 shots, p=0.003, rounds=12, W=6 C=2; count logical errors of
        // HwSlidingWindowBp vs SlidingWindowBp on identical sampled shots.
        let (rounds, p, shots) = (12, 0.003, 2000u64);
        let (dem, dr) = gross_stream(rounds, p);
        let hw = HwSlidingWindowBp::new(dem.clone(), dr.clone(), 6, 2);
        let sw = SlidingWindowBp::new(dem.clone(), dr, 6, 2);
        let (syndromes, truths) = sample_shots(&dem, shots, 0xC0FF_EE01);

        let mut hw_errors = 0u64;
        let mut sw_errors = 0u64;
        for (syn, truth) in syndromes.iter().zip(&truths) {
            let (hc, _) = hw.decode_stream(syn);
            let (sc, _) = sw.decode_stream(syn);
            if &hc.observable_flips != truth {
                hw_errors += 1;
            }
            if &sc.observable_flips != truth {
                sw_errors += 1;
            }
        }
        eprintln!("hw_errors={hw_errors} sw_errors={sw_errors} shots={shots}");
        assert!(
            hw_errors <= 2 * sw_errors + 5,
            "HW-schedule LER blew up vs exact-schedule LER: hw={hw_errors} exact={sw_errors} (shots={shots})"
        );
    }

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
            assert_eq!(
                corr, want,
                "one-window sliding decode must be bit-exact to batch"
            );
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
        assert!(
            converged_shots > 0,
            "expected some fully-converged shots at p=0.002"
        );
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
            assert_eq!(
                (&c1, &s1),
                (&c2, &s2),
                "decode_stream must be a pure function"
            );
            if s1.nonconverged == 0 {
                assert_eq!(s1.residual, 0);
            }
        }
    }
}
