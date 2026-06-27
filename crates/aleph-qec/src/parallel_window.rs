//! [`ParallelWindowDecoder`] — **parallel-window** streaming decoding (Q4-02), the concurrent
//! companion to the sequential [`SlidingWindowDecoder`](crate::SlidingWindowDecoder) (Q4-01).
//!
//! # Why parallel windows
//!
//! Sliding-window decoding is inherently *sequential*: window `k`'s input is the residual left by
//! window `k − 1`, so the windows form a dependency chain of depth `O(stream length)`. A single
//! decoder must therefore keep pace with the device *on its own*. When per-window decode is slower
//! than syndrome arrival the unprocessed queue grows without bound — the **backlog problem**
//! (Battistel et al., arXiv:2303.00054): reaction time for the next non-Clifford gate diverges and
//! fault tolerance breaks, even if the *average* decode rate looks fine.
//!
//! The parallel-window scheme (Skoric et al., arXiv:2209.08552; Tan et al., arXiv:2209.09219)
//! breaks the chain into **two layers of independent windows** so the decoding *depth* is `O(1)`
//! and throughput scales with the number of workers:
//!
//! * **Layer A** — the *even*-indexed commit regions `[0,C), [2C,3C), …` are decoded concurrently,
//!   each in its own window with a buffer of `B` rounds on *both* sides for future/past context.
//!   Each commits its own region's correction (toggling those detectors in the running residual)
//!   and XORs its observable flips into the logical correction.
//! * **Layer B** — the *odd*-indexed commit regions `[C,2C), [3C,4C), …` are then decoded
//!   concurrently against the residual **left by layer A**. Every odd region is flanked by two even
//!   regions that have already committed, so its seams are pinned on both sides — exactly the
//!   inherited boundary condition the parallel-window papers rely on.
//!
//! Both layers are embarrassingly parallel (`rayon`), and there are only two of them regardless of
//! stream length, so `P` workers give `≈ P×` the single-window throughput. That headroom is what
//! keeps the backlog bounded: pick `P` so the sustained service rate exceeds the arrival rate.
//!
//! # Seam handling (shared with the sliding decoder)
//!
//! Each window decode reuses the Q4-01 machinery: out-of-window detectors are cut at per-detector
//! **temporal-sink** nodes (kept distinct from the real spatial boundary, with free observable-less
//! drains) so a time cut never spuriously flips the logical observable. See
//! [`SlidingWindowDecoder`](crate::SlidingWindowDecoder) for the derivation of that fix.
//!
//! # Determinism under concurrency
//!
//! Windows in a layer are decoded by a pure read of the shared residual; each returns the list of
//! real detectors its committed edges toggle plus an observable mask. The toggles are then applied
//! by **XOR**, which is associative and commutative, so the result is independent of completion
//! order — there is no data race and no order-dependence even where two windows' buffers overlap.
//! For `C ≥ 2` on a graphlike (phenomenological) DEM, whose edges span at most one round, even
//! windows' write sets are in fact disjoint, so layer A and layer B each commit a well-defined,
//! seam-consistent correction; the logical-error rate matches full-batch decoding within CI once
//! the buffer `B ≳ d` (validated in `tests/parallel_window.rs`).

use crate::dem::{DemError, DetectorErrorModel};
use crate::error::Result;
use crate::matching::MatchingGraph;
use crate::syndrome::{Correction, Syndrome};
use crate::union_find::UnionFindDecoder;
use rayon::prelude::*;

/// One commit region and the window decoded to commit it: commit `[commit_lo, commit_hi)` decoded
/// against the buffered window `[win_lo, win_hi)` (all in round/time coordinates).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowPlan {
    /// First round of the window (inclusive), buffer included.
    pub win_lo: usize,
    /// Last round of the window (exclusive), buffer included.
    pub win_hi: usize,
    /// First round of the commit region (inclusive).
    pub commit_lo: usize,
    /// Last round of the commit region (exclusive).
    pub commit_hi: usize,
}

/// A parallel-window streaming decoder over a fixed [`DetectorErrorModel`] with a per-detector round
/// coordinate. Decodes the stream in two layers of mutually-independent windows (see the module
/// docs), wrapping the Union-Find decoder as the per-window base decoder.
#[derive(Clone, Debug)]
pub struct ParallelWindowDecoder {
    dem: DetectorErrorModel,
    /// Round (time index) of each detector; `detector_round.len() == dem.detectors`.
    detector_round: Vec<usize>,
    /// Number of time slices (max round + 1).
    num_slices: usize,
    /// Commit-region length `C` (rounds), `C ≥ 1`.
    commit: usize,
    /// Buffer `B` (rounds) on each side of a commit region; window length is `C + 2B` (clamped at
    /// the stream ends).
    buffer: usize,
    /// Whether the per-window Union-Find decode uses weighted growth (Q2-02).
    weighted: bool,
}

impl ParallelWindowDecoder {
    /// Build a parallel-window decoder. `detector_round[d]` is the time coordinate of detector `d`
    /// (e.g. from [`MemoryExperiment::detector_rounds`](crate::MemoryExperiment::detector_rounds)).
    /// `commit` is the commit-region length `C`; `buffer` is the `B` rounds of context on each side.
    ///
    /// # Panics
    /// If `detector_round.len() != dem.detectors` or `commit == 0`.
    pub fn new(
        dem: DetectorErrorModel,
        detector_round: Vec<usize>,
        commit: usize,
        buffer: usize,
    ) -> Self {
        assert_eq!(
            detector_round.len(),
            dem.detectors,
            "need one round per detector"
        );
        assert!(commit >= 1, "commit must be >= 1");
        let num_slices = detector_round
            .iter()
            .copied()
            .max()
            .map(|m| m + 1)
            .unwrap_or(0);
        Self {
            dem,
            detector_round,
            num_slices,
            commit,
            buffer,
            weighted: false,
        }
    }

    /// Switch the per-window base decoder to weighted (Q2-02) growth.
    pub fn weighted(mut self, yes: bool) -> Self {
        self.weighted = yes;
        self
    }

    /// Commit-region length `C`.
    pub fn commit(&self) -> usize {
        self.commit
    }

    /// Buffer `B` on each side of a commit region.
    pub fn buffer(&self) -> usize {
        self.buffer
    }

    /// The commit-region tiling `[0,C), [C,2C), …` covering the stream, each paired with its buffered
    /// window. The last region runs to the end of the stream. This is the work list both layers draw
    /// from (even indices → layer A, odd indices → layer B).
    pub fn window_plans(&self) -> Vec<WindowPlan> {
        let mut plans = Vec::new();
        let mut s = 0usize;
        while s < self.num_slices {
            let commit_hi = (s + self.commit).min(self.num_slices);
            // The final region absorbs any short tail so the commit regions tile the whole stream.
            let commit_hi = if commit_hi + self.commit > self.num_slices {
                self.num_slices
            } else {
                commit_hi
            };
            let win_lo = s.saturating_sub(self.buffer);
            let win_hi = (commit_hi + self.buffer).min(self.num_slices);
            plans.push(WindowPlan {
                win_lo,
                win_hi,
                commit_lo: s,
                commit_hi,
            });
            if commit_hi >= self.num_slices {
                break;
            }
            s = commit_hi;
        }
        plans
    }

    /// Number of windows (= commit regions) the stream tiles into.
    pub fn num_windows(&self) -> usize {
        self.window_plans().len()
    }

    /// The largest number of detectors any single window spans — the per-window working-set bound,
    /// `O(C + 2B)` and independent of total stream length.
    pub fn max_window_detectors(&self) -> usize {
        self.window_plans()
            .iter()
            .map(|p| {
                self.detector_round
                    .iter()
                    .filter(|&&r| r >= p.win_lo && r < p.win_hi)
                    .count()
            })
            .max()
            .unwrap_or(0)
    }

    /// Decode an entire stream syndrome with the two-layer parallel-window scheme. Returns the
    /// committed logical correction. Windows within each layer are decoded concurrently with rayon.
    pub fn decode_stream(&self, syndrome: &Syndrome) -> Result<Correction> {
        let nd = self.dem.detectors;
        let no = self.dem.observables;
        let mut lit = vec![false; nd];
        for &d in &syndrome.fired {
            if (d as usize) < nd {
                lit[d as usize] = true;
            }
        }

        let plans = self.window_plans();
        let mut logical = 0u64;

        // Two layers: even-indexed commit regions first, then odd-indexed against the updated
        // residual. Each layer is a parallel pure map (read `lit`) followed by a serial XOR apply.
        for parity in [0usize, 1] {
            let layer: Vec<(usize, WindowPlan)> = plans
                .iter()
                .copied()
                .enumerate()
                .filter(|(i, _)| i % 2 == parity)
                .collect();
            if layer.is_empty() {
                continue;
            }
            let results: Vec<Result<(Vec<usize>, u64)>> = layer
                .par_iter()
                .map(|(_, p)| self.decode_window(&lit, p))
                .collect();
            for r in results {
                let (toggles, obs) = r?;
                logical ^= obs;
                for d in toggles {
                    lit[d] ^= true;
                }
            }
        }

        let flips = (0..no).map(|o| (logical >> o) & 1 == 1).collect();
        Ok(Correction::new(flips))
    }

    /// Decode one window against a *read-only* snapshot of the residual `lit`. Returns the real
    /// detectors toggled by every committed edge (those touching the commit region) and the XOR of
    /// the committed edges' observable masks. Pure: it does not mutate shared state, so windows in a
    /// layer run concurrently and the caller applies the toggles by XOR afterwards.
    fn decode_window(&self, lit: &[bool], plan: &WindowPlan) -> Result<(Vec<usize>, u64)> {
        let nd = self.dem.detectors;
        let (win_lo, win_hi) = (plan.win_lo, plan.win_hi);
        let (commit_lo, commit_hi) = (plan.commit_lo, plan.commit_hi);

        // In-window (lit-able) detectors get local indices `0..n_active`.
        let mut local_of = vec![u32::MAX; nd];
        let mut globals: Vec<usize> = Vec::new();
        for (d, &r) in self.detector_round.iter().enumerate() {
            if r >= win_lo && r < win_hi {
                local_of[d] = globals.len() as u32;
                globals.push(d);
            }
        }
        let n_active = globals.len();

        // Out-of-window detectors touched by in-window mechanisms become temporal-sink nodes
        // (`n_active..n_local`), never lit, each given a free observable-less drain to the boundary.
        let mut temporal_of = vec![u32::MAX; nd];
        let mut n_local = n_active;
        let mut errors: Vec<DemError> = Vec::new();
        for e in &self.dem.errors {
            if !e.dets.iter().any(|&d| local_of[d as usize] != u32::MAX) {
                continue; // mechanism does not touch this window
            }
            let mut loc: Vec<u32> = Vec::with_capacity(e.dets.len());
            for &d in &e.dets {
                let l = local_of[d as usize];
                if l != u32::MAX {
                    loc.push(l);
                } else {
                    let t = if temporal_of[d as usize] != u32::MAX {
                        temporal_of[d as usize]
                    } else {
                        let idx = n_local as u32;
                        temporal_of[d as usize] = idx;
                        n_local += 1;
                        idx
                    };
                    loc.push(t);
                }
            }
            errors.push(DemError::new(e.prob, loc, e.obs.clone()));
        }
        for t in n_active..n_local {
            errors.push(DemError::new(0.5, vec![t as u32], vec![]));
        }

        let win_dem = DetectorErrorModel {
            detectors: n_local,
            observables: self.dem.observables,
            errors,
        };

        let fired: Vec<u32> = (0..n_active as u32)
            .filter(|&l| lit[globals[l as usize]])
            .collect();
        let win_syn = Syndrome::new(n_local, fired);

        let graph = MatchingGraph::from_dem(&win_dem)?;
        let dec = UnionFindDecoder::from_graph(&graph).weighted(self.weighted);
        let (_corr, chosen) = dec.decode_edges(&win_syn);

        let mut logical = 0u64;
        let mut toggles: Vec<usize> = Vec::new();
        for &e in &chosen {
            let ed = &graph.edges()[e];
            let mut ends: [usize; 2] = [usize::MAX, usize::MAX];
            let mut touches = false;
            for (slot, ep) in [ed.a, ed.b].into_iter().enumerate() {
                if ep < n_active {
                    let g = globals[ep];
                    ends[slot] = g;
                    let r = self.detector_round[g];
                    if r >= commit_lo && r < commit_hi {
                        touches = true;
                    }
                }
            }
            if touches {
                for &o in &ed.observables {
                    if o < 64 {
                        logical ^= 1u64 << o;
                    }
                }
                for g in ends {
                    if g != usize::MAX {
                        toggles.push(g);
                    }
                }
            }
        }
        Ok((toggles, logical))
    }

    /// Debug: number of detectors still lit after the full parallel decode (0 ⇔ the committed
    /// correction reproduces the whole syndrome). Used by tests to catch seam bugs.
    #[doc(hidden)]
    pub fn residual_after_decode(&self, syndrome: &Syndrome) -> usize {
        let nd = self.dem.detectors;
        let mut lit = vec![false; nd];
        for &d in &syndrome.fired {
            if (d as usize) < nd {
                lit[d as usize] = true;
            }
        }
        let plans = self.window_plans();
        for parity in [0usize, 1] {
            let layer: Vec<WindowPlan> = plans
                .iter()
                .copied()
                .enumerate()
                .filter(|(i, _)| i % 2 == parity)
                .map(|(_, p)| p)
                .collect();
            for p in &layer {
                if let Ok((toggles, _obs)) = self.decode_window(&lit, p) {
                    for d in toggles {
                        lit[d] ^= true;
                    }
                }
            }
        }
        lit.iter().filter(|&&x| x).count()
    }
}

impl crate::decoder::Decoder for ParallelWindowDecoder {
    /// Decode a full-stream syndrome via two-layer parallel windows. Surface-code memory DEMs are
    /// graphlike, so the per-window Union-Find decode never fails; a (non-graphlike) error degrades
    /// to no flip rather than panicking in this infallible interface.
    fn decode(&self, syndrome: &Syndrome) -> Correction {
        self.decode_stream(syndrome)
            .unwrap_or_else(|_| Correction::none(self.dem.observables))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::build_dem;
    use crate::decoder::Decoder;
    use crate::surface::SurfaceCode;

    fn stream(d: usize, rounds: usize, p: f64) -> (DetectorErrorModel, Vec<usize>) {
        let exp = SurfaceCode::new(d).memory_z_experiment(rounds);
        let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();
        (dem, exp.detector_rounds())
    }

    /// One window spanning the whole stream is exactly a batch UF decode.
    #[test]
    fn single_window_equals_batch() {
        let (dem, rounds) = stream(3, 8, 0.05);
        let num_slices = rounds.iter().copied().max().unwrap() + 1;
        // commit == whole stream, buffer irrelevant ⇒ exactly one window.
        let pw = ParallelWindowDecoder::new(dem.clone(), rounds, num_slices, 0);
        assert_eq!(pw.num_windows(), 1);
        let batch = UnionFindDecoder::new(&dem).unwrap();

        let mut z: u64 = 0x1234_5678;
        let mut next = || {
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            z
        };
        for _ in 0..500 {
            let bits: Vec<bool> = (0..dem.detectors).map(|_| next() & 7 == 0).collect();
            let syn = Syndrome::from_bits(&bits);
            assert_eq!(
                pw.decode_stream(&syn).unwrap(),
                batch.decode(&syn),
                "single window must equal batch"
            );
        }
    }

    /// The committed correction reproduces the whole syndrome (residual clears) for adequately
    /// buffered windows — catches seam bugs cheaply without a Monte-Carlo run.
    #[test]
    fn parallel_decode_is_valid() {
        let d = 3;
        let (dem, rounds) = stream(d, 12, 0.04);
        let pw = ParallelWindowDecoder::new(dem.clone(), rounds, d, d);
        assert!(pw.num_windows() >= 3, "want several windows in two layers");

        let mut z: u64 = 0xDEAD_BEEF;
        let mut next = || {
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            z
        };
        for _ in 0..300 {
            let bits: Vec<bool> = (0..dem.detectors).map(|_| next() & 7 == 0).collect();
            let syn = Syndrome::from_bits(&bits);
            assert_eq!(
                pw.residual_after_decode(&syn),
                0,
                "committed correction must reproduce the syndrome"
            );
        }
    }

    /// Window tiling: commit regions tile the stream, even/odd split into the two layers, and the
    /// per-window working set is bounded by the window size independent of stream length.
    #[test]
    fn tiling_and_bounded_working_set() {
        let d = 3;
        let (dem10, r10) = stream(d, 10, 0.03);
        let (dem40, r40) = stream(d, 40, 0.03);
        let pw10 = ParallelWindowDecoder::new(dem10, r10, d, d);
        let pw40 = ParallelWindowDecoder::new(dem40, r40, d, d);
        // Same window geometry ⇒ same per-window detector bound regardless of total rounds.
        assert_eq!(pw10.max_window_detectors(), pw40.max_window_detectors());

        // Commit regions tile [0, num_slices) with no gaps or overlaps.
        let plans = pw40.window_plans();
        assert_eq!(plans[0].commit_lo, 0);
        for w in plans.windows(2) {
            assert_eq!(w[0].commit_hi, w[1].commit_lo, "commit regions must tile");
        }
        assert_eq!(plans.last().unwrap().commit_hi, pw40.num_slices);
    }
}
