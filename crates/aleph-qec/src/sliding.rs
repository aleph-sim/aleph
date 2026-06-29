//! [`SlidingWindowDecoder`] — real-time **streaming** decoding of a continuous syndrome stream in
//! overlapping time windows (Q4-01).
//!
//! Real devices produce syndromes forever; you cannot wait for the end of the experiment to decode.
//! The sliding-window approach (Dennis et al.; Skoric et al. / Tan et al.) decodes a window of `W`
//! consecutive rounds, **commits** the correction for the first `C < W` rounds (the *commit region*),
//! then slides forward by `C`. The trailing `W − C` rounds (the *buffer*) give the commit region
//! enough future context that its correction matches what a full-batch decode would have chosen — the
//! seam between windows is the hard part, and the buffer is what tames it.
//!
//! # Forward windowing (residual carried across seams)
//!
//! A running **residual** syndrome is kept. Each window decodes the residual restricted to its
//! rounds. Mechanisms reaching out of the window are cut at an **artificial temporal boundary** that
//! is kept *distinct from the real spatial boundary*: every out-of-window detector becomes its own
//! never-lit temporal-sink node with a free, observable-less drain to the boundary. This separation
//! matters because in a memory-Z DEM the observable-flipping mechanisms are *spatial* detector↔
//! boundary edges; collapsing a time-cut measurement edge onto the same boundary would merge it with
//! a real observable edge and make a harmless "carry forward in time" spuriously flip the logical
//! observable. Routing time cuts through separate sinks avoids that.
//!
//! A correction edge that **touches the commit region** (has a real detector with round `<
//! commit_hi`) is *committed*: its observable is XORed into the running logical correction, and the
//! correction is **applied to the residual** (its real detectors toggled). Applying the committed
//! corrections clears every commit-region defect and leaves the seam detectors in a consistent state,
//! so the next window — decoding the *updated* residual — continues correctly. This carry is what
//! makes the per-window decodes compose into one valid global correction; decoding windows
//! independently does not. Buffer-internal defects stay in the residual and are re-decoded next
//! window with fresh future context; the only error is committing a seam-crossing correction with
//! limited lookahead, which vanishes as the buffer `W − C` grows. The commit regions
//! `[0,C), [C,2C), …` tile the stream; the final window commits everything to the end.
//!
//! With `W` equal to the whole stream this is a single batch decode; as `W` shrinks the logical-error
//! rate approaches the batch rate from above, reaching it within CI once the buffer `W − C ≳ d`
//! (validated in `tests/sliding_window.rs`). The working set is `O(W)` — independent of stream length
//! — so an unbounded stream decodes in bounded memory.

use crate::dem::{DemError, DetectorErrorModel};
use crate::error::Result;
use crate::matching::MatchingGraph;
use crate::syndrome::{Correction, Syndrome};
use crate::union_find::UnionFindDecoder;

/// A sliding-window streaming decoder over a fixed [`DetectorErrorModel`] with a per-detector time
/// (round) coordinate. Wraps the Union-Find decoder as the per-window base decoder.
#[derive(Clone, Debug)]
pub struct SlidingWindowDecoder {
    dem: DetectorErrorModel,
    /// Round (time index) of each detector; `detector_round.len() == dem.detectors`.
    detector_round: Vec<usize>,
    /// Number of time slices (max round + 1).
    num_slices: usize,
    /// Window length `W` (rounds).
    window: usize,
    /// Commit-region length `C` (rounds), `1 ≤ C ≤ W`.
    commit: usize,
    /// Whether the per-window Union-Find decode uses weighted growth (Q2-02).
    weighted: bool,
}

/// A single window's matching problem, exported for reuse (software decode + RTL window-graph gen).
#[derive(Clone, Debug)]
pub struct WindowExport {
    /// Window DEM: detectors `0..n_active` are in-window (lit-able), `n_active..dem.detectors` are
    /// temporal-sink nodes (never lit), and `dem.detectors` is the spatial boundary.
    pub dem: DetectorErrorModel,
    /// Number of in-window (lit-able) detectors.
    pub n_active: usize,
    /// `globals[l]` = global detector id of active local detector `l` (`l < n_active`).
    pub globals: Vec<usize>,
}

impl SlidingWindowDecoder {
    /// Build a sliding-window decoder. `detector_round[d]` is the time coordinate of detector `d`
    /// (e.g. from [`MemoryExperiment::detector_rounds`](crate::MemoryExperiment::detector_rounds)).
    ///
    /// # Panics
    /// If `detector_round.len() != dem.detectors`, or `commit` is not in `1..=window`.
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
        Self {
            dem,
            detector_round,
            num_slices,
            window,
            commit,
            weighted: false,
        }
    }

    /// Switch the per-window base decoder to weighted (Q2-02) growth.
    pub fn weighted(mut self, yes: bool) -> Self {
        self.weighted = yes;
        self
    }

    /// Window length `W`.
    pub fn window(&self) -> usize {
        self.window
    }

    /// The largest number of detectors any single window spans — the bound on the per-window working
    /// set, which is `O(W)` and independent of the total stream length. Exposed so a bounded-memory
    /// test can assert it does not grow with the number of rounds.
    pub fn max_window_detectors(&self) -> usize {
        let mut best = 0;
        let mut s = 0;
        while s < self.num_slices {
            let win_hi = (s + self.window).min(self.num_slices);
            let n = self
                .detector_round
                .iter()
                .filter(|&&r| r >= s && r < win_hi)
                .count();
            best = best.max(n);
            s += self.commit;
        }
        best
    }

    /// Decode an entire stream syndrome by sliding the window across the rounds, committing each
    /// window's commit-region correction. Returns the committed logical correction.
    pub fn decode_stream(&self, syndrome: &Syndrome) -> Result<Correction> {
        let nd = self.dem.detectors;
        let no = self.dem.observables;
        let mut lit = vec![false; nd];
        for &d in &syndrome.fired {
            if (d as usize) < nd {
                lit[d as usize] = true;
            }
        }

        let mut logical = 0u64;
        let mut s = 0usize;
        while s < self.num_slices {
            let win_hi = (s + self.window).min(self.num_slices); // exclusive
            let last = win_hi >= self.num_slices;
            let commit_hi = if last {
                self.num_slices
            } else {
                s + self.commit
            };
            logical ^= self.decode_window(&mut lit, s, win_hi, commit_hi, true)?;
            s += self.commit;
        }

        let flips = (0..no).map(|o| (logical >> o) & 1 == 1).collect();
        Ok(Correction::new(flips))
    }

    /// Decode the window covering rounds `[s, win_hi)` against the residual `lit`, committing every
    /// chosen correction edge that touches the commit region `[s, commit_hi)`: apply it to `lit`
    /// (toggle its real detectors) and, if `accumulate_obs`, return the XOR of the committed edges'
    /// observable masks. Out-of-window detectors are routed to per-detector temporal-sink nodes
    /// (distinct from the spatial boundary, free obs-less drains) so a time cut never flips the
    /// logical observable.
    /// Build the per-window DEM for the window covering rounds `[s, win_hi)`: in-window detectors get
    /// local indices `0..n_active` (lit-able); out-of-window detectors touched by in-window mechanisms
    /// become temporal-sink nodes `n_active..n_local` (never lit, each a free observable-less drain to
    /// the boundary). Returns the DEM plus the local→global map for the active detectors. This is the
    /// single source of truth for both the software decode and the exported RTL window graph (Q6-20).
    pub fn window_dem(&self, s: usize, win_hi: usize) -> Result<WindowExport> {
        let nd = self.dem.detectors;

        let mut local_of = vec![u32::MAX; nd];
        let mut globals: Vec<usize> = Vec::new();
        for (d, &r) in self.detector_round.iter().enumerate() {
            if r >= s && r < win_hi {
                local_of[d] = globals.len() as u32;
                globals.push(d);
            }
        }
        let n_active = globals.len();

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
        // Free drains: each temporal node → boundary (weight ~0, no observable).
        for t in n_active..n_local {
            errors.push(DemError::new(0.5, vec![t as u32], vec![]));
        }

        Ok(WindowExport {
            dem: DetectorErrorModel {
                detectors: n_local,
                observables: self.dem.observables,
                errors,
            },
            n_active,
            globals,
        })
    }

    fn decode_window(
        &self,
        lit: &mut [bool],
        s: usize,
        win_hi: usize,
        commit_hi: usize,
        accumulate_obs: bool,
    ) -> Result<u64> {
        let WindowExport {
            dem: win_dem,
            n_active,
            globals,
        } = self.window_dem(s, win_hi)?;

        // Window syndrome: only active (in-window) detectors can be lit.
        let fired: Vec<u32> = (0..n_active as u32)
            .filter(|&l| lit[globals[l as usize]])
            .collect();
        let win_syn = Syndrome::new(win_dem.detectors, fired);

        let graph = MatchingGraph::from_dem(&win_dem)?;
        let dec = UnionFindDecoder::from_graph(&graph).weighted(self.weighted);
        let (_corr, chosen) = dec.decode_edges(&win_syn);

        let mut logical = 0u64;
        for &e in &chosen {
            let ed = &graph.edges()[e];
            // Real (in-window) endpoints of this edge and whether any sits in the commit region.
            let mut ends: [usize; 2] = [usize::MAX, usize::MAX];
            let mut touches = false;
            for (slot, ep) in [ed.a, ed.b].into_iter().enumerate() {
                if ep < n_active {
                    let g = globals[ep];
                    ends[slot] = g;
                    if self.detector_round[g] < commit_hi {
                        touches = true;
                    }
                }
            }
            if touches {
                if accumulate_obs {
                    for &o in &ed.observables {
                        if o < 64 {
                            logical ^= 1u64 << o;
                        }
                    }
                }
                for g in ends {
                    if g != usize::MAX {
                        lit[g] ^= true;
                    }
                }
            }
        }
        Ok(logical)
    }

    /// Debug: number of detectors still lit after the full sliding decode (0 ⇔ the committed
    /// correction is valid, i.e. reproduces the whole syndrome). Used by tests to catch seam bugs.
    #[doc(hidden)]
    pub fn residual_after_decode(&self, syndrome: &Syndrome) -> usize {
        let nd = self.dem.detectors;
        let mut lit = vec![false; nd];
        for &d in &syndrome.fired {
            if (d as usize) < nd {
                lit[d as usize] = true;
            }
        }
        let mut s = 0usize;
        while s < self.num_slices {
            let win_hi = (s + self.window).min(self.num_slices);
            let last = win_hi >= self.num_slices;
            let commit_hi = if last {
                self.num_slices
            } else {
                s + self.commit
            };
            let _ = self.decode_window(&mut lit, s, win_hi, commit_hi, false);
            s += self.commit;
        }
        lit.iter().filter(|&&x| x).count()
    }
}

impl crate::decoder::Decoder for SlidingWindowDecoder {
    /// Decode a full-stream syndrome via sliding windows. Surface-code memory DEMs are graphlike, so
    /// the per-window Union-Find decode never fails; a (non-graphlike) error degrades to no flip
    /// rather than panicking in this infallible interface.
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

    /// A long memory-Z stream DEM + its detector rounds.
    fn stream(d: usize, rounds: usize, p: f64) -> (DetectorErrorModel, Vec<usize>) {
        let exp = SurfaceCode::new(d).memory_z_experiment(rounds);
        let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(p, p)).unwrap();
        (dem, exp.detector_rounds())
    }

    /// With `W` equal to the whole stream, the sliding decoder is exactly a batch UF decode.
    #[test]
    fn full_window_equals_batch() {
        let (dem, rounds) = stream(3, 8, 0.05);
        let num_slices = rounds.iter().copied().max().unwrap() + 1;
        let sw = SlidingWindowDecoder::new(dem.clone(), rounds, num_slices, num_slices);
        let batch = UnionFindDecoder::new(&dem).unwrap();

        let mut z: u64 = 0xABCD;
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
                sw.decode_stream(&syn).unwrap(),
                batch.decode(&syn),
                "full window must equal batch"
            );
        }
    }

    /// The per-window working set is bounded by `W` and does not grow with stream length.
    #[test]
    fn working_set_is_bounded() {
        let d = 3;
        let (dem10, r10) = stream(d, 10, 0.03);
        let (dem40, r40) = stream(d, 40, 0.03);
        let sw10 = SlidingWindowDecoder::new(dem10, r10, 5, 2);
        let sw40 = SlidingWindowDecoder::new(dem40, r40, 5, 2);
        // Same window size ⇒ same per-window detector bound regardless of total rounds.
        assert_eq!(sw10.max_window_detectors(), sw40.max_window_detectors());
    }
}
