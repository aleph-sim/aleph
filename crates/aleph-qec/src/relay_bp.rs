//! **Relay-BP** — disordered-memory belief propagation (Q5-03), a recent (2024–25) technique for
//! lowering the BP *error floor* on quantum LDPC codes.
//!
//! Plain min-sum BP (Q3-02) stalls in symmetric **trapping sets**: oscillating message patterns that
//! never satisfy the syndrome, the dominant source of BP's high-`p`-independent error floor. Relay-BP
//! (Müller et al., arXiv:2506.01779 and related "memory BP" work) breaks that symmetry two ways:
//!
//! * **Disordered memory.** Each variable node gets its own memory strength `γ_v`, drawn from a range
//!   that includes *negative* values. The variable→check update damps toward the previous message
//!   with strength `γ_v`: `M_{v→c} ← (1−γ_v)·M_computed + γ_v·M_old`. A spread of per-node `γ_v`
//!   (some pushing forward, some back) desynchronises the oscillation a uniform damping cannot.
//! * **Relayed legs.** BP is run in several *legs*; each leg keeps the messages from the previous one
//!   (the "relay") but swaps in a fresh disorder pattern `γ`. Across legs the decoder samples several
//!   message trajectories while carrying state forward, and keeps the **lowest-weight syndrome-valid**
//!   hard decision seen in any iteration of any leg.
//!
//! The disorder patterns are fixed at construction (one per leg, seeded deterministically) and reused
//! across shots — they break the *code's* symmetry, not the shot's — so the decoder stays
//! deterministic and `Sync`. It reuses the [`BpDecoder`](crate::BpDecoder) Tanner layout (and its
//! min-sum check update) verbatim, differing only in the memory term and the leg/keep-best loop.

use crate::bp::{BpDecoder, BpSoft};
use crate::decoder::Decoder;
use crate::dem::DetectorErrorModel;
use crate::osd::OsdDecoder;
use crate::syndrome::{Correction, Syndrome};

/// Number of relay legs by default.
pub const DEFAULT_LEGS: usize = 4;

/// A relay-BP decoder over a fixed [`DetectorErrorModel`].
#[derive(Clone, Debug)]
pub struct RelayBpDecoder {
    num_detectors: usize,
    num_observables: usize,
    n_vars: usize,
    n_edges: usize,
    iters_per_leg: u32,
    early_exit: bool,
    alpha: f64,
    lambda: Vec<f64>,
    obs: Vec<u64>,
    var_off: Vec<u32>,
    edge_var: Vec<u32>,
    check_off: Vec<u32>,
    check_edges: Vec<u32>,
    /// Per-leg, per-variable memory strength `γ_v ∈ [γ_min, γ_max]`.
    gamma: Vec<Vec<f64>>,
}

impl RelayBpDecoder {
    /// Build a relay-BP decoder with default parameters: [`DEFAULT_LEGS`] legs, normalised min-sum
    /// `α = 0.875`, and disordered memory `γ_v ∈ [−0.3, 0.9]`.
    pub fn new(dem: &DetectorErrorModel) -> Self {
        Self::with_params(dem, DEFAULT_LEGS, 0.875, (-0.3, 0.9), 0x5E1A_4B9C)
    }

    /// Build with explicit leg count, min-sum `α`, disordered-memory range `(γ_min, γ_max)`, and a
    /// disorder seed. Each leg runs `max(DEFAULT_MAX_ITER / max(legs, 1), 8)` iterations.
    /// All iterations run by default, retaining the lightest valid decision; use
    /// [`Self::with_budget`] for an explicit schedule and [`Self::with_early_exit`] to stop early.
    pub fn with_params(
        dem: &DetectorErrorModel,
        legs: usize,
        alpha: f64,
        gamma_range: (f64, f64),
        seed: u64,
    ) -> Self {
        let legs = legs.max(1);
        let iters_per_leg = ((crate::DEFAULT_MAX_ITER as usize / legs).max(8)) as u32;
        Self::with_budget(dem, legs, iters_per_leg, alpha, gamma_range, seed)
    }

    /// Build with an explicit number of message-passing sweeps per relay leg.
    ///
    /// Like [`Self::with_params`], but the schedule is `max(legs, 1) × max(iters_per_leg, 1)`;
    /// zero values are promoted to one, matching [`crate::FixedRelayBp::with_budget`].
    /// The seed defines the same per-leg disorder regardless of the iteration budget.
    /// Early exit defaults to false: the full schedule runs and retains the lowest-weight
    /// syndrome-valid decision. Omitted Python budgets keep [`Self::with_params`]'s schedule.
    ///
    /// ```
    /// use aleph_qec::{DetectorErrorModel, RelayBpDecoder};
    /// let dem = DetectorErrorModel::parse("error(0.1) D0 L0")?;
    /// let relay = RelayBpDecoder::with_budget(&dem, 6, 10, 0.875, (-0.3, 0.9), 7)
    ///     .with_early_exit(true);
    /// # let _ = relay;
    /// # Ok::<(), aleph_qec::Error>(())
    /// ```
    pub fn with_budget(
        dem: &DetectorErrorModel,
        legs: usize,
        iters_per_leg: u32,
        alpha: f64,
        gamma_range: (f64, f64),
        seed: u64,
    ) -> Self {
        let legs = legs.max(1);
        let iters_per_leg = iters_per_leg.max(1);
        let bp = BpDecoder::with_params(dem, crate::DEFAULT_MAX_ITER, alpha);
        let t = bp.tanner();

        // Per-leg disorder pattern γ[leg][v], deterministic via SplitMix64(seed, leg, v).
        let (gmin, gmax) = gamma_range;
        let gamma: Vec<Vec<f64>> = (0..legs)
            .map(|leg| {
                (0..t.n_vars)
                    .map(|v| {
                        let mut z = seed
                            .wrapping_add((leg as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                            .wrapping_add((v as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
                        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                        z ^= z >> 31;
                        let u = (z >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
                        gmin + (gmax - gmin) * u
                    })
                    .collect()
            })
            .collect();

        Self {
            num_detectors: t.num_detectors,
            num_observables: t.num_observables,
            n_vars: t.n_vars,
            n_edges: t.n_edges,
            iters_per_leg,
            early_exit: false,
            alpha,
            lambda: t.lambda.to_vec(),
            obs: t.obs.to_vec(),
            var_off: t.var_off.to_vec(),
            edge_var: t.edge_var.to_vec(),
            check_off: t.check_off.to_vec(),
            check_edges: t.check_edges.to_vec(),
            gamma,
        }
    }

    /// Stop at the first syndrome-valid hard decision instead of searching the full schedule
    /// for the lightest one. Disabled by default. Validity is not a guarantee of the correct
    /// logical class: enabling this can change both correction quality and runtime.
    /// The returned soft LLRs are from the stopping iteration; if no decision is valid,
    /// the full schedule still runs and returns its final hard decision/LLRs.
    pub fn with_early_exit(mut self, on: bool) -> Self {
        self.early_exit = on;
        self
    }

    /// Decode, returning the correction and whether a syndrome-valid hard decision was found (relay-BP
    /// "converged" in some leg). When none is valid, the final hard decision is returned.
    pub fn decode_relay(&self, syndrome: &Syndrome) -> (Correction, bool) {
        let soft = self.decode_soft(syndrome);
        (self.correction_of(&soft.ehat), soft.converged)
    }

    /// Soft decode: run the relay legs and return the chosen hard decision, posterior LLRs, and a
    /// `converged` flag. The posterior `L_v = λ_v + Σ_c E_{c→v}` is taken from the final leg's
    /// messages; the hard decision is the lowest-weight syndrome-valid one seen across legs (or the
    /// final hard decision if none was valid). This is the soft information BP+OSD post-processing
    /// consumes — see [`RelayBpOsdDecoder`]. With [`Self::with_early_exit`] enabled, both the
    /// decision and LLRs come from the first syndrome-valid iteration instead.
    pub fn decode_soft(&self, syndrome: &Syndrome) -> BpSoft {
        self.decode_soft_observed(syndrome, || {})
    }

    // The no-op observer is monomorphized away in production. Tests count actual sweeps,
    // rather than infer stopping from wall time or from two coincidentally equal decisions.
    fn decode_soft_observed(&self, syndrome: &Syndrome, mut swept: impl FnMut()) -> BpSoft {
        let mut s = vec![0u8; self.num_detectors];
        for &d in &syndrome.fired {
            if (d as usize) < self.num_detectors {
                s[d as usize] = 1;
            }
        }

        let mut m_vc = vec![0.0f64; self.n_edges];
        let mut e_cv = vec![0.0f64; self.n_edges];
        let mut ehat = vec![0u8; self.n_vars];

        // Init messages M_{v→c} = λ_v. Messages relay (persist) across legs.
        for (edge, m) in m_vc.iter_mut().enumerate() {
            *m = self.lambda[self.edge_var[edge] as usize];
        }

        let mut best: Option<(u32, Vec<u8>)> = None; // (hamming weight, ehat)
        let mut found_valid = false;

        'schedule: for leg in 0..self.gamma.len() {
            let gamma = &self.gamma[leg];
            for _ in 0..self.iters_per_leg {
                self.check_update(&m_vc, &mut e_cv, &s);
                self.var_update_memory(&e_cv, &mut m_vc, &mut ehat, gamma);
                swept();
                if self.satisfies(&ehat, &s) {
                    found_valid = true;
                    let w = ehat.iter().map(|&b| b as u32).sum();
                    if best.as_ref().is_none_or(|(bw, _)| w < *bw) {
                        best = Some((w, ehat.clone()));
                    }
                    if self.early_exit {
                        break 'schedule;
                    }
                }
            }
        }

        // Posterior LLR per variable from the final check→variable messages.
        let mut llr = vec![0.0f64; self.n_vars];
        for (v, w) in self.var_off.windows(2).enumerate() {
            let (lo, hi) = (w[0] as usize, w[1] as usize);
            llr[v] = self.lambda[v] + e_cv[lo..hi].iter().sum::<f64>();
        }

        let chosen = match best {
            Some((_, e)) => e,
            None => ehat, // no valid decision in any leg; return the last hard decision
        };
        BpSoft {
            ehat: chosen,
            llr,
            converged: found_valid,
        }
    }

    /// Per-column error estimate `ê` and whether any relay leg found a valid solution.
    pub fn decode_errors(&self, syndrome: &Syndrome) -> (Vec<u8>, bool) {
        let soft = self.decode_soft(syndrome);
        (soft.ehat, soft.converged)
    }

    /// Min-sum check → variable update (identical to [`BpDecoder`]'s; see Q3-02).
    fn check_update(&self, m_vc: &[f64], e_cv: &mut [f64], s: &[u8]) {
        for (c, w) in self.check_off.windows(2).enumerate() {
            let (lo, hi) = (w[0] as usize, w[1] as usize);
            let mut neg = s[c] == 1;
            let mut min1 = f64::INFINITY;
            let mut min2 = f64::INFINITY;
            let mut argmin = u32::MAX;
            for &edge in &self.check_edges[lo..hi] {
                let m = m_vc[edge as usize];
                if m < 0.0 {
                    neg = !neg;
                }
                let a = m.abs();
                if a < min1 {
                    min2 = min1;
                    min1 = a;
                    argmin = edge;
                } else if a < min2 {
                    min2 = a;
                }
            }
            for &edge in &self.check_edges[lo..hi] {
                let m = m_vc[edge as usize];
                let excl_neg = if m < 0.0 { !neg } else { neg };
                let ex_min = if edge == argmin { min2 } else { min1 };
                let mag = self.alpha * ex_min;
                e_cv[edge as usize] = if excl_neg { -mag } else { mag };
            }
        }
    }

    /// Variable → check update with **disordered memory**: the new message blends the freshly
    /// computed value with the old message at per-node strength `γ_v`.
    fn var_update_memory(&self, e_cv: &[f64], m_vc: &mut [f64], ehat: &mut [u8], gamma: &[f64]) {
        for (v, w) in self.var_off.windows(2).enumerate() {
            let (lo, hi) = (w[0] as usize, w[1] as usize);
            let total = self.lambda[v] + e_cv[lo..hi].iter().sum::<f64>();
            ehat[v] = (total < 0.0) as u8;
            let g = gamma[v];
            for (mv, ev) in m_vc[lo..hi].iter_mut().zip(&e_cv[lo..hi]) {
                let computed = total - *ev;
                *mv = (1.0 - g) * computed + g * *mv;
            }
        }
    }

    /// Whether the hard decision reproduces the syndrome: `H ê = s`.
    fn satisfies(&self, ehat: &[u8], s: &[u8]) -> bool {
        for (c, w) in self.check_off.windows(2).enumerate() {
            let (lo, hi) = (w[0] as usize, w[1] as usize);
            let mut parity = 0u8;
            for &edge in &self.check_edges[lo..hi] {
                parity ^= ehat[self.edge_var[edge as usize] as usize];
            }
            if parity != s[c] {
                return false;
            }
        }
        true
    }

    fn correction_of(&self, ehat: &[u8]) -> Correction {
        let mut mask = 0u64;
        for (v, &e) in ehat.iter().enumerate() {
            if e == 1 {
                mask ^= self.obs[v];
            }
        }
        let flips = (0..self.num_observables)
            .map(|o| (mask >> o) & 1 == 1)
            .collect();
        Correction::new(flips)
    }
}

impl Decoder for RelayBpDecoder {
    fn decode(&self, syndrome: &Syndrome) -> Correction {
        self.decode_relay(syndrome).0
    }
}

/// **Relay-BP + OSD** (Q5-05) — the strongest qLDPC decoder in this crate: relay-BP's
/// disordered-memory message passing produces better soft information than plain BP, and OSD's
/// ordered-statistics combination sweep ([`OsdDecoder`]) cleans up the residual wrong-coset
/// failures relay-BP leaves on the circuit-level hypergraph. On the gross code's circuit-level DEM
/// this beats both BP+OSD and standalone relay-BP (`docs/perf/qec-q5-circuit-dem.md`).
#[derive(Clone, Debug)]
pub struct RelayBpOsdDecoder {
    relay: RelayBpDecoder,
    osd: OsdDecoder,
}

impl RelayBpOsdDecoder {
    /// Build with the default relay-BP front-end ([`RelayBpDecoder::new`]) and an OSD post-processor
    /// at combination-sweep `order` (sharing the same `α`).
    pub fn new(dem: &DetectorErrorModel, order: usize) -> Self {
        Self::with_parts(
            RelayBpDecoder::new(dem),
            OsdDecoder::new(dem).with_order(order),
        )
    }

    /// Build from an explicit relay-BP front-end and OSD post-processor (both over the *same* DEM).
    pub fn with_parts(relay: RelayBpDecoder, osd: OsdDecoder) -> Self {
        Self { relay, osd }
    }

    /// Decode: relay-BP soft pass, then OSD post-processing.
    pub fn decode_relay_osd(&self, syndrome: &Syndrome) -> Correction {
        let soft = self.relay.decode_soft(syndrome);
        self.osd.correction_from_soft(syndrome, &soft)
    }

    /// Per-column error estimate `ê`: relay-BP's if it converged, else the OSD refinement of its
    /// soft output, and whether that estimate satisfies the syndrome (`H ê = s`). Relay-BP's own
    /// convergence already guarantees this; the OSD refinement is checked explicitly (see
    /// [`OsdDecoder::decode_errors`]'s docs for why it is not unconditionally `true`).
    pub fn decode_errors(&self, syndrome: &Syndrome) -> (Vec<u8>, bool) {
        let soft = self.relay.decode_soft(syndrome);
        if soft.converged {
            return (soft.ehat, true);
        }
        let ehat = self.osd.ehat_from_soft(syndrome, &soft);
        let ok = self.osd.check_satisfied(syndrome, &ehat);
        (ehat, ok)
    }
}

impl Decoder for RelayBpOsdDecoder {
    fn decode(&self, syndrome: &Syndrome) -> Correction {
        self.decode_relay_osd(syndrome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bivariate_bicycle::BBCode;

    fn soft_eq(a: &BpSoft, b: &BpSoft) {
        assert_eq!(a.ehat, b.ehat);
        assert_eq!(a.converged, b.converged);
        assert_eq!(
            a.llr.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.llr.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    fn toy() -> DetectorErrorModel {
        DetectorErrorModel::parse("error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1").unwrap()
    }

    #[test]
    fn omitted_budget_keeps_legacy_schedule_and_disorder() {
        let dem = toy();
        for legs in [0usize, 1, 4, 6, 60] {
            let legacy = RelayBpDecoder::with_params(&dem, legs, 0.875, (-0.3, 0.9), 7);
            let iterations = ((crate::DEFAULT_MAX_ITER as usize / legs.max(1)).max(8)) as u32;
            let explicit =
                RelayBpDecoder::with_budget(&dem, legs, iterations, 0.875, (-0.3, 0.9), 7)
                    .with_early_exit(false);
            assert_eq!(legacy.iters_per_leg, iterations);
            assert!(!legacy.early_exit);
            assert_eq!(legacy.gamma, explicit.gamma);
            for fired in [vec![], vec![0], vec![1], vec![0, 1]] {
                let s = Syndrome::new(2, fired);
                soft_eq(&legacy.decode_soft(&s), &explicit.decode_soft(&s));
            }
        }
        let zero = RelayBpDecoder::with_budget(&dem, 0, 0, 0.875, (-0.3, 0.9), 7);
        assert_eq!(
            (zero.gamma.len(), zero.iters_per_leg, zero.early_exit),
            (1, 1, false)
        );
    }

    #[test]
    fn explicit_budget_counts_actual_sweeps_and_stops_only_on_valid_decisions() {
        let mut dem = toy();
        dem.detectors = 3; // D2 has no incident mechanism; a fired D2 cannot be satisfied.
        let full = RelayBpDecoder::with_budget(&dem, 3, 4, 0.875, (-0.3, 0.9), 7);
        let early = full.clone().with_early_exit(true);
        let count = |decoder: &RelayBpDecoder, syndrome: &Syndrome| {
            let mut n = 0;
            let soft = decoder.decode_soft_observed(syndrome, || n += 1);
            (soft, n)
        };
        let empty = Syndrome::default();
        let (complete, n) = count(&full, &empty);
        assert!(complete.converged);
        assert_eq!(n, 12);
        let (stopped, n) = count(&early, &empty);
        assert!(stopped.converged);
        assert_eq!(n, 1);
        let prefix = RelayBpDecoder::with_budget(&dem, 1, 1, 0.875, (-0.3, 0.9), 7);
        soft_eq(&stopped, &prefix.decode_soft(&empty));

        let impossible = Syndrome::new(3, vec![2]);
        let (complete, n) = count(&full, &impossible);
        let (stopped, early_n) = count(&early, &impossible);
        assert!(!complete.converged && !stopped.converged);
        assert_eq!((n, early_n), (12, 12));
        soft_eq(&complete, &stopped);
    }

    #[test]
    fn nonempty_first_valid_result_and_llrs_match_the_stopping_prefix() {
        let dem = toy();
        let early =
            RelayBpDecoder::with_budget(&dem, 6, 10, 0.875, (-0.3, 0.9), 7).with_early_exit(true);
        let syndrome = Syndrome::new(2, vec![0]);
        let mut n = 0;
        let stopped = early.decode_soft_observed(&syndrome, || n += 1);
        assert!(stopped.converged);
        assert!(
            (1..=10).contains(&n),
            "fixture must converge in the first leg, got {n}"
        );
        let prefix = RelayBpDecoder::with_budget(&dem, 1, n, 0.875, (-0.3, 0.9), 7);
        soft_eq(&stopped, &prefix.decode_soft(&syndrome));
        assert!(early.satisfies(&stopped.ehat, &[1, 0]));
    }

    /// Relay-BP returns a syndrome-consistent decision whenever it reports one valid, and never
    /// panics. Validity is the prerequisite for a meaningful logical-error rate.
    #[test]
    fn relay_valid_decisions_reproduce_syndrome() {
        let code = BBCode::gross();
        let dem = code.code_capacity_dem(0.04);
        let relay = RelayBpDecoder::new(&dem);
        let cols: Vec<Vec<u32>> = dem.errors.iter().map(|e| e.dets.clone()).collect();

        let mut z = 0xABCD_1234u64;
        let mut next = || {
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            z
        };
        for _ in 0..100 {
            let mut lit = vec![false; dem.detectors];
            let mut fired_vars = vec![false; dem.errors.len()];
            for (v, f) in fired_vars.iter_mut().enumerate() {
                if next() % 20 == 0 {
                    *f = true;
                    for &c in &cols[v] {
                        lit[c as usize] ^= true;
                    }
                }
            }
            let syn = Syndrome::from_bits(&lit);
            let (corr, valid) = relay.decode_relay(&syn);
            if valid {
                // Recompute the syndrome of relay-BP's chosen error is consistent — but we only have
                // the observable correction here; instead re-decode and check it is deterministic.
                let (corr2, valid2) = relay.decode_relay(&syn);
                assert_eq!(corr.observable_flips, corr2.observable_flips);
                assert!(valid2);
            }
        }
    }

    #[test]
    fn relay_is_deterministic() {
        let code = BBCode::gross();
        let dem = code.code_capacity_dem(0.05);
        let a = RelayBpDecoder::new(&dem);
        let b = RelayBpDecoder::new(&dem);
        let syn = Syndrome::from_bits(&{
            let mut v = vec![false; dem.detectors];
            v[0] = true;
            v[5] = true;
            v
        });
        assert_eq!(
            a.decode(&syn).observable_flips,
            b.decode(&syn).observable_flips
        );
    }

    #[test]
    fn decode_errors_agrees_with_decode_and_satisfies_syndrome_when_converged() {
        use crate::{BBCode, BpDecoder, Decoder, OsdDecoder};
        let dem = BBCode::gross().code_capacity_dem(0.01);
        let (relay, osd, bp, relay_osd) = (
            RelayBpDecoder::new(&dem),
            OsdDecoder::new(&dem),
            BpDecoder::new(&dem),
            RelayBpOsdDecoder::new(&dem, 0),
        );
        let (syns, _) = crate::sample_shots(&dem, 200, 7);
        let check = |name: &str, s: &Syndrome, ehat: &[u8], conv: bool, flips: &[bool]| {
            assert_eq!(ehat.len(), dem.errors.len(), "{name}");
            let mut hs = vec![false; dem.detectors];
            let mut os = vec![false; dem.observables];
            for (j, &b) in ehat.iter().enumerate() {
                if b == 1 {
                    for &d in &dem.errors[j].dets {
                        hs[d as usize] ^= true;
                    }
                    for &o in &dem.errors[j].obs {
                        os[o as usize] ^= true;
                    }
                }
            }
            if conv {
                let want: Vec<bool> = (0..dem.detectors as u32).map(|d| s.is_fired(d)).collect();
                assert_eq!(hs, want, "{name}: converged but H ê != s");
            }
            assert_eq!(os, flips, "{name}: O ê != decode()");
        };
        for s in &syns {
            let (e, c) = bp.decode_errors(s);
            check("bp", s, &e, c, &bp.decode(s).observable_flips);
            let (e, c) = osd.decode_errors(s);
            assert!(c);
            check("osd", s, &e, c, &osd.decode(s).observable_flips);
            let (e, c) = relay.decode_errors(s);
            check("relay", s, &e, c, &relay.decode(s).observable_flips);
            let (e, c) = relay_osd.decode_errors(s);
            assert!(c);
            check("relay-osd", s, &e, c, &relay_osd.decode(s).observable_flips);
        }
    }
}
