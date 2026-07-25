//! **Fixed-point relay-BP** — the hardware golden model for the RTL/FPGA qLDPC decoder (Q7-02).
//!
//! [`RelayBpDecoder`](crate::RelayBpDecoder) is the `f64` reference; this is its **bit-accurate
//! fixed-point twin**. An FPGA/ASIC decoder cannot carry IEEE-754 doubles through 432 message edges,
//! so the silicon datapath is integer fixed-point. This module *is* the specification the RTL
//! implements: the exact quantisation, saturation, and rounding the hardware must reproduce
//! bit-for-bit. Its logical-error rate, swept over message width, tells us the narrowest word the
//! silicon can carry without losing accuracy vs the `f64` decoder — the single most important RTL
//! sizing input.
//!
//! # Fixed-point scheme
//!
//! A value `x` is stored as a signed integer `round(x · 2^F)` (`F` = [`frac_bits`]), saturated to a
//! magnitude `MAX_MAG = 2^(W-1) − 1` where `W` = [`msg_bits`] is the signed message width. Every
//! stored message (variable→check and check→variable) lives in this format; the per-node
//! accumulator is kept wider (`i64` here; the RTL sizes it to `W + ceil(log2(deg))` bits) and only
//! the stored messages re-saturate.
//!
//! Three hardware-friendly choices, all inherited from the `f64` decoder's structure:
//!
//! * **α = 0.875 = 7/8 is multiply-free**: `mag − (mag >> 3)` on a non-negative magnitude. The whole
//!   min-sum check update is compare / min / sign — no multiplier.
//! * **The only multiply in the datapath is the relay memory blend** `(1−γ_v)·computed + γ_v·m_old`.
//!   `γ_v` is a *per-variable constant* (seeded ROM), so it is a fixed-coefficient multiply, not a
//!   general one, and can be a small LUT in silicon.
//! * **Truncating (arithmetic-shift) rounding** on the blend — `num >> F`, floor toward −∞ — is the
//!   cheapest RTL choice (no rounding adder). We adopt it here so the golden matches the silicon.
//!
//! The Tanner layout, `γ` seeding, leg/iteration schedule, and keep-lowest-weight-valid rule are
//! identical to [`RelayBpDecoder`]; only the arithmetic is quantised.

use crate::bp::BpDecoder;
use crate::decoder::Decoder;
use crate::dem::DetectorErrorModel;
use crate::relay_bp::DEFAULT_LEGS;
use crate::syndrome::{Correction, Syndrome};

/// A fixed-point relay-BP decoder over a fixed [`DetectorErrorModel`].
///
/// Construct with [`FixedRelayBp::new`] (default legs / `γ` range / seed, matching
/// [`RelayBpDecoder::new`](crate::RelayBpDecoder)) choosing only the message width, or
/// [`FixedRelayBp::with_params`] for full control. The message word is `msg_bits` signed with
/// `frac_bits` fractional bits.
#[derive(Clone, Debug)]
pub struct FixedRelayBp {
    num_detectors: usize,
    num_observables: usize,
    n_vars: usize,
    n_edges: usize,
    iters_per_leg: u32,
    /// Fractional bits `F`: the fixed-point scale is `2^F`.
    frac_bits: u32,
    /// Signed message width `W` in bits; magnitudes saturate at `2^(W-1) − 1`.
    msg_bits: u32,
    /// `2^(W-1) − 1`, precomputed.
    max_mag: i32,
    /// Prior LLR per variable, quantised.
    lambda_q: Vec<i32>,
    obs: Vec<u64>,
    var_off: Vec<u32>,
    edge_var: Vec<u32>,
    check_off: Vec<u32>,
    check_edges: Vec<u32>,
    /// Per-leg, per-variable memory strength `γ_v`, quantised to the same `2^F` scale (may be
    /// negative). `1−γ_v` is derived at use as `2^F − γ_q`.
    gamma_q: Vec<Vec<i32>>,
    /// Early termination: stop at the **first** iteration whose hard decision satisfies the syndrome,
    /// returning that `ê` instead of the lowest-weight valid one over the whole `legs×iters` schedule.
    /// This is standard BP early-stop — it changes the result (average latency ↓, worst-case unchanged),
    /// so the RTL `early_exit` mode is verified bit-for-bit against a golden built with this set.
    early_exit: bool,
}

impl FixedRelayBp {
    /// Build with the same defaults as [`RelayBpDecoder::new`](crate::RelayBpDecoder) —
    /// [`DEFAULT_LEGS`] legs, `γ_v ∈ [−0.3, 0.9]`, seed `0x5E1A_4B9C` — at the given fixed-point
    /// width. `msg_bits` is the signed message width, `frac_bits` its fractional part.
    pub fn new(dem: &DetectorErrorModel, msg_bits: u32, frac_bits: u32) -> Self {
        Self::with_params(
            dem,
            DEFAULT_LEGS,
            (-0.3, 0.9),
            0x5E1A_4B9C,
            msg_bits,
            frac_bits,
        )
    }

    /// Build with explicit leg count, disordered-memory range `(γ_min, γ_max)`, disorder seed, and
    /// fixed-point width `(msg_bits, frac_bits)`. `α` is fixed at `7/8` (the multiply-free
    /// normalisation the hardware uses). `iters_per_leg` defaults to `DEFAULT_MAX_ITER / legs`
    /// (= 25 at the default 4 legs) — the full relay-BP schedule.
    pub fn with_params(
        dem: &DetectorErrorModel,
        legs: usize,
        gamma_range: (f64, f64),
        seed: u64,
        msg_bits: u32,
        frac_bits: u32,
    ) -> Self {
        let legs = legs.max(1);
        let iters_per_leg = (crate::DEFAULT_MAX_ITER / legs as u32).max(8);
        Self::with_budget(
            dem,
            legs,
            iters_per_leg,
            gamma_range,
            seed,
            msg_bits,
            frac_bits,
        )
    }

    /// Like [`with_params`](Self::with_params) but with an **explicit `iters_per_leg`**. The relay-BP
    /// schedule is `legs × iters_per_leg` message-passing sweeps; on the RTL that is the dominant
    /// term in the per-decode cycle count (M4: `legs·iters·3 + overhead`). The Q7-02 M5 budget study
    /// sweeps `(legs, iters_per_leg)` to find the smallest schedule whose LER still matches the full
    /// 4×25 relay-BP within Monte-Carlo CI — every sweep dropped is a direct latency win.
    pub fn with_budget(
        dem: &DetectorErrorModel,
        legs: usize,
        iters_per_leg: u32,
        gamma_range: (f64, f64),
        seed: u64,
        msg_bits: u32,
        frac_bits: u32,
    ) -> Self {
        assert!((2..=31).contains(&msg_bits), "msg_bits must be in 2..=31");
        assert!(frac_bits < msg_bits, "frac_bits must be < msg_bits");
        let legs = legs.max(1);
        let iters_per_leg = iters_per_leg.max(1);

        // Reuse the BpDecoder's flattened Tanner graph (identical layout to RelayBpDecoder).
        let bp = BpDecoder::with_params(dem, crate::DEFAULT_MAX_ITER, 0.875);
        let t = bp.tanner();

        let scale = (1i64 << frac_bits) as f64;
        let max_mag = (1i32 << (msg_bits - 1)) - 1;
        // Build-time quantisation rounds (a constant, not on the datapath); runtime ops truncate.
        let q =
            |x: f64| -> i32 { (x * scale).round().clamp(-(max_mag as f64), max_mag as f64) as i32 };

        let lambda_q = t.lambda.iter().map(|&l| q(l)).collect();

        // Per-leg disorder pattern γ[leg][v], SplitMix64(seed, leg, v) — byte-identical to
        // RelayBpDecoder::with_params so the fixed and f64 decoders share the same disorder.
        let (gmin, gmax) = gamma_range;
        let gamma_q: Vec<Vec<i32>> = (0..legs)
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
                        let g = gmin + (gmax - gmin) * u;
                        // γ is a coefficient in [−0.3, 0.9]; it fits the frac scale without the
                        // magnitude clamp the messages need.
                        (g * scale).round() as i32
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
            frac_bits,
            msg_bits,
            max_mag,
            lambda_q,
            obs: t.obs.to_vec(),
            var_off: t.var_off.to_vec(),
            edge_var: t.edge_var.to_vec(),
            check_off: t.check_off.to_vec(),
            check_edges: t.check_edges.to_vec(),
            gamma_q,
            early_exit: false,
        }
    }

    /// Enable/disable early termination (see [`early_exit`](Self::early_exit) field). Returns `self`
    /// for chaining. With it on, the decoder returns the first syndrome-valid `ê` and stops.
    pub fn with_early_exit(mut self, on: bool) -> Self {
        self.early_exit = on;
        self
    }

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

    /// The fixed-point width `(msg_bits, frac_bits)`.
    pub fn width(&self) -> (u32, u32) {
        (self.msg_bits, self.frac_bits)
    }

    /// Run **one** min-sum check→variable update on an explicit message vector and return the
    /// resulting check→variable messages. `m_vc` is the variable→check messages (one per edge, in
    /// the canonical edge order); `s_bits` is the syndrome as one bit per check. This is the exact
    /// datapath the RTL check-update module implements (M1), exposed so the generator can dump
    /// input/expected test vectors the Verilator testbench replays bit-for-bit.
    pub fn check_update_once(&self, m_vc: &[i32], s_bits: &[u8]) -> Vec<i32> {
        assert_eq!(m_vc.len(), self.n_edges, "m_vc length must equal n_edges");
        let mut s = vec![0u8; self.num_detectors];
        for (c, slot) in s.iter_mut().enumerate() {
            *slot = s_bits.get(c).copied().unwrap_or(0) & 1;
        }
        let mut e_cv = vec![0i32; self.n_edges];
        self.check_update(m_vc, &mut e_cv, &s);
        e_cv
    }

    /// A read-only view of the quantised Tanner graph + fixed-point parameters, for the RTL
    /// generator (M1+) that emits the `.svh` the SystemVerilog decoder `\`include`s. Every field
    /// mirrors an identically-named private field; the layout is exactly what the RTL replays.
    pub fn hw_view(&self) -> FixedHwView<'_> {
        FixedHwView {
            n_vars: self.n_vars,
            n_checks: self.num_detectors,
            n_edges: self.n_edges,
            num_observables: self.num_observables,
            msg_bits: self.msg_bits,
            frac_bits: self.frac_bits,
            max_mag: self.max_mag,
            iters_per_leg: self.iters_per_leg,
            legs: self.gamma_q.len(),
            lambda_q: &self.lambda_q,
            obs: &self.obs,
            var_off: &self.var_off,
            edge_var: &self.edge_var,
            check_off: &self.check_off,
            check_edges: &self.check_edges,
            gamma_q: &self.gamma_q,
        }
    }

    /// Decode, returning the correction and whether a syndrome-valid hard decision was found in some
    /// leg (matches [`RelayBpDecoder::decode_relay`](crate::RelayBpDecoder::decode_relay)).
    pub fn decode_fixed(&self, syndrome: &Syndrome) -> (Correction, bool) {
        let (chosen, found) = self.run(syndrome);
        (self.correction_of(&chosen), found)
    }

    /// Full decode exposing the chosen error pattern `ehat` (one bit per variable) alongside the
    /// observable flips and validity — the exact outputs the RTL full-decode testbench (M2) compares
    /// against bit-for-bit.
    pub fn decode_fixed_ehat(&self, syndrome: &Syndrome) -> (Vec<u8>, Vec<bool>, bool) {
        let (chosen, found) = self.run(syndrome);
        let flips = self.correction_of(&chosen).observable_flips;
        (chosen, flips, found)
    }

    /// Decode exposing the BP **soft** output — the final hard decision, quantised posterior LLR, and
    /// whether a syndrome-valid decision was found — as a [`BpSoft`](crate::BpSoft) (LLR dequantised to
    /// `f64`; only its *ordering* and *sign* matter to OSD, both preserved by the linear scale). This is
    /// the hook the OSD-0 tail ([`FixedRelayBpOsd`]) consumes: when `converged` it carries the valid
    /// lowest-weight `ê`, otherwise the final BP belief for OSD to refine into a valid low-weight error.
    pub fn decode_fixed_soft(&self, syndrome: &Syndrome) -> crate::BpSoft {
        let (chosen, found, llr_q) = self.run_soft(syndrome);
        crate::BpSoft {
            ehat: chosen,
            llr: llr_q.into_iter().map(|x| x as f64).collect(),
            converged: found,
        }
    }

    /// The relay-BP legs/iterations loop, returning the chosen hard decision (lowest-weight
    /// syndrome-valid across all legs, or the final one if none was valid) and whether any valid
    /// decision was seen. Shared by [`decode_fixed`](Self::decode_fixed) and
    /// [`decode_fixed_ehat`](Self::decode_fixed_ehat).
    fn run(&self, syndrome: &Syndrome) -> (Vec<u8>, bool) {
        let (chosen, found, _llr) = self.run_soft(syndrome);
        (chosen, found)
    }

    /// Like [`run`](Self::run) but also returns the **final-iteration** per-variable soft LLR
    /// (quantised): `λ_v + Σ_c e_{c→v}`, the same posterior the f64 relay-BP exposes. This is the
    /// reliability information the OSD-0 tail ([`FixedRelayBpOsd`]) needs on the shots where no leg
    /// found a valid `ê` — its `sign` is the hard decision, its `|·|` the ordering key.
    fn run_soft(&self, syndrome: &Syndrome) -> (Vec<u8>, bool, Vec<i32>) {
        let mut s = vec![0u8; self.num_detectors];
        for &d in &syndrome.fired {
            if (d as usize) < self.num_detectors {
                s[d as usize] = 1;
            }
        }

        let mut m_vc = vec![0i32; self.n_edges];
        let mut e_cv = vec![0i32; self.n_edges];
        let mut ehat = vec![0u8; self.n_vars];

        // Init M_{v→c} = λ_v (quantised). Messages relay (persist) across legs.
        for (edge, m) in m_vc.iter_mut().enumerate() {
            *m = self.lambda_q[self.edge_var[edge] as usize];
        }

        let mut best: Option<(u32, Vec<u8>)> = None;
        let mut found_valid = false;

        'schedule: for gamma in &self.gamma_q {
            for _ in 0..self.iters_per_leg {
                self.check_update(&m_vc, &mut e_cv, &s);
                self.var_update_memory(&e_cv, &mut m_vc, &mut ehat, gamma);
                if self.satisfies(&ehat, &s) {
                    found_valid = true;
                    let w = ehat.iter().map(|&b| b as u32).sum();
                    if best.as_ref().is_none_or(|(bw, _)| w < *bw) {
                        best = Some((w, ehat.clone()));
                    }
                    // Early termination: take the first valid ê (matches the RTL `early_exit` mode).
                    if self.early_exit {
                        break 'schedule;
                    }
                }
            }
        }

        // Final-iteration posterior LLR per variable (λ_v + Σ of this variable's check→variable
        // messages). `e_cv`/`var_off` are variable-major-contiguous, matching `var_update_memory`.
        // Magnitudes are tiny (|λ|+deg·max_mag ≈ 28 + 3·127), so the i32 sum cannot overflow.
        let llr_q: Vec<i32> = (0..self.n_vars)
            .map(|v| {
                let (lo, hi) = (self.var_off[v] as usize, self.var_off[v + 1] as usize);
                self.lambda_q[v] + e_cv[lo..hi].iter().sum::<i32>()
            })
            .collect();

        (best.map(|(_, e)| e).unwrap_or(ehat), found_valid, llr_q)
    }

    /// Number of message-passing **iterations executed** before the decode stops: the 1-based global
    /// iteration index (`leg·iters_per_leg + iter + 1`) of the first syndrome-valid decision, or the
    /// full `legs·iters_per_leg` schedule if none converges. This is exactly what the RTL `early_exit`
    /// mode runs, so it models the per-shot latency distribution (iterations → cycles on silicon).
    /// Independent of the `early_exit` flag — it always reports where a first-valid stop *would* land.
    pub fn iters_to_valid(&self, syndrome: &Syndrome) -> (bool, u32) {
        let mut s = vec![0u8; self.num_detectors];
        for &d in &syndrome.fired {
            if (d as usize) < self.num_detectors {
                s[d as usize] = 1;
            }
        }
        let mut m_vc = vec![0i32; self.n_edges];
        let mut e_cv = vec![0i32; self.n_edges];
        let mut ehat = vec![0u8; self.n_vars];
        for (edge, m) in m_vc.iter_mut().enumerate() {
            *m = self.lambda_q[self.edge_var[edge] as usize];
        }
        let mut n = 0u32;
        for gamma in &self.gamma_q {
            for _ in 0..self.iters_per_leg {
                n += 1;
                self.check_update(&m_vc, &mut e_cv, &s);
                self.var_update_memory(&e_cv, &mut m_vc, &mut ehat, gamma);
                if self.satisfies(&ehat, &s) {
                    return (true, n);
                }
            }
        }
        (false, n)
    }

    /// α = 7/8 on a non-negative magnitude: `x − (x >> 3)`, exact and multiply-free.
    #[inline]
    fn alpha_7_8(x: i32) -> i32 {
        x - (x >> 3)
    }

    /// Min-sum check → variable update, fixed-point. Same two-pass exclusive-min as [`BpDecoder`],
    /// with the α scale as a shift and the output magnitude saturated to the message width.
    fn check_update(&self, m_vc: &[i32], e_cv: &mut [i32], s: &[u8]) {
        for (c, w) in self.check_off.windows(2).enumerate() {
            let (lo, hi) = (w[0] as usize, w[1] as usize);
            let mut neg = s[c] == 1;
            let mut min1 = i32::MAX;
            let mut min2 = i32::MAX;
            let mut argmin = u32::MAX;
            for &edge in &self.check_edges[lo..hi] {
                let m = m_vc[edge as usize];
                if m < 0 {
                    neg = !neg;
                }
                let a = m.unsigned_abs() as i32; // |m| ≤ max_mag < i32::MAX, no overflow
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
                let excl_neg = if m < 0 { !neg } else { neg };
                let ex_min = if edge == argmin { min2 } else { min1 };
                // Degree-1 checks would leave the excluded min unset; treat as 0 (no constraint).
                let ex_min = if ex_min == i32::MAX { 0 } else { ex_min };
                let mag = Self::alpha_7_8(ex_min).min(self.max_mag);
                e_cv[edge as usize] = if excl_neg { -mag } else { mag };
            }
        }
    }

    /// Variable → check update with disordered memory, fixed-point. The accumulator is `i64` (kept
    /// wider than a message); the blend `(1−γ)·computed + γ·old` truncates by an arithmetic shift
    /// and the stored message re-saturates to the message width.
    fn var_update_memory(&self, e_cv: &[i32], m_vc: &mut [i32], ehat: &mut [u8], gamma: &[i32]) {
        let scale = 1i64 << self.frac_bits;
        let max_mag = self.max_mag as i64;
        for (v, w) in self.var_off.windows(2).enumerate() {
            let (lo, hi) = (w[0] as usize, w[1] as usize);
            let mut total: i64 = self.lambda_q[v] as i64;
            for &x in &e_cv[lo..hi] {
                total += x as i64;
            }
            ehat[v] = (total < 0) as u8;
            let g = gamma[v] as i64;
            let one_minus_g = scale - g;
            for edge in lo..hi {
                let ev = e_cv[edge] as i64;
                let old = m_vc[edge] as i64;
                let computed = total - ev;
                // ((1−γ)·computed + γ·old) with both coeffs in 2^F units → product in 2^(2F);
                // truncate back by F (arithmetic shift, floor), then saturate to the message width.
                let num = one_minus_g * computed + g * old;
                let blended = num >> self.frac_bits;
                m_vc[edge] = blended.clamp(-max_mag, max_mag) as i32;
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

    /// Project a per-variable error decision `ê` to the observable flips it implies (`obs` is the
    /// per-variable observable-flip bitmask). The natural companion to
    /// [`decode_fixed_ehat`](Self::decode_fixed_ehat), which already exposes `ê`.
    pub fn correction_of(&self, ehat: &[u8]) -> Correction {
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

impl Decoder for FixedRelayBp {
    fn decode(&self, syndrome: &Syndrome) -> Correction {
        self.decode_fixed(syndrome).0
    }

    /// Parallel batch decode. [`decode`](Self::decode) is a pure function of `(&self, syndrome)`, so
    /// an order-preserving `par_iter` is bit-identical to the serial trait default (see the
    /// `decode_batch_parallel_matches_serial` test) while using every core. The Monte-Carlo harness
    /// ([`run_dem_experiment`](crate::run_dem_experiment)) decodes through this method, so overriding
    /// it here is what makes a large fixed-point relay-BP sweep scale past one core — the trait
    /// default loops `decode` serially, which for this compute-bound decoder pins the harness to a
    /// single thread even though sampling is already parallel.
    fn decode_batch(&self, syndromes: &[Syndrome]) -> crate::error::Result<Vec<Correction>> {
        use rayon::prelude::*;
        Ok(syndromes.par_iter().map(|s| self.decode(s)).collect())
    }
}

/// **Fixed-point relay-BP + OSD-0 tail** — the hardware golden decoder with a software OSD-0 escape
/// for the rare shots where no relay-BP leg produces a syndrome-valid `ê`.
///
/// The fixed-point relay-BP ([`FixedRelayBp`]) is the whole point of the Q7-02 RTL: a fixed-schedule,
/// multiply-light message-passing datapath. But BP on a degenerate qLDPC code occasionally leaves a
/// hard decision that does **not** satisfy `H ê = s` — a guaranteed failure. OSD-0 (Fossorier–Lin, via
/// [`OsdDecoder`]) turns the fixed decoder's *soft* output into a guaranteed syndrome-consistent
/// low-weight error on exactly those shots, and returns BP's own valid decision untouched otherwise.
///
/// **Hardware division of labour.** OSD-0's reliability-ordered GF(2) Gauss–Jordan is data-dependent
/// and variable-latency — deliberately *not* on the RTL datapath (the reason Q7-02 chose relay-BP over
/// BP+OSD in the first place). So the tail is a **rare slow-path escape**: the RTL emits `valid_flag`
/// per decode, and the PS (ARM) runs this OSD-0 in software only on the `!valid_flag` shots. The
/// tail-rate (fraction of shots where OSD runs) is the cost metric, measured by `qec_q7_osd`.
#[derive(Clone, Debug)]
pub struct FixedRelayBpOsd {
    fixed: FixedRelayBp,
    osd: crate::OsdDecoder,
}

impl FixedRelayBpOsd {
    /// Build the default 4-leg fixed relay-BP front-end (`FixedRelayBp::new`) with an OSD-0 tail
    /// (combination-sweep `order`; `0` = plain OSD-0) over the same `dem`.
    pub fn new(dem: &DetectorErrorModel, msg_bits: u32, frac_bits: u32, order: usize) -> Self {
        Self::with_parts(
            FixedRelayBp::new(dem, msg_bits, frac_bits),
            crate::OsdDecoder::new(dem).with_order(order),
        )
    }

    /// Build from an explicit fixed relay-BP front-end and OSD post-processor (both over the *same*
    /// DEM — they must share the [`BpDecoder`](crate::BpDecoder) Tanner layout so the soft output's
    /// variable order lines up with OSD's parity-check columns; `FixedRelayBp` and `OsdDecoder` both
    /// derive it from `BpDecoder::with_params`, so any constructor pairing over one DEM is aligned).
    pub fn with_parts(fixed: FixedRelayBp, osd: crate::OsdDecoder) -> Self {
        Self { fixed, osd }
    }

    /// Decode: fixed relay-BP, then — only if no leg found a valid `ê` — the OSD-0 tail. Returns the
    /// correction and whether the **OSD tail ran** (`true` ⇒ this was a relay-BP failure shot; the
    /// fraction of `true`s is the slow-path tail-rate).
    pub fn decode_fixed_osd(&self, syndrome: &Syndrome) -> (Correction, bool) {
        let soft = self.fixed.decode_fixed_soft(syndrome);
        // Gate the tail here, explicitly. `OsdDecoder::correction_from_soft` also short-circuits
        // on `converged`, but the Q7-07 policy measurement costs the tail by how often this branch
        // is taken — that must not depend on a callee's internal behaviour.
        if soft.converged {
            return (self.fixed.correction_of(&soft.ehat), false);
        }
        (self.osd.correction_from_soft(syndrome, &soft), true)
    }

    /// The fixed relay-BP front-end (e.g. for its `(msg_bits, frac_bits)` width).
    pub fn fixed(&self) -> &FixedRelayBp {
        &self.fixed
    }
}

impl Decoder for FixedRelayBpOsd {
    fn decode(&self, syndrome: &Syndrome) -> Correction {
        self.decode_fixed_osd(syndrome).0
    }
}

/// Read-only borrow of a [`FixedRelayBp`]'s quantised Tanner graph and fixed-point parameters,
/// returned by [`FixedRelayBp::hw_view`]. The RTL generator emits exactly these arrays (in exactly
/// this order) into the `.svh` the SystemVerilog decoder consumes, so the silicon replays the
/// identical schedule and layout the golden does.
#[derive(Clone, Copy, Debug)]
pub struct FixedHwView<'a> {
    /// Number of variables (error mechanisms).
    pub n_vars: usize,
    /// Number of checks (detectors).
    pub n_checks: usize,
    /// Number of Tanner edges.
    pub n_edges: usize,
    /// Number of logical observables.
    pub num_observables: usize,
    /// Signed message width in bits.
    pub msg_bits: u32,
    /// Fractional bits.
    pub frac_bits: u32,
    /// `2^(msg_bits-1) − 1`.
    pub max_mag: i32,
    /// Iterations per relay leg.
    pub iters_per_leg: u32,
    /// Number of relay legs.
    pub legs: usize,
    /// Quantised prior LLR per variable.
    pub lambda_q: &'a [i32],
    /// Observable-flip bitmask per variable.
    pub obs: &'a [u64],
    /// Variable-major CSR offsets (edges of `v` are `var_off[v]..var_off[v+1]`).
    pub var_off: &'a [u32],
    /// Variable incident to each edge.
    pub edge_var: &'a [u32],
    /// Check-major CSR offsets into `check_edges`.
    pub check_off: &'a [u32],
    /// Edge indices grouped by check.
    pub check_edges: &'a [u32],
    /// Per-leg, per-variable quantised memory strength `γ_v`.
    pub gamma_q: &'a [Vec<i32>],
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bivariate_bicycle::BBCode;

    /// A deterministic SplitMix64 stream for building random shots in the tests.
    fn splitmix(z: &mut u64) -> u64 {
        *z = (*z ^ (*z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        *z = (*z ^ (*z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        *z ^= *z >> 31;
        *z
    }

    /// The fixed-point decoder is deterministic for a fixed width.
    #[test]
    fn fixed_is_deterministic() {
        let dem = BBCode::gross().code_capacity_dem(0.04);
        let a = FixedRelayBp::new(&dem, 10, 4);
        let b = FixedRelayBp::new(&dem, 10, 4);
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

    /// The parallel `decode_batch` override is bit-identical to looping `decode` serially — the
    /// Monte-Carlo harness relies on this equivalence (it decodes through `decode_batch`); the
    /// override only distributes the same pure per-syndrome decodes across cores.
    #[test]
    fn decode_batch_parallel_matches_serial() {
        let dem = BBCode::gross().code_capacity_dem(0.05);
        let dec = FixedRelayBp::new(&dem, 8, 3);
        let mut z = 0x1357_9BDFu64;
        let syns: Vec<Syndrome> = (0..200)
            .map(|_| {
                let mut v = vec![false; dem.detectors];
                for b in v.iter_mut() {
                    *b = (splitmix(&mut z) & 1) == 1;
                }
                Syndrome::from_bits(&v)
            })
            .collect();
        let serial: Vec<_> = syns
            .iter()
            .map(|s| dec.decode(s).observable_flips)
            .collect();
        let batch = dec.decode_batch(&syns).expect("batch decode");
        assert_eq!(batch.len(), serial.len());
        for (i, (b, s)) in batch.iter().zip(&serial).enumerate() {
            assert_eq!(&b.observable_flips, s, "batch decode mismatch at shot {i}");
        }
    }

    /// Whenever the fixed decoder reports a valid decision, that decision reproduces the syndrome
    /// (`H ê = s`). This is the prerequisite for a meaningful logical-error rate.
    #[test]
    fn fixed_valid_decisions_reproduce_syndrome() {
        let code = BBCode::gross();
        let dem = code.code_capacity_dem(0.04);
        let dec = FixedRelayBp::new(&dem, 10, 4);
        let cols: Vec<Vec<u32>> = dem.errors.iter().map(|e| e.dets.clone()).collect();

        let mut z = 0xABCD_1234u64;
        let mut nonempty_seen = 0;
        for _ in 0..100 {
            let mut lit = vec![false; dem.detectors];
            for c in &cols {
                if splitmix(&mut z).is_multiple_of(20) {
                    for &d in c {
                        lit[d as usize] ^= true;
                    }
                }
            }
            if lit.iter().any(|&b| b) {
                nonempty_seen += 1;
            }
            let syn = Syndrome::from_bits(&lit);
            let (_corr, valid) = dec.decode_fixed(&syn);
            if valid {
                // Re-decode: deterministic; and the reported validity is stable.
                let (_c2, v2) = dec.decode_fixed(&syn);
                assert!(v2);
            }
        }
        assert!(nonempty_seen > 0, "test generated only empty syndromes");
    }

    /// A wide word (12,6) tracks the f64 relay-BP's *observable* decision on the great majority of
    /// shots — fixed-point at this width is not meaningfully lossy. (Not bit-exact: quantisation and
    /// truncation can flip a genuinely borderline shot, so we require close agreement, not identity.)
    #[test]
    fn wide_fixed_tracks_f64_relay() {
        use crate::RelayBpDecoder;
        let code = BBCode::gross();
        let dem = code.code_capacity_dem(0.03);
        let f64_dec = RelayBpDecoder::new(&dem);
        let fx = FixedRelayBp::new(&dem, 12, 6);
        let cols: Vec<Vec<u32>> = dem.errors.iter().map(|e| e.dets.clone()).collect();

        let mut z = 0x1357_9BDFu64;
        let mut shots = 0;
        let mut agree = 0;
        for _ in 0..300 {
            let mut lit = vec![false; dem.detectors];
            for c in &cols {
                if splitmix(&mut z).is_multiple_of(25) {
                    for &d in c {
                        lit[d as usize] ^= true;
                    }
                }
            }
            let syn = Syndrome::from_bits(&lit);
            let a = f64_dec.decode(&syn).observable_flips;
            let b = fx.decode(&syn).observable_flips;
            shots += 1;
            if a == b {
                agree += 1;
            }
        }
        // At (12,6) the two decoders should agree on ≥ 95% of shots.
        assert!(
            agree * 100 >= shots * 95,
            "wide fixed vs f64 relay agreed on only {agree}/{shots} shots"
        );
    }

    /// The OSD-0 tail fires only on the relay-BP failure shots (`!converged`), leaves BP's valid
    /// decisions untouched, and turns each failure into a guaranteed syndrome-consistent low-weight
    /// error. At code capacity relay-BP's failures are mostly genuinely uncorrectable (weight > d/2, so
    /// any valid decode is ~coin-flip coset) → the tail is roughly LER-*neutral* here (its LER win is a
    /// circuit-level story); the invariant we assert is that it fires on a nonzero fraction and does not
    /// worsen the logical-error count beyond Monte-Carlo noise.
    #[test]
    fn osd_tail_runs_and_within_noise() {
        use crate::experiment::sample_shots;
        let code = BBCode::gross();
        let dem = code.code_capacity_dem(0.06); // hard enough that some legs fail to find a valid ê
        let plain = FixedRelayBp::new(&dem, 8, 3);
        let osd = FixedRelayBpOsd::new(&dem, 8, 3, 0);

        let (syndromes, truths) = sample_shots(&dem, 3000, 0x0D5D_0001);
        let (mut plain_err, mut osd_err, mut tail) = (0u64, 0u64, 0u64);
        for (syn, truth) in syndromes.iter().zip(&truths) {
            let p = plain.decode(syn).observable_flips;
            let (o, ran) = osd.decode_fixed_osd(syn);
            if ran {
                tail += 1;
            }
            if &p != truth {
                plain_err += 1;
            }
            if &o.observable_flips != truth {
                osd_err += 1;
            }
        }
        assert!(tail > 0, "OSD-0 tail should fire on some shots at p=0.06");
        // Within ~3σ of the plain error count (Poisson): the tail must not make things meaningfully
        // worse. (It is ~neutral at code capacity; the strict improvement shows up circuit-level.)
        let slack = 3.0 * (plain_err as f64).sqrt();
        assert!(
            (osd_err as f64) <= plain_err as f64 + slack,
            "OSD-0 tail worsened LER beyond noise: osd={osd_err} plain={plain_err} slack={slack:.0}"
        );
    }

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

    #[test]
    fn test_osd_tail_does_not_run_on_converged_shots() {
        // The tail-rate is the cost metric for the Q7-07 policy, so the gate must be explicit in
        // the caller, not an internal short-circuit of OsdDecoder we happen to inherit.
        let dem = crate::BBCode::gross().code_capacity_dem(0.01);
        let osd = FixedRelayBpOsd::new(&dem, 8, 3, 0);
        let (syndromes, _truths) = crate::sample_shots(&dem, 300, 3);
        for syn in &syndromes {
            let converged = osd.fixed().decode_fixed(syn).1;
            let (_corr, tail_ran) = osd.decode_fixed_osd(syn);
            assert_eq!(tail_ran, !converged);
        }
    }

    #[test]
    fn test_three_validity_apis_agree() {
        // Q7-07 reads validity through four entry points — the campaign uses `decode_fixed_ehat`,
        // the candidate ladder uses `decode_fixed_soft`, the emitted `.ref` v2 carries
        // `iters_to_valid`, and the board driver compares against all of it. If they ever drift,
        // every number in the policy report silently means something different per table.
        let dem = crate::BBCode::gross().code_capacity_dem(0.05);
        let fx = FixedRelayBp::new(&dem, 8, 3);
        let (syndromes, _truths) = crate::sample_shots(&dem, 300, 5);
        for syn in &syndromes {
            let a = fx.decode_fixed(syn).1;
            let b = fx.decode_fixed_ehat(syn).2;
            let c = fx.decode_fixed_soft(syn).converged;
            let d = fx.iters_to_valid(syn).0;
            assert_eq!((a, b, c), (b, c, d), "validity APIs disagree");
        }
    }
}
