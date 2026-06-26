//! [`BpDecoder`] — a normalised **min-sum belief-propagation** decoder over the Tanner graph of a
//! [`DetectorErrorModel`].
//!
//! Unlike MWPM ([`MwpmDecoder`](crate::MwpmDecoder)) and Union-Find
//! ([`UnionFindDecoder`](crate::UnionFindDecoder)), which need a *graphlike* DEM (every mechanism
//! flips ≤ 2 detectors), belief propagation works on an arbitrary parity-check (Tanner) graph — so
//! it is the natural decoder for **qLDPC** codes (Phase Q5), where checks touch many qubits. This is
//! the CPU reference the GPU BP kernel (Q3-02) is oracle-checked against, and the front half of the
//! BP+OSD decoder (Q5-02).
//!
//! # The algorithm (min-sum)
//!
//! Variables are the DEM's error mechanisms (prior log-likelihood ratio `λ_v = ln((1-p_v)/p_v)`),
//! checks are its detectors. Given a syndrome `s` (which detectors fired) the decoder passes messages
//! on the Tanner graph for a fixed number of iterations:
//!
//! * **check → variable** (min-sum): `E_{c→v} = (-1)^{s_c} · α · (∏_{v'≠v} sign M_{v'→c}) ·
//!   min_{v'≠v} |M_{v'→c}|` — the violated-check sign flip `(-1)^{s_c}` and the exclusive-minimum
//!   magnitude, optionally scaled by a normalisation factor `α ≤ 1`.
//! * **variable → check**: `M_{v→c} = λ_v + Σ_{c'≠c} E_{c'→v}`.
//! * **hard decision**: `ê_v = [λ_v + Σ_c E_{c→v} < 0]`. Stop early once `H ê = s`.
//!
//! All arithmetic is `f64` in a fixed edge order, so an external decoder (the GPU port) that
//! replays the identical loop reproduces `ê` bit-for-bit.
//!
//! Pure BP is degeneracy-limited on surface codes (split beliefs on equivalent errors), so its
//! standalone logical accuracy there is *below* MWPM/UF — the fix is BP+OSD (Q5-02). On a repetition
//! code, and on low-weight surface-code errors, it converges to the correct correction.

use crate::decoder::Decoder;
use crate::dem::DetectorErrorModel;
use crate::syndrome::{Correction, Syndrome};

/// Default belief-propagation iteration cap.
pub const DEFAULT_MAX_ITER: u32 = 100;

/// A min-sum belief-propagation decoder for a fixed [`DetectorErrorModel`].
///
/// Construct from a DEM; the Tanner graph is flattened into CSR arrays (variable-major edges plus a
/// check-major index) reused across shots. Decode is `&self` (the message buffers are per-decode
/// scratch in a thread-local), so the decoder is `Sync` and the harness can decode in parallel.
#[derive(Clone, Debug)]
pub struct BpDecoder {
    num_detectors: usize,
    num_observables: usize,
    n_vars: usize,
    n_edges: usize,
    max_iter: u32,
    /// Normalised min-sum scale `α ∈ (0, 1]` (1.0 = plain min-sum).
    alpha: f64,
    /// Prior LLR of each variable, `ln((1-p)/p)`.
    lambda: Vec<f64>,
    /// Observable-flip bitmask of each variable.
    obs: Vec<u64>,
    /// Variable-major CSR: edges of variable `v` are `var_off[v]..var_off[v+1]`.
    var_off: Vec<u32>,
    /// Check incident to each edge.
    edge_check: Vec<u32>,
    /// Variable incident to each edge.
    edge_var: Vec<u32>,
    /// Check-major CSR: edge indices of check `c` are `check_edges[check_off[c]..check_off[c+1]]`.
    check_off: Vec<u32>,
    check_edges: Vec<u32>,
}

impl BpDecoder {
    /// Build a min-sum BP decoder for `dem` with the default iteration cap and plain min-sum (`α=1`).
    pub fn new(dem: &DetectorErrorModel) -> Self {
        Self::with_params(dem, DEFAULT_MAX_ITER, 1.0)
    }

    /// Build with an explicit iteration cap and normalisation factor `α` (clamped to `(0, 1]`).
    pub fn with_params(dem: &DetectorErrorModel, max_iter: u32, alpha: f64) -> Self {
        let n_vars = dem.errors.len();
        let n_checks = dem.detectors;

        // Variable-major edges: for each mechanism, one edge per *distinct* detector it flips (a
        // detector listed an even number of times cancels by parity and is not a check edge).
        let mut var_off = vec![0u32; n_vars + 1];
        let mut edge_check: Vec<u32> = Vec::new();
        let mut edge_var: Vec<u32> = Vec::new();
        let mut lambda = vec![0.0f64; n_vars];
        let mut obs = vec![0u64; n_vars];
        for (v, e) in dem.errors.iter().enumerate() {
            // Parity-reduce the (sorted) detector list: keep detectors appearing an odd number of
            // times. `dets` is sorted by `DemError::new`, so runs are contiguous.
            let mut i = 0;
            while i < e.dets.len() {
                let d = e.dets[i];
                let mut cnt = 0;
                while i < e.dets.len() && e.dets[i] == d {
                    cnt += 1;
                    i += 1;
                }
                if cnt % 2 == 1 && (d as usize) < n_checks {
                    edge_check.push(d);
                    edge_var.push(v as u32);
                }
            }
            var_off[v + 1] = edge_check.len() as u32;

            let p = e.prob.clamp(1e-12, 0.5);
            lambda[v] = ((1.0 - p) / p).ln();
            obs[v] = e
                .obs
                .iter()
                .filter(|&&o| o < 64)
                .fold(0u64, |m, &o| m | (1u64 << o));
        }
        let n_edges = edge_check.len();

        // Check-major index (counting sort of edges by their check).
        let mut check_off = vec![0u32; n_checks + 1];
        for &c in &edge_check {
            check_off[c as usize + 1] += 1;
        }
        for c in 0..n_checks {
            check_off[c + 1] += check_off[c];
        }
        let mut check_edges = vec![0u32; n_edges];
        let mut cursor = check_off.clone();
        for (edge, &c) in edge_check.iter().enumerate() {
            let slot = &mut cursor[c as usize];
            check_edges[*slot as usize] = edge as u32;
            *slot += 1;
        }

        BpDecoder {
            num_detectors: n_checks,
            num_observables: dem.observables,
            n_vars,
            n_edges,
            max_iter,
            alpha: alpha.clamp(1e-6, 1.0),
            lambda,
            obs,
            var_off,
            edge_check,
            edge_var,
            check_off,
            check_edges,
        }
    }

    /// Read-only view of the flattened Tanner graph + BP parameters, for an external decoder (the
    /// GPU port, Q3-02) that replays the identical message-passing schedule and so must consume the
    /// identical layout. Every field mirrors the identically-named private field.
    pub fn tanner(&self) -> TannerGraph<'_> {
        TannerGraph {
            num_detectors: self.num_detectors,
            num_observables: self.num_observables,
            n_vars: self.n_vars,
            n_edges: self.n_edges,
            max_iter: self.max_iter,
            alpha: self.alpha,
            lambda: &self.lambda,
            obs: &self.obs,
            var_off: &self.var_off,
            edge_check: &self.edge_check,
            edge_var: &self.edge_var,
            check_off: &self.check_off,
            check_edges: &self.check_edges,
        }
    }

    /// Number of logical observables (correction width).
    pub fn num_observables(&self) -> usize {
        self.num_observables
    }

    /// Decode `syndrome`, returning the correction and whether BP **converged** (`H ê = s` within
    /// the iteration cap). The bool is exposed for diagnostics; [`decode`](Decoder::decode) drops it.
    pub fn decode_bp(&self, syndrome: &Syndrome) -> (Correction, bool) {
        // Syndrome bits over checks.
        let mut s = vec![0u8; self.num_detectors];
        for &d in &syndrome.fired {
            if (d as usize) < self.num_detectors {
                s[d as usize] = 1;
            }
        }

        // Per-decode message buffers.
        let mut m_vc = vec![0.0f64; self.n_edges]; // variable → check
        let mut e_cv = vec![0.0f64; self.n_edges]; // check → variable
        let mut ehat = vec![0u8; self.n_vars];

        // Init: M_{v→c} = λ_v.
        for (edge, m) in m_vc.iter_mut().enumerate() {
            *m = self.lambda[self.edge_var[edge] as usize];
        }

        let mut converged = false;
        for _ in 0..self.max_iter {
            self.check_update(&m_vc, &mut e_cv, &s);
            self.var_update(&e_cv, &mut m_vc, &mut ehat);
            if self.satisfies(&ehat, &s) {
                converged = true;
                break;
            }
        }

        let mut mask = 0u64;
        for (v, &e) in ehat.iter().enumerate() {
            if e == 1 {
                mask ^= self.obs[v];
            }
        }
        let flips = (0..self.num_observables)
            .map(|o| (mask >> o) & 1 == 1)
            .collect();
        (Correction::new(flips), converged)
    }

    /// Min-sum check → variable update: `E_{c→v} = (-1)^{s_c} α · (∏_{v'≠v} sign) · min_{v'≠v} |·|`.
    fn check_update(&self, m_vc: &[f64], e_cv: &mut [f64], s: &[u8]) {
        for (c, w) in self.check_off.windows(2).enumerate() {
            let lo = w[0] as usize;
            let hi = w[1] as usize;
            // First pass: overall sign, two smallest magnitudes, and the argmin edge.
            let mut neg = s[c] == 1; // running sign: true ⇒ negative
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
            // Second pass: exclude each edge's own contribution.
            for &edge in &self.check_edges[lo..hi] {
                let m = m_vc[edge as usize];
                // Sign excluding this edge: overall sign times this edge's sign.
                let excl_neg = if m < 0.0 { !neg } else { neg };
                let ex_min = if edge == argmin { min2 } else { min1 };
                let mag = self.alpha * ex_min;
                e_cv[edge as usize] = if excl_neg { -mag } else { mag };
            }
        }
    }

    /// Variable → check update + posterior hard decision.
    fn var_update(&self, e_cv: &[f64], m_vc: &mut [f64], ehat: &mut [u8]) {
        for (v, w) in self.var_off.windows(2).enumerate() {
            let lo = w[0] as usize;
            let hi = w[1] as usize;
            let mut total = self.lambda[v];
            for &x in &e_cv[lo..hi] {
                total += x;
            }
            ehat[v] = (total < 0.0) as u8;
            for (mv, ev) in m_vc[lo..hi].iter_mut().zip(&e_cv[lo..hi]) {
                *mv = total - *ev;
            }
        }
    }

    /// Whether the hard decision reproduces the syndrome: `H ê = s`.
    fn satisfies(&self, ehat: &[u8], s: &[u8]) -> bool {
        for (c, w) in self.check_off.windows(2).enumerate() {
            let lo = w[0] as usize;
            let hi = w[1] as usize;
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
}

impl Decoder for BpDecoder {
    fn decode(&self, syndrome: &Syndrome) -> Correction {
        self.decode_bp(syndrome).0
    }
}

/// Read-only borrow of a [`BpDecoder`]'s flattened Tanner graph and BP parameters.
///
/// Returned by [`BpDecoder::tanner`] for the Q3-02 GPU port, which replays the identical min-sum
/// schedule and therefore must upload exactly these arrays in exactly this order to stay
/// numerically identical. Every field mirrors the identically-named private field.
#[derive(Clone, Copy, Debug)]
pub struct TannerGraph<'a> {
    /// Number of checks (detectors).
    pub num_detectors: usize,
    /// Number of logical observables.
    pub num_observables: usize,
    /// Number of variables (error mechanisms).
    pub n_vars: usize,
    /// Number of Tanner edges (variable–check incidences).
    pub n_edges: usize,
    /// Iteration cap.
    pub max_iter: u32,
    /// Normalised min-sum scale `α`.
    pub alpha: f64,
    /// Prior LLR per variable.
    pub lambda: &'a [f64],
    /// Observable-flip bitmask per variable.
    pub obs: &'a [u64],
    /// Variable-major CSR offsets into the edge arrays.
    pub var_off: &'a [u32],
    /// Check incident to each edge.
    pub edge_check: &'a [u32],
    /// Variable incident to each edge.
    pub edge_var: &'a [u32],
    /// Check-major CSR offsets into `check_edges`.
    pub check_off: &'a [u32],
    /// Edge indices grouped by check.
    pub check_edges: &'a [u32],
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::build_dem;
    use crate::dem::DemError;
    use crate::surface::SurfaceCode;

    /// A length-`n` repetition-code DEM: `n` data-bit errors (each flips its two adjacent checks,
    /// boundary bits flip one) + `n-1` checks. A textbook min-sum target.
    fn repetition_dem(n: usize, p: f64) -> DetectorErrorModel {
        // checks 0..n-1 sit between data bits; data bit i flips checks i-1 and i.
        let n_checks = n - 1;
        let mut errors = Vec::new();
        for i in 0..n {
            let mut dets = Vec::new();
            if i > 0 {
                dets.push((i - 1) as u32);
            }
            if i < n_checks {
                dets.push(i as u32);
            }
            // Make the leftmost data bit carry the logical observable.
            let obs = if i == 0 { vec![0u32] } else { vec![] };
            errors.push(DemError::new(p, dets, obs));
        }
        DetectorErrorModel {
            detectors: n_checks,
            observables: 1,
            errors,
        }
    }

    /// BP on a repetition code recovers every single-bit error (syndrome = the bit's two checks).
    #[test]
    fn repetition_recovers_single_errors() {
        let n = 12;
        let dem = repetition_dem(n, 0.05);
        let bp = BpDecoder::new(&dem);
        for bit in 0..n {
            // The syndrome that error `bit` produces.
            let e = &dem.errors[bit];
            let s = Syndrome::new(n - 1, e.dets.clone());
            let (corr, conv) = bp.decode_bp(&s);
            assert!(
                conv,
                "BP must converge on a single repetition error (bit {bit})"
            );
            // The correction must reproduce the syndrome's logical effect: observable flips iff the
            // true error flips the observable.
            let truth = e.obs.contains(&0);
            assert_eq!(corr.observable_flips[0], truth, "bit {bit}");
        }
    }

    /// BP reproduces the empty syndrome as no correction.
    #[test]
    fn empty_syndrome_no_correction() {
        let dem = repetition_dem(8, 0.05);
        let bp = BpDecoder::new(&dem);
        let s = Syndrome::new(7, vec![]);
        let (corr, conv) = bp.decode_bp(&s);
        assert!(conv);
        assert_eq!(corr, Correction::none(1));
    }

    /// BP converges (reproduces the syndrome) on low-weight surface-code errors at small distance.
    #[test]
    fn surface_low_weight_converges() {
        let d = 3;
        let exp = SurfaceCode::new(d).memory_z_experiment(d);
        let dem = build_dem(&exp.annotated, &exp.phenomenological_mechanisms(0.01, 0.01)).unwrap();
        let bp = BpDecoder::with_params(&dem, 200, 0.875);
        // Each single mechanism's syndrome should be reproduced by BP's hard decision.
        let mut converged = 0;
        let n = dem.errors.len();
        for e in &dem.errors {
            if e.dets.is_empty() {
                continue;
            }
            let s = Syndrome::new(dem.detectors, e.dets.clone());
            let (_c, conv) = bp.decode_bp(&s);
            if conv {
                converged += 1;
            }
        }
        // The vast majority of weight-1 syndromes converge (some degenerate ones may not — that is
        // the known BP-on-surface limitation BP+OSD fixes).
        assert!(
            converged * 100 >= n * 90,
            "only {converged}/{n} weight-1 syndromes converged"
        );
    }
}
