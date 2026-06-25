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

/// Qubit count at/above which `auto` prefers the Metal GPU state vector over the
/// CPU one — *when the Metal backend is available* (P5.6-07). The GPU SV is FP32
/// but ~4.7–6.1× the CPU SV at n≈24–28 (docs/perf/metal.md); below this the CPU
/// FP64 SV is exact and fast enough that the GPU's dispatch overhead isn't worth
/// the precision drop. Bounded above by [`SV_EXACT_CAP`] (both cap at 28 qubits).
pub const GPU_PREFER_N: u32 = 24;

/// Largest FP64 CUDA state held **in core** on the reference 20 GiB card — mirrors
/// `aleph_cuda::MAX_CUDA_QUBITS` (16 GiB of `2^30` FP64 amplitudes). Above this the
/// reach policy switches to the out-of-core paged executor ([`Reach::Paged`]).
/// Duplicated here because `aleph-backend` must not depend on `aleph-cuda` (the
/// dependency runs the other way); `aleph-cuda`'s `cuda_caps_match` test pins the
/// two copies together.
pub const MAX_CUDA_QUBITS: u32 = 30;

/// Largest FP32 CUDA state held in core — mirrors `aleph_cuda::MAX_CUDA_QUBITS_F32`
/// (16 GiB of `2^31` FP32 amplitudes, one qubit past FP64). Paged beyond.
pub const MAX_CUDA_QUBITS_F32: u32 = 31;

/// Amplitude precision preference for the GPU state-vector path. The reach policy
/// resolves [`Precision::Auto`] to FP32 (faster + more in-core reach, ~1e-5
/// accuracy — the same trade the Metal `auto` pick makes); `F64` / `F32` are
/// explicit overrides honoured verbatim.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Precision {
    /// Let the policy choose (resolves to FP32 on the GPU path).
    #[default]
    Auto,
    /// Force FP64 amplitudes.
    F64,
    /// Force FP32 amplitudes.
    F32,
}

/// Whether the chosen state vector fits in device memory or must stream tiles
/// through the GPU out-of-core (P5.10-02 / P5.11-01 paging).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum Reach {
    /// The whole `2^n` state fits in device memory — the normal in-core run.
    #[default]
    InCore,
    /// `2^n` exceeds the in-core cap; stream device-sized tiles of `tile_qubits`
    /// low bits through the GPU (the paged executor's split parameter).
    Paged { tile_qubits: u32 },
}

impl Reach {
    /// Reach for a concrete `kind` at `num_qubits`: the CUDA kinds page above their
    /// in-core cap (paged tile one qubit below the cap, for streaming headroom);
    /// every other kind is always in-core. Shared by the `auto` policy and the
    /// CLI/Python explicit-`--backend cuda` path so both compute paging the same way.
    pub fn for_kind(kind: BackendKind, num_qubits: u32) -> Reach {
        match kind.cuda_in_core_cap() {
            Some(cap) if num_qubits > cap => Reach::Paged {
                tile_qubits: cap.saturating_sub(1),
            },
            _ => Reach::InCore,
        }
    }
}

/// Runtime environment for the reach policy: which GPU backends the caller has
/// probed as available, plus the precision preference. Avoids growing the
/// positional argument list as backends are added (cf. the older
/// `select_from_env(_, metal_available)`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub struct SelectEnv {
    /// Metal GPU backend usable (Apple Silicon + device), as probed by the caller.
    pub metal_available: bool,
    /// CUDA GPU backend usable (NVIDIA/Linux + device), as probed by the caller.
    pub cuda_available: bool,
    /// Amplitude precision preference for the GPU path.
    pub precision: Precision,
}

/// Resolved, abstract backend label produced by the heuristic.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BackendKind {
    /// Dense state vector — exact, memory grows as 2^n.
    Statevector,
    /// Stabilizer tableau — Clifford-only, O(n²) memory.
    Stabilizer,
    /// MPS tensor network — bounded-entanglement, approximate beyond χ.
    Mps,
    /// Metal GPU dense state vector — FP32, Apple Silicon only. A manual override
    /// always, and — via [`select_from_env`] — an `auto` pick for large dense
    /// circuits when the caller reports the Metal backend is available. The pure
    /// [`select_from`] never returns it (it assumes Metal unavailable).
    Metal,
    /// CUDA GPU dense state vector — **FP64**, NVIDIA/Linux only (P5). A manual
    /// override, and — via [`select_from_full`] — an `auto` pick for large dense
    /// circuits when the caller reports CUDA availability and FP64 precision. In
    /// core to [`MAX_CUDA_QUBITS`] qubits, out-of-core ([`Reach::Paged`]) beyond.
    Cuda,
    /// CUDA GPU dense state vector — **FP32**, NVIDIA/Linux only (P5.10/5.11). The
    /// `auto` GPU pick at default precision (2× throughput, +1 qubit of in-core
    /// reach vs FP64, ~1e-5 accuracy). In core to [`MAX_CUDA_QUBITS_F32`], paged
    /// beyond.
    CudaF32,
}

impl std::fmt::Display for BackendKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            BackendKind::Statevector => "state vector",
            BackendKind::Stabilizer => "stabilizer",
            BackendKind::Mps => "MPS",
            BackendKind::Metal => "Metal GPU",
            BackendKind::Cuda => "CUDA GPU (FP64)",
            BackendKind::CudaF32 => "CUDA GPU (FP32)",
        })
    }
}

impl BackendKind {
    /// Every kind, in display order. Drives the user-facing parser and its
    /// error message off the single [`BackendKind::aliases`] source. A new
    /// variant is *caught at compile time* by the exhaustive match in
    /// `aliases`, not here; the `every_kind_round_trips` test fails if a
    /// variant is added without extending this list too.
    pub const ALL: [BackendKind; 6] = [
        BackendKind::Statevector,
        BackendKind::Stabilizer,
        BackendKind::Mps,
        BackendKind::Metal,
        BackendKind::Cuda,
        BackendKind::CudaF32,
    ];

    /// User-facing names that resolve to this kind: the canonical name first,
    /// then the established aliases (`sv`, `stab`).
    ///
    /// This exhaustive match is the **single place** backend names are wired —
    /// adding a [`BackendKind`] variant is a compile error here, which is how
    /// "adding a backend is a compile-error-guided change" (P4-12) is enforced.
    pub fn aliases(self) -> &'static [&'static str] {
        match self {
            BackendKind::Statevector => &["statevector", "sv"],
            BackendKind::Stabilizer => &["stabilizer", "stab"],
            BackendKind::Mps => &["mps"],
            BackendKind::Metal => &["metal", "gpu"],
            BackendKind::Cuda => &["cuda", "cuda-f64"],
            BackendKind::CudaF32 => &["cuda-f32", "cuda-fp32"],
        }
    }

    /// Canonical (preferred) user-facing name — the first of [`Self::aliases`].
    pub fn canonical_name(self) -> &'static str {
        self.aliases()[0]
    }

    /// In-core qubit cap for the CUDA kinds (FP64 / FP32), or `None` for any
    /// non-CUDA kind. Drives [`Reach::for_kind`]'s paging decision.
    pub fn cuda_in_core_cap(self) -> Option<u32> {
        match self {
            BackendKind::Cuda => Some(MAX_CUDA_QUBITS),
            BackendKind::CudaF32 => Some(MAX_CUDA_QUBITS_F32),
            _ => None,
        }
    }
}

/// A user's backend choice parsed from a string, shared by the CLI and the
/// Python binding so both surfaces accept exactly one vocabulary.
///
/// `Auto` defers to [`select_explained`]; `Fixed` is an explicit override.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BackendRequest {
    /// Pick automatically from circuit structure (the [`select_explained`]
    /// heuristic).
    Auto,
    /// An explicit, user-pinned backend.
    Fixed(BackendKind),
}

impl BackendRequest {
    /// The literal that selects the auto heuristic.
    pub const AUTO: &'static str = "auto";

    /// Parse one user-facing backend token: `auto`, any canonical name, or any
    /// alias (`sv`, `stab`, …). The **single parse site** used by both
    /// `aleph-cli` (a clap `value_parser`) and `aleph-py`, so a name accepted
    /// on one surface is accepted on the other.
    ///
    /// `Err` carries a ready-to-print message listing the whole vocabulary.
    pub fn from_user_str(s: &str) -> Result<Self, String> {
        if s == Self::AUTO {
            return Ok(BackendRequest::Auto);
        }
        for kind in BackendKind::ALL {
            if kind.aliases().contains(&s) {
                return Ok(BackendRequest::Fixed(kind));
            }
        }
        Err(unknown_backend_message(s))
    }

    /// Whether noise simulation may run for this request. Noise runs only on the
    /// state-vector trajectory engine, so `auto` (which uses SV for noise) and an
    /// explicit state vector are eligible; an explicit stabilizer/MPS is not.
    ///
    /// Shared by the CLI and Python so both surfaces gate noise on exactly the
    /// same set (the policy lived as two opposite-polarity `matches!` before).
    pub fn allows_noise(self) -> bool {
        matches!(
            self,
            BackendRequest::Auto | BackendRequest::Fixed(BackendKind::Statevector)
        )
    }
}

/// "unknown backend "x"; expected one of: auto, statevector (sv), …" — built
/// from [`BackendKind::ALL`]/[`BackendKind::aliases`] so it never drifts from
/// what the parser actually accepts.
fn unknown_backend_message(got: &str) -> String {
    use std::fmt::Write as _;
    let mut names = String::from(BackendRequest::AUTO);
    for kind in BackendKind::ALL {
        let aliases = kind.aliases();
        let _ = write!(names, ", {}", aliases[0]);
        if aliases.len() > 1 {
            let _ = write!(names, " ({})", aliases[1..].join(", "));
        }
    }
    format!("unknown backend {got:?}; expected one of: {names}")
}

/// Read-only structural features of a circuit, computed in a single scan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CircuitFeatures {
    /// Number of qubits the circuit declares.
    pub num_qubits: u32,
    /// Total layer count (`circuit.layers().len()`); diagnostics only.
    pub depth: usize,
    /// Number of layers containing at least one two-qubit gate.
    pub twoq_depth: usize,
    /// Every `Gate` instruction is Clifford with no external control
    /// (`Measure`/`Barrier` allowed). A controlled-Clifford (e.g. controlled-H)
    /// is not Clifford, and the stabilizer backend rejects external controls,
    /// so any `g.controls` clears this flag.
    pub all_clifford: bool,
    /// Every two-qubit gate acts on adjacent qubits (`|q0 - q1| == 1`).
    /// (vacuously true when there is no two-qubit gate; gates of other arity do not affect this flag)
    pub all_twoq_nearest_neighbor: bool,
    /// No gate exceeds the MPS backend's 1q/2q kernels: nothing acts on 3+
    /// qubits and nothing carries an external control. A Toffoli/CCZ or any
    /// controlled gate disqualifies MPS.
    pub all_gates_at_most_2q: bool,
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
    let mut all_gates_at_most_2q = true;
    for inst in insts {
        match inst {
            Instruction::Gate(g) => {
                // `is_clifford()` describes the BASE gate only: a controlled-
                // Clifford (e.g. controlled-H) is not Clifford, and the
                // stabilizer backend rejects any gate with external controls.
                // So an external control disqualifies the stabilizer route.
                if !g.gate.is_clifford() || !g.controls.is_empty() {
                    all_clifford = false;
                }
                if g.qubits.len() == 2 && g.qubits[0].abs_diff(g.qubits[1]) != 1 {
                    all_twoq_nearest_neighbor = false;
                }
                // The MPS backend addresses gates by target arity through its
                // 1q/2q kernels; a 3q+ gate or any external control spans more
                // qubits than those kernels handle, so route such gates to SV.
                if g.qubits.len() > 2 || !g.controls.is_empty() {
                    all_gates_at_most_2q = false;
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

    // Second pass: reuse the canonical layer scheduler rather than re-deriving it.
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
        all_gates_at_most_2q,
    }
}

/// A resolved backend choice plus a one-line human-readable rationale.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Selection {
    /// The chosen backend.
    pub kind: BackendKind,
    /// In-core vs out-of-core paging. Only the CUDA kinds ever page; every other
    /// kind is always [`Reach::InCore`].
    pub reach: Reach,
    /// Why this backend was chosen (for CLI diagnostics).
    pub reason: &'static str,
}

impl Selection {
    /// An in-core selection (the common case for every non-paging backend).
    const fn in_core(kind: BackendKind, reason: &'static str) -> Self {
        Self {
            kind,
            reach: Reach::InCore,
            reason,
        }
    }
}

/// Full reach-aware decision rule (P5.11-06): map features + the runtime
/// [`SelectEnv`] (GPU availability + precision preference) to a backend kind and
/// its [`Reach`]. Pure; total. The ordered rule:
///
/// 1. **Clifford** → stabilizer.
/// 2. **Exact-and-fits** (`n ≤ SV_EXACT_CAP`) → CPU FP64 state vector, or the
///    Metal GPU FP32 SV when available and `n ≥ GPU_PREFER_N` (P5.6-07). CUDA is
///    *not* auto-picked here — it is the large-circuit reach engine below; a user
///    who wants the GPU on a small circuit pins it explicitly.
/// 3. **Large, nearest-neighbor, shallow** → MPS (bounded memory).
/// 4. **Large and dense, CUDA available** → the CUDA SV at the requested precision
///    ([`Precision::Auto`] ⇒ FP32), **in core** up to its cap and **paged**
///    ([`Reach::Paged`]) beyond — the P5.10/5.11 reach lever (`n > 30` FP64 /
///    `n > 31` FP32 streams out-of-core).
/// 5. **Otherwise** → CPU FP64 state vector (the honest fallback; OOMs past ~30 q
///    with no GPU, exactly as before this policy existed).
pub fn select_from_full(f: &CircuitFeatures, env: &SelectEnv) -> Selection {
    if f.all_clifford {
        return Selection::in_core(BackendKind::Stabilizer, "all gates are Clifford");
    }
    if f.num_qubits <= SV_EXACT_CAP {
        if env.metal_available && f.num_qubits >= GPU_PREFER_N {
            return Selection::in_core(
                BackendKind::Metal,
                "large dense circuit; Metal GPU state vector (FP32) outpaces the CPU at this scale",
            );
        }
        return Selection::in_core(BackendKind::Statevector, "exact and fits in memory");
    }
    if f.all_twoq_nearest_neighbor && f.all_gates_at_most_2q && f.twoq_depth <= MPS_DEPTH_THRESHOLD
    {
        return Selection::in_core(
            BackendKind::Mps,
            "nearest-neighbor and shallow; too large for exact simulation",
        );
    }
    if env.cuda_available {
        return cuda_selection(f.num_qubits, env.precision);
    }
    Selection::in_core(
        BackendKind::Statevector,
        "too large for exact and not MPS-suitable",
    )
}

/// The CUDA reach branch: pick FP32 (default / [`Precision::Auto`]) or FP64, then
/// in-core vs paged by comparing `n` to that precision's in-core cap. The paged
/// split defaults to one qubit below the cap, leaving device headroom for the
/// streaming group buffer; callers (CLI `--tile-qubits`) may override it.
fn cuda_selection(num_qubits: u32, precision: Precision) -> Selection {
    let use_f32 = !matches!(precision, Precision::F64);
    let kind = if use_f32 {
        BackendKind::CudaF32
    } else {
        BackendKind::Cuda
    };
    let reach = Reach::for_kind(kind, num_qubits);
    let reason = match (use_f32, reach) {
        (true, Reach::Paged { .. }) => "too large for in-core; CUDA FP32 out-of-core paging",
        (false, Reach::Paged { .. }) => "too large for in-core; CUDA FP64 out-of-core paging",
        (true, Reach::InCore) => "large dense circuit; CUDA FP32 state vector",
        (false, Reach::InCore) => "large dense circuit; CUDA FP64 state vector",
    };
    Selection {
        kind,
        reach,
        reason,
    }
}

/// Apply the ordered decision rule to pre-computed features, given whether the
/// Metal GPU backend is available. Pure; total. A thin wrapper over
/// [`select_from_full`] with no CUDA and the default ([`Precision::Auto`])
/// precision, preserving the pre-P5.11-06 behaviour for existing callers.
pub fn select_from_env(f: &CircuitFeatures, metal_available: bool) -> Selection {
    select_from_full(
        f,
        &SelectEnv {
            metal_available,
            cuda_available: false,
            precision: Precision::Auto,
        },
    )
}

/// Apply the ordered decision rule to pre-computed features, assuming no GPU.
/// Pure; total. Equivalent to [`select_from_env`] with `metal_available = false`.
pub fn select_from(f: &CircuitFeatures) -> Selection {
    select_from_env(f, false)
}

/// Analyze `c` and apply the decision rule (no GPU), returning kind + rationale.
pub fn select_explained(c: &Circuit) -> Selection {
    select_from(&analyze(c))
}

/// Analyze `c` and apply the decision rule given Metal availability (P5.6-07).
pub fn select_explained_env(c: &Circuit, metal_available: bool) -> Selection {
    select_from_env(&analyze(c), metal_available)
}

/// Analyze `c` and apply the full reach-aware rule (P5.11-06) — the entry point
/// the CLI / Python use once they have probed GPU availability and a precision
/// preference.
pub fn select_explained_full(c: &Circuit, env: &SelectEnv) -> Selection {
    select_from_full(&analyze(c), env)
}

/// Analyze `c` and return the chosen backend kind (AC-exact signature).
pub fn select_backend(c: &Circuit) -> BackendKind {
    select_explained(c).kind
}

#[cfg(test)]
mod tests {
    use super::*;
    use aleph_core::{Gate, GateInstance, Param};
    use aleph_ir::Circuit;

    #[test]
    fn backend_kind_display_labels() {
        assert_eq!(BackendKind::Statevector.to_string(), "state vector");
        assert_eq!(BackendKind::Stabilizer.to_string(), "stabilizer");
        assert_eq!(BackendKind::Mps.to_string(), "MPS");
        assert_eq!(BackendKind::Metal.to_string(), "Metal GPU");
        assert_eq!(BackendKind::Cuda.to_string(), "CUDA GPU (FP64)");
        assert_eq!(BackendKind::CudaF32.to_string(), "CUDA GPU (FP32)");
    }

    #[test]
    fn caps_have_expected_values() {
        assert_eq!(SV_EXACT_CAP, 28);
        assert_eq!(MPS_DEPTH_THRESHOLD, 64);
    }

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
        c.add_gate(GateInstance::new(
            Gate::Rz(Param::Concrete(0.3)),
            vec![0u32],
        ))
        .unwrap();
        c.add_gate(GateInstance::new(
            Gate::Rz(Param::Concrete(0.3)),
            vec![1u32],
        ))
        .unwrap();
        c.add_gate(GateInstance::new(Gate::Cnot, vec![0u32, 1u32]))
            .unwrap();
        let f = analyze(&c);
        assert_eq!(f.depth, 2);
        assert_eq!(f.twoq_depth, 1);
    }

    fn feats(
        num_qubits: u32,
        twoq_depth: usize,
        all_clifford: bool,
        all_twoq_nearest_neighbor: bool,
    ) -> CircuitFeatures {
        CircuitFeatures {
            num_qubits,
            depth: twoq_depth + 3,
            twoq_depth,
            all_clifford,
            all_twoq_nearest_neighbor,
            all_gates_at_most_2q: true,
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

    // --- P5.11-06: CUDA precision + reach policy ---

    fn cuda_env(precision: Precision) -> SelectEnv {
        SelectEnv {
            metal_available: false,
            cuda_available: true,
            precision,
        }
    }

    // A large, dense, long-range circuit: too big for exact CPU, not MPS-suitable —
    // the branch the CUDA reach engine owns.
    fn large_dense(n: u32) -> CircuitFeatures {
        feats(n, 10, false, false)
    }

    #[test]
    fn cuda_auto_large_dense_picks_fp32_in_core() {
        // n within the FP32 in-core cap → CudaF32, no paging. Auto ⇒ FP32.
        let s = select_from_full(&large_dense(30), &cuda_env(Precision::Auto));
        assert_eq!(s.kind, BackendKind::CudaF32);
        assert_eq!(s.reach, Reach::InCore);
    }

    #[test]
    fn cuda_fp32_in_core_boundary_at_cap() {
        // n == MAX_CUDA_QUBITS_F32 (31) is still in core (>, not >=).
        let s = select_from_full(&large_dense(MAX_CUDA_QUBITS_F32), &cuda_env(Precision::F32));
        assert_eq!(s.kind, BackendKind::CudaF32);
        assert_eq!(s.reach, Reach::InCore);
    }

    #[test]
    fn cuda_fp32_beyond_cap_pages() {
        // n=32 > FP32 cap → paged, tile one below the cap (headroom for streaming).
        let s = select_from_full(&large_dense(32), &cuda_env(Precision::F32));
        assert_eq!(s.kind, BackendKind::CudaF32);
        assert_eq!(
            s.reach,
            Reach::Paged {
                tile_qubits: MAX_CUDA_QUBITS_F32 - 1
            }
        );
    }

    #[test]
    fn cuda_fp64_in_core_and_paged_boundaries() {
        // FP64 cap is 30: n=30 in core, n=31 pages.
        let in_core = select_from_full(&large_dense(MAX_CUDA_QUBITS), &cuda_env(Precision::F64));
        assert_eq!(in_core.kind, BackendKind::Cuda);
        assert_eq!(in_core.reach, Reach::InCore);

        let paged = select_from_full(&large_dense(MAX_CUDA_QUBITS + 1), &cuda_env(Precision::F64));
        assert_eq!(paged.kind, BackendKind::Cuda);
        assert_eq!(
            paged.reach,
            Reach::Paged {
                tile_qubits: MAX_CUDA_QUBITS - 1
            }
        );
    }

    #[test]
    fn explicit_fp64_overrides_default_fp32() {
        let s = select_from_full(&large_dense(30), &cuda_env(Precision::F64));
        assert_eq!(
            s.kind,
            BackendKind::Cuda,
            "F64 flag must force the FP64 kind"
        );
    }

    #[test]
    fn cuda_not_chosen_for_small_or_clifford() {
        // Small dense circuits keep the exact CPU path even with CUDA available.
        let small = select_from_full(&feats(20, 50, false, true), &cuda_env(Precision::Auto));
        assert_eq!(small.kind, BackendKind::Statevector);
        // Clifford always wins, regardless of GPU availability.
        let cliff = select_from_full(&feats(5000, 100, true, false), &cuda_env(Precision::Auto));
        assert_eq!(cliff.kind, BackendKind::Stabilizer);
    }

    #[test]
    fn no_cuda_large_dense_falls_to_cpu_statevector() {
        // Without CUDA the large-dense branch is the honest CPU SV fallback (legacy).
        let s = select_from_full(&large_dense(32), &SelectEnv::default());
        assert_eq!(s.kind, BackendKind::Statevector);
        assert_eq!(s.reach, Reach::InCore);
    }

    #[test]
    fn select_backend_matches_select_from() {
        let c = bell();
        assert_eq!(select_backend(&c), select_from(&analyze(&c)).kind);
        assert_eq!(select_backend(&c), BackendKind::Stabilizer);
    }

    // FIX 1 regression: a circuit whose only multi-qubit gate is Toffoli must NOT
    // route to MPS (MPS rejects 3q gates at runtime).
    #[test]
    fn rule_large_3q_gate_avoids_mps() {
        let f = CircuitFeatures {
            num_qubits: 30,
            depth: 5,
            twoq_depth: 0,
            all_clifford: false,
            all_twoq_nearest_neighbor: true,
            all_gates_at_most_2q: false,
        };
        assert_eq!(select_from(&f).kind, BackendKind::Statevector);
    }

    // FIX 1: analyze must set all_gates_at_most_2q=false when a Toffoli is present.
    #[test]
    fn analyze_toffoli_sets_not_at_most_2q() {
        let mut c = Circuit::new(4, 0);
        c.add_gate(GateInstance::new(Gate::Toffoli, vec![0u32, 1u32, 2u32]))
            .unwrap();
        let f = analyze(&c);
        assert!(
            !f.all_gates_at_most_2q,
            "Toffoli must clear all_gates_at_most_2q"
        );
        assert!(!f.all_clifford, "Toffoli is not Clifford");
    }

    // A controlled-Clifford (controlled-H) has a Clifford BASE gate but is not
    // itself Clifford, and the stabilizer/MPS backends reject external controls.
    // `analyze` must clear both `all_clifford` and `all_gates_at_most_2q` so a
    // library caller that builds such a gate is routed to the state vector.
    #[test]
    fn analyze_controlled_clifford_is_not_clifford_nor_mps() {
        let mut c = Circuit::new(2, 0);
        c.add_gate(GateInstance::controlled(Gate::H, vec![1u32], vec![0u32]))
            .unwrap();
        let f = analyze(&c);
        assert!(!f.all_clifford, "controlled-H is not Clifford");
        assert!(
            !f.all_gates_at_most_2q,
            "an external control disqualifies the MPS kernels"
        );
    }

    // Regression: a large circuit of only controlled-Clifford gates must NOT
    // route to stabilizer or MPS (both reject external controls) — it falls
    // through to the state vector.
    #[test]
    fn rule_large_controlled_clifford_avoids_stabilizer_and_mps() {
        let mut c = Circuit::new(30, 0);
        // Nearest-neighbor controlled-H ladder: Clifford base, NN, but controlled.
        for q in 0u32..29 {
            c.add_gate(GateInstance::controlled(Gate::H, vec![q + 1], vec![q]))
                .unwrap();
        }
        assert_eq!(select_backend(&c), BackendKind::Statevector);
    }

    // FIX 6: boundary — n == SV_EXACT_CAP must still pick Statevector (<=, not <).
    #[test]
    fn rule_boundary_n_equals_cap_picks_statevector() {
        let s = select_from(&feats(SV_EXACT_CAP, 10, false, true));
        assert_eq!(s.kind, BackendKind::Statevector);
    }

    // FIX 6: boundary — twoq_depth == MPS_DEPTH_THRESHOLD must still pick Mps (<=, not <).
    #[test]
    fn rule_boundary_twoq_depth_equals_threshold_picks_mps() {
        let s = select_from(&feats(30, MPS_DEPTH_THRESHOLD, false, true));
        assert_eq!(s.kind, BackendKind::Mps);
    }

    // --- P4-12: shared user-facing backend vocabulary ---

    #[test]
    fn auto_parses_to_auto() {
        assert_eq!(
            BackendRequest::from_user_str("auto"),
            Ok(BackendRequest::Auto)
        );
    }

    // Every kind must round-trip through every one of its aliases. Also guards
    // BackendKind::ALL: a variant missing from ALL is unreachable here and the
    // canonical-name assertion below fails.
    #[test]
    fn every_kind_round_trips_through_all_its_aliases() {
        for kind in BackendKind::ALL {
            for &alias in kind.aliases() {
                assert_eq!(
                    BackendRequest::from_user_str(alias),
                    Ok(BackendRequest::Fixed(kind)),
                    "alias {alias:?} must resolve to {kind:?}"
                );
            }
            // canonical_name is itself a valid alias.
            assert!(kind.aliases().contains(&kind.canonical_name()));
        }
    }

    #[test]
    fn established_aliases_resolve() {
        use BackendKind::*;
        assert_eq!(
            BackendRequest::from_user_str("sv"),
            Ok(BackendRequest::Fixed(Statevector))
        );
        assert_eq!(
            BackendRequest::from_user_str("statevector"),
            Ok(BackendRequest::Fixed(Statevector))
        );
        assert_eq!(
            BackendRequest::from_user_str("stab"),
            Ok(BackendRequest::Fixed(Stabilizer))
        );
        assert_eq!(
            BackendRequest::from_user_str("stabilizer"),
            Ok(BackendRequest::Fixed(Stabilizer))
        );
        assert_eq!(
            BackendRequest::from_user_str("mps"),
            Ok(BackendRequest::Fixed(Mps))
        );
    }

    #[test]
    fn allows_noise_only_for_auto_and_statevector() {
        use BackendKind::*;
        assert!(BackendRequest::Auto.allows_noise());
        assert!(BackendRequest::Fixed(Statevector).allows_noise());
        assert!(!BackendRequest::Fixed(Stabilizer).allows_noise());
        assert!(!BackendRequest::Fixed(Mps).allows_noise());
        // The Metal GPU backend has no noise engine (CPU-only trajectories).
        assert!(!BackendRequest::Fixed(Metal).allows_noise());
    }

    #[test]
    fn unknown_backend_lists_the_whole_vocabulary() {
        // `qsim` is not any backend alias; `cuda` IS one now (P5.11-06), so it must
        // appear in the listed vocabulary rather than be the rejected token.
        let e = BackendRequest::from_user_str("qsim").unwrap_err();
        assert!(e.contains("\"qsim\""), "echoes the bad token: {e}");
        for token in [
            "auto",
            "statevector",
            "sv",
            "stabilizer",
            "stab",
            "mps",
            "metal",
            "gpu",
            "cuda",
            "cuda-f32",
        ] {
            assert!(e.contains(token), "message must list {token:?}: {e}");
        }
    }

    // FIX 6: empty circuit is vacuously Clifford, NN, and all-at-most-2q.
    #[test]
    fn analyze_empty_circuit_is_vacuously_clifford() {
        let c = Circuit::new(3, 0);
        let f = analyze(&c);
        assert!(f.all_clifford);
        assert_eq!(f.twoq_depth, 0);
        assert!(f.all_twoq_nearest_neighbor);
        assert!(f.all_gates_at_most_2q);
        assert_eq!(select_backend(&c), BackendKind::Stabilizer);
    }
}
