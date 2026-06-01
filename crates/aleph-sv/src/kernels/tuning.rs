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
pub struct ChunkPolicy {
    pub min_amps: usize,
    pub grain: usize,
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

/// Target-position buckets. Constants are design choices, not tuned.
/// `+ HIGH_BAND` (not `n - HIGH_BAND`) avoids underflow for small `n`.
const LOW_BAND: u32 = 2;
const HIGH_BAND: u32 = 2;

/// Classify by the *dominant* (maximum) target index — that governs the
/// outer stride and whether `par_units` flattening is in play.
pub(crate) fn pos_class(max_target: u32, n: u32) -> PosClass {
    debug_assert!(max_target < n, "max_target {max_target} must be < n {n}");
    if max_target < LOW_BAND {
        PosClass::Low
    } else if max_target + HIGH_BAND >= n {
        PosClass::High
    } else {
        PosClass::Mid
    }
}

/// The per-CPU chunk-policy table.
///
/// Empirical result (P2-04 sweep, 2026-06-01, `scripts/tune-chunks.sh`,
/// raw data in `docs/perf/data/p2-04-tune-{epyc,ryzen}.log`; analysis in
/// `docs/perf/phase2-p2-04.md`): **every cell on both reference CPUs is
/// within ~0.4 % (noise) of `grain = 64`**, the pre-P2-04 default, so no
/// cell is encoded away from `DEFAULT_POLICY`. The genuine signals were
/// negative — they confirm the default rather than beat it:
///   * `grain >= 256` *regresses* the stride-heavy AVX-512 kernels
///     (cnot/cphase/zdiag at a mid target) by 8–15 % on EPYC; `grain = 64`
///     sits safely in the optimal band.
///   * `min_amps` is inert at perf-relevant sizes: at n >= ~21 the state
///     length always exceeds any sane cutoff, so the kernel is always
///     parallel and the cutoff never fires. It only gates tiny states,
///     which are microseconds regardless.
///   * Ryzen (scalar, no AVX-512) is bandwidth-bound and totally flat
///     across the whole grid — consistent with P2-02/P2-03.
///
/// The table therefore returns `DEFAULT_POLICY` for all current CPUs. The
/// CPU-dispatch machinery is retained as the wired, tested hook for a
/// future CPU or kernel that *does* show chunk sensitivity — adding a
/// tuned cell is a one-line match arm, re-measured via the same sweep.
pub(crate) fn chunk_policy(cpu: RefCpu, _class: GateClass, _pos: PosClass) -> ChunkPolicy {
    match cpu {
        // All arms measured == DEFAULT_POLICY (grain 64 near-optimal; see
        // doc above). Differentiated cells go here once a sweep finds one.
        RefCpu::Generic | RefCpu::Epyc8124P | RefCpu::Ryzen3900 => DEFAULT_POLICY,
    }
}

/// Resolve the effective policy for a kernel invocation. Precedence:
/// test override → env per-field override → table.
#[inline]
pub(crate) fn resolve_policy(class: GateClass, pos: PosClass) -> ChunkPolicy {
    #[cfg(any(test, feature = "internal-bench"))]
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

/// Returns the parsed value of `ALEPH_PAR_MIN_AMPS` env var, if set and valid.
fn env_min_amps() -> Option<usize> {
    static V: OnceLock<Option<usize>> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("ALEPH_PAR_MIN_AMPS")
            .ok()
            .and_then(|s| s.parse().ok())
    })
}

/// Returns the parsed value of `ALEPH_PAR_GRAIN` env var, if set and valid.
fn env_grain() -> Option<usize> {
    static V: OnceLock<Option<usize>> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("ALEPH_PAR_GRAIN")
            .ok()
            .and_then(|s| s.parse().ok())
    })
}

/// Detect and cache the reference CPU model for this process.
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
    // The brand-string leaves 0x8000_0002..=0x8000_0004 are valid only when
    // the max extended leaf (0x8000_0000 -> eax) is >= 0x8000_0004; guard first.
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
    let s = String::from_utf8_lossy(&bytes);
    Some(s.trim_end_matches(['\0', ' ']).to_string())
}

#[cfg(not(target_arch = "x86_64"))]
fn cpu_brand_string() -> Option<String> {
    None
}

/// Test-only policy override (thread-local), so a later invariance test
/// can force several policies in one process without fighting the env
/// `OnceLock`s.
///
/// Exposed publicly under `internal-bench` so integration tests (which
/// compile as a separate crate) can reach `test_override::set/get`
/// without `cfg(test)` being in scope.
#[cfg(any(test, feature = "internal-bench"))]
pub mod test_override {
    use super::ChunkPolicy;
    use std::cell::Cell;
    thread_local! {
        static OVERRIDE: Cell<Option<ChunkPolicy>> = const { Cell::new(None) };
    }
    #[allow(dead_code)] // used from integration tests under internal-bench
    pub fn set(p: Option<ChunkPolicy>) {
        OVERRIDE.with(|c| c.set(p));
    }
    pub fn get() -> Option<ChunkPolicy> {
        OVERRIDE.with(|c| c.get())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generic_cell_is_the_legacy_default() {
        for class in [
            GateClass::OneQGeneric,
            GateClass::OneQDiag,
            GateClass::OneQAntidiag,
            GateClass::TwoQDense,
            GateClass::TwoQCnot,
            GateClass::TwoQCz,
            GateClass::TwoQSwap,
            GateClass::TwoQDiag,
            GateClass::ThreeQ,
        ] {
            for pos in [PosClass::Low, PosClass::Mid, PosClass::High] {
                assert_eq!(chunk_policy(RefCpu::Generic, class, pos), DEFAULT_POLICY);
            }
        }
    }

    // Empirical conclusion of the P2-04 sweep: grain=64 (the default) is
    // within noise of optimal on both reference CPUs, so their high-traffic
    // cells are left at DEFAULT_POLICY. This guards against silently
    // encoding a different value without re-running scripts/tune-chunks.sh.
    #[test]
    fn reference_cpu_high_traffic_cells_measured_as_default() {
        let high_traffic = [
            (GateClass::OneQGeneric, PosClass::Low),
            (GateClass::OneQDiag, PosClass::Mid),
            (GateClass::TwoQCnot, PosClass::Mid),
            (GateClass::TwoQDiag, PosClass::High),
        ];
        for cpu in [RefCpu::Epyc8124P, RefCpu::Ryzen3900] {
            for (class, pos) in high_traffic {
                assert_eq!(chunk_policy(cpu, class, pos), DEFAULT_POLICY);
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
        // n=1 (single-qubit circuit): the only target (0) classifies as Low
        // because the Low check (max_target < LOW_BAND) fires before High.
        // Documented, not a bug — chunk policy for n=1 is moot (state is
        // 2 amplitudes, always sequential).
        assert_eq!(pos_class(0, 1), PosClass::Low);
        assert_eq!(pos_class(0, 2), PosClass::Low);
        assert_eq!(pos_class(1, 2), PosClass::Low);
    }

    #[test]
    fn cpu_model_env_override() {
        assert_eq!(detect_cpu_from(Some("epyc"), None), RefCpu::Epyc8124P);
        assert_eq!(detect_cpu_from(Some("ryzen"), None), RefCpu::Ryzen3900);
        assert_eq!(detect_cpu_from(Some("generic"), None), RefCpu::Generic);
        assert_eq!(detect_cpu_from(Some("nonsense"), None), RefCpu::Generic);
    }

    #[test]
    fn cpu_model_brand_match() {
        assert_eq!(
            detect_cpu_from(None, Some("AMD EPYC 8124P 16-Core Processor")),
            RefCpu::Epyc8124P
        );
        assert_eq!(
            detect_cpu_from(None, Some("AMD Ryzen 9 3900 12-Core Processor")),
            RefCpu::Ryzen3900
        );
        assert_eq!(
            detect_cpu_from(None, Some("Intel(R) Xeon(R) Silver 4114")),
            RefCpu::Generic
        );
    }
}
