//! Gate-validation and matrix helpers shared by the two CUDA state-vector
//! backends — the hand-written NVRTC kernels ([`crate::sv`]) and the cuStateVec
//! integration ([`crate::cuquantum`]). Keeping the validation in one place means
//! both backends reject the exact same inputs and pull amplitudes through the
//! identical unitarity guard, so the oracle suite that pins one also pins the
//! other (ADR 0006 finiteness reject included).

use aleph_backend::BackendError;
use aleph_core::{Complex, GateError, GateInstance, GateMatrix};

/// `max_{i,j} |(U·U†)_{i,j} - δ_{i,j}|`, mirroring `aleph_sv::validation`
/// (crate-private there). NaN-propagating: any NaN entry yields NaN so the
/// caller's `is_finite` reject fires (ADR 0006).
pub(crate) fn unitarity_deviation(matrix: &GateMatrix) -> f64 {
    fn max_dev<const N: usize>(m: &[[Complex; N]; N]) -> f64 {
        let mut worst = 0.0_f64;
        for (i, row_i) in m.iter().enumerate() {
            for (j, row_j) in m.iter().enumerate() {
                let mut acc = Complex::new(0.0, 0.0);
                for (a, b) in row_i.iter().zip(row_j.iter()) {
                    acc += a * b.conj();
                }
                let want = if i == j { 1.0 } else { 0.0 };
                let dev = (acc - Complex::new(want, 0.0)).norm();
                if dev.is_nan() {
                    return f64::NAN;
                }
                if dev > worst {
                    worst = dev;
                }
            }
        }
        worst
    }
    match matrix {
        GateMatrix::M2x2(m) => max_dev::<2>(m),
        GateMatrix::M4x4(m) => max_dev::<4>(m),
        GateMatrix::M8x8(m) => max_dev::<8>(m),
    }
}

/// Control-qubit bitmask `Σ 1 << c`.
pub(crate) fn control_mask(controls: &[u32]) -> u32 {
    controls.iter().fold(0u32, |acc, &c| acc | (1u32 << c))
}

/// If `matrix` is diagonal — every off-diagonal entry within [`aleph_core::AMPLITUDE_TOL`]
/// of zero — return its diagonal as the interleaved `[re, im]` buffer
/// (`[d00.re, d00.im, d11.re, d11.im, …]`, length `2·dim`); otherwise `None`.
///
/// Diagonal gates (Z, S, T, Rz, Phase, and their controlled forms CZ / CPhase /
/// multi-controlled Z) get routed to the custom `apply_diag` kernels (P5-06),
/// which beat both the dense `apply_kq` and cuStateVec's generic apply. The test
/// is numeric, not gate-type-based, so it stays backend-agnostic (CLAUDE.md:
/// "don't hardcode gate types in kernels") and catches any diagonal unitary.
pub(crate) fn diagonal_of(matrix: &GateMatrix) -> Option<Vec<f64>> {
    fn check<const N: usize>(m: &[[Complex; N]; N]) -> Option<Vec<f64>> {
        for (i, row) in m.iter().enumerate() {
            for (j, z) in row.iter().enumerate() {
                // NaN norms compare false here, but `validate_and_extract` has
                // already rejected non-finite matrices, so entries are finite.
                if i != j && z.norm() > aleph_core::AMPLITUDE_TOL {
                    return None;
                }
            }
        }
        let mut diag = Vec::with_capacity(2 * N);
        for (i, row) in m.iter().enumerate() {
            diag.push(row[i].re);
            diag.push(row[i].im);
        }
        Some(diag)
    }
    match matrix {
        GateMatrix::M2x2(m) => check(m),
        GateMatrix::M4x4(m) => check(m),
        GateMatrix::M8x8(m) => check(m),
    }
}

/// Row-major interleaved `[re, im]` of an `N×N` complex matrix.
pub(crate) fn flatten_matrix<const N: usize>(m: &[[Complex; N]; N]) -> Vec<f64> {
    let mut out = Vec::with_capacity(2 * N * N);
    for row in m {
        for z in row {
            out.push(z.re);
            out.push(z.im);
        }
    }
    out
}

/// Row-major interleaved `[re, im]` of a flat complex slice. `UnitaryKq` carries
/// its `2^k × 2^k` matrix as a raw slice (the fixed-size [`flatten_matrix`] only
/// covers `k ≤ 3`, since `GateMatrix` stops at 8×8), so the k=4,5 GPU path
/// flattens through here instead.
pub(crate) fn flatten_kq(data: &[Complex]) -> Vec<f64> {
    let mut out = Vec::with_capacity(2 * data.len());
    for z in data {
        out.push(z.re);
        out.push(z.im);
    }
    out
}

/// Structural validation for a fused `UnitaryKq` block against an `n`-qubit
/// state (P5.9-02). Mirrors the operand checks in [`validate_and_extract`] —
/// `k` in 2..=5, arity == `k`, the `2^k × 2^k` data length, every operand and
/// control in range and distinct — but skips matrix materialisation and the
/// unitarity guard: `UnitaryKq` has no fixed-size `GateMatrix` to check, and it
/// is emitted only by `passes::FuseKq` from already-unitary members (matching
/// the CPU SV backend, which also intercepts `UnitaryKq` before its guard).
/// Both GPU backends call this so they admit identical fused blocks.
pub(crate) fn validate_kq(
    num_qubits: u32,
    k: u8,
    data_len: usize,
    qubits: &[u32],
    controls: &[u32],
) -> Result<(), BackendError> {
    let k = k as usize;
    if !(2..=5).contains(&k) {
        return Err(BackendError::UnsupportedGate { kind: "UnitaryKq" });
    }
    if qubits.len() != k {
        return Err(BackendError::ArityMismatch {
            kind: "UnitaryKq",
            expected: k,
            got: qubits.len(),
        });
    }
    // data is the row-major 2^k × 2^k matrix ⇒ 2^(2k) complex entries.
    if data_len != 1usize << (2 * k) {
        return Err(BackendError::InvalidState {
            reason: "UnitaryKq data length is not 2^(2k)",
        });
    }
    let mut seen: smallvec::SmallVec<[u32; 6]> = smallvec::SmallVec::new();
    for &q in qubits.iter().chain(controls.iter()) {
        if q >= num_qubits {
            return Err(BackendError::QubitOutOfRange {
                qubit: q,
                num_qubits,
            });
        }
        if seen.contains(&q) {
            return Err(BackendError::DuplicateQubit { qubit: q });
        }
        seen.push(q);
    }
    Ok(())
}

/// Validate a gate against an `n`-qubit state and return its dense matrix.
///
/// Checks (in order, matching the original `CudaSvBackend::apply_gate`): arity,
/// every operand/control qubit in range and distinct, the matrix is
/// representable, and it is unitary within [`aleph_core::AMPLITUDE_TOL`] with a
/// finiteness reject. Both GPU backends call this so they reject identical
/// inputs.
///
/// Fused `UnitaryKq` blocks have no fixed-size `GateMatrix`, so both backends
/// intercept them *before* this call and route through [`validate_kq`] +
/// `flatten_kq` (P5.9-02). The `UnitaryKq` arm here stays as a safety net for
/// any path that forgets to intercept: it surfaces `UnsupportedGate` rather
/// than mis-applying a block whose matrix `gate.matrix()` cannot return.
pub(crate) fn validate_and_extract(
    num_qubits: u32,
    gate: &GateInstance,
) -> Result<GateMatrix, BackendError> {
    let expected = gate.gate.arity();
    let got = gate.qubits.len();
    if expected != got {
        return Err(BackendError::ArityMismatch {
            kind: gate.gate.name(),
            expected,
            got,
        });
    }
    let mut seen: smallvec::SmallVec<[u32; 6]> = smallvec::SmallVec::new();
    for &q in gate.qubits.iter().chain(gate.controls.iter()) {
        if q >= num_qubits {
            return Err(BackendError::QubitOutOfRange {
                qubit: q,
                num_qubits,
            });
        }
        if seen.contains(&q) {
            return Err(BackendError::DuplicateQubit { qubit: q });
        }
        seen.push(q);
    }
    if matches!(gate.gate, aleph_core::Gate::UnitaryKq { .. }) {
        return Err(BackendError::UnsupportedGate {
            kind: gate.gate.name(),
        });
    }
    let matrix = gate.gate.matrix().map_err(|e| match e {
        GateError::SymbolicParam => BackendError::SymbolicParam,
        GateError::NonFiniteParam => BackendError::NonFiniteParam {
            kind: gate.gate.name(),
        },
        GateError::Unrepresentable => BackendError::UnsupportedGate {
            kind: gate.gate.name(),
        },
    })?;
    let deviation = unitarity_deviation(&matrix);
    if !deviation.is_finite() || deviation > aleph_core::AMPLITUDE_TOL {
        return Err(BackendError::NonUnitaryMatrix { deviation });
    }
    Ok(matrix)
}
