//! Checked dense-state memory planning, independent of any device implementation.
//! Callers supply the allocator/readout layout and a runtime capacity snapshot.

/// Requested views and the concrete executor's scratch requirements.
#[derive(Clone, Copy, Debug, Default)]
pub struct DenseMemoryRequest {
    pub qubits: u32,
    /// Bytes per complex amplitude (not per scalar).
    pub amplitude_bytes: u64,
    pub shots: u64,
    pub expectation: bool,
    pub marginal_qubits: Option<u32>,
    pub export_amplitudes: bool,
    /// Persistent gate scratch plus transient execution buffers, including pool retention.
    pub execution_scratch_bytes: u64,
    /// In-circuit measurement/reset also needs reductions when no view requests them.
    pub measurements: bool,
}

/// Concrete executor's readout layout; no backend implementation types appear here.
#[derive(Clone, Copy, Debug)]
pub struct DenseMemoryLayout {
    pub sample_bytes_per_amplitude: u64,
    pub reduction_block: u64,
    pub reduction_bytes_per_block: u64,
    pub accumulator_bytes: u64,
}

/// Available incremental memory, after accounting for existing live/cached allocations.
/// Snapshots are advisory: other processes can allocate after the check.
#[derive(Clone, Copy, Debug)]
pub struct MemoryBudget {
    pub device_free_bytes: u64,
    pub device_total_bytes: u64,
    pub device_headroom_bytes: u64,
    pub host_available_bytes: u64,
    pub host_headroom_bytes: u64,
}

/// Explainable allocation breakdown. Sequential readout scratch is summed conservatively
/// because a retaining pool may keep each stage's freed blocks reserved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseMemoryEstimate {
    pub state_bytes: u64,
    pub execution_scratch_bytes: u64,
    pub reduction_bytes: u64,
    pub sampling_bytes: u64,
    pub marginal_bytes: u64,
    pub device_peak_bytes: u64,
    pub host_peak_bytes: u64,
}

fn overflow() -> String {
    "dense memory estimate overflow: reduce qubits, shots or requested output".into()
}
fn mul(a: u64, b: u64) -> Result<u64, String> {
    a.checked_mul(b).ok_or_else(overflow)
}
fn sum(values: &[u64]) -> Result<u64, String> {
    values
        .iter()
        .try_fold(0u64, |a, b| a.checked_add(*b).ok_or_else(overflow))
}
fn pow2(n: u32) -> Result<u64, String> {
    1u64.checked_shl(n).ok_or_else(overflow)
}

/// Estimate incremental allocation before constructing a state. No allocation proportional
/// to qubits or shots occurs here. Host counts use a documented conservative 512+2*n bytes
/// per distinct outcome (bounded by shots), plus targets/sample vectors.
pub fn estimate_dense_memory(
    r: DenseMemoryRequest,
    l: DenseMemoryLayout,
) -> Result<DenseMemoryEstimate, String> {
    if !matches!(r.amplitude_bytes, 8 | 16) || l.reduction_block == 0 {
        return Err("invalid dense memory layout".into());
    }
    let n = pow2(r.qubits)?;
    // Marginals refer to a subset of this state, not a larger register.
    if r.marginal_qubits.is_some_and(|k| k > r.qubits) {
        return Err("marginal width exceeds state qubits".into());
    }
    let state_bytes = mul(n, r.amplitude_bytes)?;
    if state_bytes > isize::MAX as u64 {
        return Err(overflow());
    }
    let reduce = r.measurements || r.shots > 0 || r.expectation || r.marginal_qubits.is_some();
    let reduction_bytes = sum(&[
        l.accumulator_bytes,
        if reduce {
            mul(n.div_ceil(l.reduction_block), l.reduction_bytes_per_block)?
        } else {
            0
        },
    ])?;
    let sampling_bytes = if r.shots > 0 {
        sum(&[mul(n, l.sample_bytes_per_amplitude)?, mul(r.shots, 16)?])?
    } else {
        0
    };
    let marginal_bytes = match r.marginal_qubits {
        Some(k) if k > 0 => sum(&[mul(pow2(k)?, 8)?, mul(k as u64, 4)?])?,
        _ => 0,
    };
    let counts_host = sum(&[
        mul(r.shots.min(n), sum(&[512, 2 * r.qubits as u64])?)?,
        mul(r.shots, 16)?,
    ])?;
    let marginal_host = if let Some(k) = r.marginal_qubits {
        mul(pow2(k)?, 8)?
    } else {
        0
    };
    // FP32 widening/FP64 conversion and cached amplitudes + NumPy output each peak
    // at at most two complex128 arrays. Input IR and already-retained results are
    // excluded: runtime host availability already accounts for them.
    let export_host = if r.export_amplitudes { mul(n, 32)? } else { 0 };
    Ok(DenseMemoryEstimate {
        state_bytes,
        execution_scratch_bytes: r.execution_scratch_bytes,
        reduction_bytes,
        sampling_bytes,
        marginal_bytes,
        device_peak_bytes: sum(&[
            state_bytes,
            r.execution_scratch_bytes,
            reduction_bytes,
            sampling_bytes,
            marginal_bytes,
        ])?,
        host_peak_bytes: sum(&[
            counts_host,
            marginal_host,
            export_host,
            r.execution_scratch_bytes,
        ])?,
    })
}

impl DenseMemoryEstimate {
    /// Reject infeasible requests with the limiting allocations. A fit does not reserve memory
    /// or guarantee allocation success; callers must retain normal runtime error handling.
    pub fn check(self, b: MemoryBudget) -> Result<Self, String> {
        let device = b
            .device_free_bytes
            .min(b.device_total_bytes)
            .saturating_sub(b.device_headroom_bytes);
        if self.device_peak_bytes > device {
            return Err(format!("CUDA memory plan needs {} device bytes (state {}, execution scratch {}, reductions {}, sampling {}, marginals {}), available {} after headroom {} (free {}, total {}). Requested outputs cannot fit; paged execution is norm-only and cannot replace counts/expectation/marginals/export. Reduce qubits/shots or use a larger/free device.", self.device_peak_bytes, self.state_bytes, self.execution_scratch_bytes, self.reduction_bytes, self.sampling_bytes, self.marginal_bytes, device, b.device_headroom_bytes, b.device_free_bytes, b.device_total_bytes));
        }
        let host = b.host_available_bytes.saturating_sub(b.host_headroom_bytes);
        if self.host_peak_bytes > host {
            return Err(format!("CUDA memory plan needs {} incremental host bytes for requested outputs, available {} after headroom {}; reduce shots/export size", self.host_peak_bytes, host, b.host_headroom_bytes));
        }
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const GIB: u64 = 1 << 30;
    fn layout() -> DenseMemoryLayout {
        DenseMemoryLayout {
            sample_bytes_per_amplitude: 16,
            reduction_block: 256,
            reduction_bytes_per_block: 16,
            accumulator_bytes: 16,
        }
    }
    fn request(n: u32) -> DenseMemoryRequest {
        DenseMemoryRequest {
            qubits: n,
            amplitude_bytes: 8,
            ..Default::default()
        }
    }
    fn budget(gib: u64) -> MemoryBudget {
        MemoryBudget {
            device_free_bytes: gib * GIB,
            device_total_bytes: gib * GIB,
            device_headroom_bytes: 0,
            host_available_bytes: 256 * GIB,
            host_headroom_bytes: 0,
        }
    }
    #[test]
    fn counts_do_not_fit_reference_card_despite_state_fitting() {
        for n in [30, 31] {
            let mut r = request(n);
            r.shots = 1024;
            let e = estimate_dense_memory(r, layout()).unwrap();
            assert_eq!(e.sampling_bytes, (1u64 << n) * 16 + 16384);
            assert!(e.check(budget(20)).unwrap_err().contains("sampling"));
            r.shots = 0;
            assert!(estimate_dense_memory(r, layout())
                .unwrap()
                .check(budget(20))
                .is_ok());
        }
    }
    #[test]
    fn runtime_sizes_and_free_memory_drive_fit() {
        let r = request(31);
        let e = estimate_dense_memory(r, layout()).unwrap();
        assert!(e.check(budget(8)).is_err());
        assert!(e.check(budget(80)).is_ok());
        let mut b = budget(80);
        b.device_free_bytes = GIB;
        assert!(e.check(b).is_err());
        b = budget(80);
        b.device_free_bytes = e.device_peak_bytes;
        b.device_headroom_bytes = 1;
        assert!(e.check(b).is_err());
        b.device_headroom_bytes = 0;
        assert!(e.check(b).is_ok());
        let mut r = r;
        r.amplitude_bytes = 16;
        assert!(estimate_dense_memory(r, layout())
            .unwrap()
            .check(budget(80))
            .is_ok());
    }
    #[test]
    fn requested_views_scratch_and_host_limits_are_accounted() {
        let mut r = request(24);
        r.expectation = true;
        r.marginal_qubits = Some(20);
        r.export_amplitudes = true;
        r.execution_scratch_bytes = 4096;
        let e = estimate_dense_memory(r, layout()).unwrap();
        assert_eq!(e.marginal_bytes, 8 * (1 << 20) + 80);
        assert_eq!(e.reduction_bytes, 16 + 16 * (1 << 16));
        assert_eq!(e.host_peak_bytes, 32 * (1 << 24) + 8 * (1 << 20) + 4096);
        let mut b = budget(20);
        b.host_available_bytes = GIB / 4;
        assert!(e.check(b).unwrap_err().contains("host"));
        r.shots = 1;
        assert_eq!(
            estimate_dense_memory(r, layout()).unwrap().sampling_bytes,
            16 * (1 << 24) + 16
        );
    }
    #[test]
    fn counts_reach_scales_with_capacity_and_precision() {
        for bytes in [8, 16] {
            for n in 0..=35 {
                let mut r = request(n);
                r.amplitude_bytes = bytes;
                r.shots = 32;
                let e = estimate_dense_memory(r, layout()).unwrap();
                for gib in [8, 20, 80] {
                    assert_eq!(
                        e.check(budget(gib)).is_ok(),
                        e.device_peak_bytes <= gib * GIB,
                        "n={n}, bytes={bytes}, budget={gib}"
                    );
                }
            }
        }
        let mut r = request(31);
        r.amplitude_bytes = 16;
        r.shots = 32;
        assert!(estimate_dense_memory(r, layout())
            .unwrap()
            .check(budget(80))
            .is_ok());
        let mut r = request(29);
        r.shots = 32;
        assert!(estimate_dense_memory(r, layout())
            .unwrap()
            .check(budget(8))
            .is_err());
        assert!(estimate_dense_memory(r, layout())
            .unwrap()
            .check(budget(20))
            .is_ok());
    }

    #[test]
    fn overflow_and_invalid_layouts_are_errors() {
        for n in [60, 64, 65535] {
            assert!(estimate_dense_memory(request(n), layout()).is_err());
        }
        let mut r = request(1);
        r.shots = u64::MAX;
        assert!(estimate_dense_memory(r, layout()).is_err());
        r = request(2);
        r.marginal_qubits = Some(3);
        assert!(estimate_dense_memory(r, layout()).is_err());
        r = request(2);
        r.execution_scratch_bytes = u64::MAX;
        assert!(estimate_dense_memory(r, layout()).is_err());
    }
}
