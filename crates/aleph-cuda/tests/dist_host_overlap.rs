//! P6-01b: GPU work must enqueue without blocking the host. `DistSvBackend`
//! drives every device from one host thread, so a host block per instruction
//! serialises the devices. Timing-based, so it lives in its own test binary:
//! other tests sharing GPU 0's legacy stream would make the host wait on
//! *their* work.
#![cfg(all(target_os = "linux", feature = "cuda"))]

use aleph_cuda::{CudaSvBackend, CudaSvBackendF32, DeviceSv};

fn gpu64() -> Option<CudaSvBackend> {
    CudaSvBackend::with_seed(0).ok()
}

fn gpu32() -> Option<CudaSvBackendF32> {
    CudaSvBackendF32::with_seed(0).ok()
}

/// A `DiagonalPhase` must enqueue and return, not block the host until the
/// kernel drains: `DistSvBackend` issues rank programs serially from one host
/// thread, and a host block per phase polynomial would serialise the devices
/// (QFT is ~one DiagonalPhase per H after fusion).
#[test]
fn diagonal_phase_does_not_block_the_host() {
    use aleph_backend::Backend;
    use aleph_ir::{DiagonalPhase, PhaseTerm};
    use std::time::Instant;
    let Some(mut be) = gpu64() else { return };
    let n = 27; // 2 GiB FP64: one pass is milliseconds
    let mut st = be.alloc_rank(n, 0).unwrap();
    let dp = DiagonalPhase {
        n_qubits: n,
        terms: (0..400u32)
            .map(|t| PhaseTerm {
                conds: vec![1u64 << (t % n), 1u64 << ((t * 7 + 3) % n)].into(),
                angle: 0.001 * f64::from(t + 1),
            })
            .collect(),
    };
    be.apply_diagonal_phase(&mut st, &dp).unwrap(); // warm-up (JIT, pool)
    let ctx = aleph_cuda::CudaContext::new(0).unwrap();
    ctx.synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..4 {
        be.apply_diagonal_phase(&mut st, &dp).unwrap();
    }
    let host = t0.elapsed().as_secs_f64();
    ctx.synchronize().unwrap();
    let total = t0.elapsed().as_secs_f64();
    assert!(
        host < 0.5 * total,
        "host blocked: enqueue {host:.4}s of {total:.4}s total"
    );
}

#[test]
fn diagonal_phase_does_not_block_the_host_f32() {
    use aleph_backend::Backend;
    use aleph_ir::{DiagonalPhase, PhaseTerm};
    use std::time::Instant;
    let Some(mut be) = gpu32() else { return };
    let n = 28;
    let mut st = be.alloc_rank(n, 0).unwrap();
    let dp = DiagonalPhase {
        n_qubits: n,
        terms: (0..400u32)
            .map(|t| PhaseTerm {
                conds: vec![1u64 << (t % n), 1u64 << ((t * 7 + 3) % n)].into(),
                angle: 0.001 * f64::from(t + 1),
            })
            .collect(),
    };
    be.apply_diagonal_phase(&mut st, &dp).unwrap();
    let ctx = aleph_cuda::CudaContext::new(0).unwrap();
    ctx.synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..4 {
        be.apply_diagonal_phase(&mut st, &dp).unwrap();
    }
    let host = t0.elapsed().as_secs_f64();
    ctx.synchronize().unwrap();
    let total = t0.elapsed().as_secs_f64();
    assert!(
        host < 0.5 * total,
        "host blocked: enqueue {host:.4}s of {total:.4}s total"
    );
}
