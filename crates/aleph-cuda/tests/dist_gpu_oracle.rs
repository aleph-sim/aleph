//! P6-01a: distributed SV on one GPU (DistSvBackend + LocalExchange) vs the
//! CPU oracle. Skips without a CUDA device.
#![cfg(all(target_os = "linux", feature = "cuda"))]

use aleph_core::Complex;
use aleph_cuda::{CudaSvBackend, CudaSvBackendF32, DeviceSv};

fn gpu64() -> Option<CudaSvBackend> {
    match CudaSvBackend::with_seed(0) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping dist GPU test: {e}");
            None
        }
    }
}

fn gpu32() -> Option<CudaSvBackendF32> {
    match CudaSvBackendF32::with_seed(0) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping dist GPU test: {e}");
            None
        }
    }
}

fn ramp(len: usize, salt: f64) -> Vec<Complex<f64>> {
    (0..len)
        .map(|i| Complex::new(i as f64 * 0.001 + salt, -(i as f64) * 0.002))
        .collect()
}

#[test]
fn device_sv_alloc_copy_roundtrip_f64() {
    let Some(mut be) = gpu64() else { return };
    let r0 = be.alloc_rank(5, 0).unwrap();
    let r1 = be.alloc_rank(5, 1).unwrap();
    let d0 = be.download(&r0).unwrap();
    let d1 = be.download(&r1).unwrap();
    assert_eq!(d0.len(), 32);
    assert_eq!(d0[0], Complex::new(1.0, 0.0));
    assert!(d0[1..].iter().all(|a| *a == Complex::new(0.0, 0.0)));
    assert!(d1.iter().all(|a| *a == Complex::new(0.0, 0.0)));

    let src = be.upload(5, &ramp(32, 0.5)).unwrap();
    let mut dst = be.alloc_rank(5, 3).unwrap();
    be.copy_amps(&src, 4, &mut dst, 20, 8).unwrap();
    let got = be.download(&dst).unwrap();
    let want = ramp(32, 0.5);
    for i in 0..32 {
        let w = if (20..28).contains(&i) {
            want[i - 16]
        } else {
            Complex::new(0.0, 0.0)
        };
        assert_eq!(got[i], w, "amp {i}");
    }
}

#[test]
fn device_sv_alloc_copy_roundtrip_f32() {
    let Some(mut be) = gpu32() else { return };
    let src = be.upload(4, &ramp(16, 0.25)).unwrap();
    let mut dst = be.alloc_rank(4, 1).unwrap();
    be.copy_amps(&src, 0, &mut dst, 8, 8).unwrap();
    let got = be.download(&dst).unwrap();
    let want = ramp(16, 0.25);
    for i in 8..16 {
        assert!((got[i] - want[i - 8]).norm() < 1e-6, "amp {i}");
    }
    assert!(got[..8].iter().all(|a| a.norm() == 0.0));
    let r0 = be.alloc_rank(4, 0).unwrap();
    assert!((be.download(&r0).unwrap()[0] - Complex::new(1.0, 0.0)).norm() < 1e-7);
}
