//! GPU discovery is independent of NVRTC/backend readiness.
#![cfg(all(target_os = "linux", feature = "cuda"))]
use aleph_cuda::{device_available, sv_backend_init_count, CudaSvBackendF32};
#[test]
fn discovery_does_not_construct_a_backend() {
    let before = sv_backend_init_count();
    let available = device_available();
    assert_eq!(sv_backend_init_count(), before);
    if available {
        CudaSvBackendF32::with_seed(0).expect("CUDA hardware smoke requires working NVRTC kernels");
        assert_eq!(sv_backend_init_count(), before + 1);
        assert!(cudarc::nvrtc::compile_ptx("invalid CUDA source").is_err());
        assert!(device_available());
    } else {
        assert!(CudaSvBackendF32::with_seed(0).is_err());
        assert_eq!(sv_backend_init_count(), before + 1);
        eprintln!("no CUDA device/driver: discovery unavailable, backend smoke not executed");
    }
}
