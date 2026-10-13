//! Discovery must not compile shaders or construct a state-vector backend.
#![cfg(all(target_os = "macos", feature = "metal"))]
use aleph_metal::{device_available, sv_backend_init_count, MetalSvBackend};
#[test]
fn discovery_does_not_construct_a_backend() {
    let before = sv_backend_init_count();
    let available = device_available();
    assert_eq!(sv_backend_init_count(), before);
    if available {
        MetalSvBackend::with_seed(0).expect("discovered Metal device must pass smoke construction");
        assert_eq!(sv_backend_init_count(), before + 1);
        let ctx = aleph_metal::MetalContext::new().unwrap();
        assert!(matches!(
            ctx.make_compute_pipeline("invalid Metal source", "missing"),
            Err(aleph_metal::Error::ShaderCompile(_))
        ));
        // Device discovery must not conflate compilation readiness with absence.
        assert!(device_available());
    } else {
        assert!(matches!(
            aleph_metal::MetalContext::new(),
            Err(aleph_metal::Error::NoDevice)
        ));
        assert!(MetalSvBackend::with_seed(0).is_err());
        assert_eq!(sv_backend_init_count(), before + 1);
        eprintln!("no Metal device: discovery unavailable, backend smoke not executed");
    }
}
