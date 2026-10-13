//! Public CLI regression: CPU/Clifford routes must not initialize a GPU SV backend.
use aleph_backend::{BackendKind, BackendRequest};
use aleph_cli::{cli::Precision, exec::run_circuit};
use std::path::Path;

fn gpu_inits() -> u64 {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        aleph_metal::sv_backend_init_count()
    }
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    {
        aleph_cuda::sv_backend_init_count()
    }
    #[cfg(not(any(
        all(target_os = "macos", feature = "metal"),
        all(target_os = "linux", feature = "cuda")
    )))]
    {
        0
    }
}

#[test]
fn public_cpu_and_clifford_runs_do_not_construct_gpu_backends() {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/circuits/bell_phi_plus.qasm");
    let before = gpu_inits();
    for request in [
        BackendRequest::Fixed(BackendKind::Statevector),
        BackendRequest::Fixed(BackendKind::Stabilizer),
        BackendRequest::Fixed(BackendKind::Mps),
        BackendRequest::Auto,
    ] {
        let mut output = Vec::new();
        run_circuit(
            &path,
            Some(32),
            false,
            false,
            &[],
            Some(7),
            Precision::Auto,
            request,
            128,
            None,
            &[],
            &mut output,
        )
        .unwrap();
        assert!(!output.is_empty());
        assert_eq!(gpu_inits(), before, "CPU route constructed GPU backend");
    }
    // Auto's Clifford -> SV downgrade must also avoid device probing.
    run_circuit(
        &path,
        None,
        true,
        false,
        &[],
        Some(7),
        Precision::Auto,
        BackendRequest::Auto,
        128,
        None,
        &[],
        &mut Vec::new(),
    )
    .unwrap();
    assert_eq!(gpu_inits(), before);
}
