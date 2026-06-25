//! P5.11-06: the in-core qubit caps are duplicated in `aleph-backend`'s reach
//! policy (which must not depend on `aleph-cuda`). This pins the two copies so a
//! change to the real caps can't silently desync the selector's paging thresholds.

#![cfg(all(target_os = "linux", feature = "cuda"))]

#[test]
fn cuda_caps_match_backend_policy() {
    assert_eq!(
        aleph_cuda::MAX_CUDA_QUBITS,
        aleph_backend::MAX_CUDA_QUBITS,
        "FP64 in-core cap drifted from the aleph-backend reach policy"
    );
    assert_eq!(
        aleph_cuda::MAX_CUDA_QUBITS_F32,
        aleph_backend::MAX_CUDA_QUBITS_F32,
        "FP32 in-core cap drifted from the aleph-backend reach policy"
    );
}
