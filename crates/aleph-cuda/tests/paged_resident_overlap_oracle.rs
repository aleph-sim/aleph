//! P8-14 full-state event-order and residency checks on forced small tiles.
#![cfg(all(target_os = "linux", feature = "cuda"))]
use aleph_backend::run;
use aleph_core::{Gate, GateInstance, Param};
use aleph_cuda::CudaSvBackend;
use aleph_ir::Circuit;
use aleph_oracle::HasAmplitudes;
use aleph_sv::NaiveSvBackend;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn backend() -> Option<CudaSvBackend> {
    match CudaSvBackend::with_seed(564) {
        Ok(b) => Some(b),
        Err(e) => {
            assert!(
                std::env::var_os("ALEPH_REQUIRE_CUDA").is_none(),
                "required device: {e}"
            );
            eprintln!("no GPU: {e}");
            None
        }
    }
}

#[test]
fn resident_pipeline_matches_cpu_and_both_paging_references() {
    let Some(mut gpu) = backend() else { return };
    let mut rng = StdRng::seed_from_u64(564);
    for n in [5, 8, 10] {
        let mut c = Circuit::new(n, 0);
        for _ in 0..4 {
            for q in 0..n {
                c.ry(rng.gen::<f64>(), q).unwrap();
            }
            c.barrier(0..n).unwrap();
            for q in 0..n - 1 {
                c.cnot(q, q + 1).unwrap();
            }
            c.add_gate(GateInstance::controlled(
                Gate::Phase(Param::Concrete(0.31)),
                vec![0],
                vec![n - 1],
            ))
            .unwrap();
            c.cnot(n - 2, n - 1).unwrap();
            // Generic matrix uploads exercise per-slot scratch on both streams.
            c.swap(0, n - 1).unwrap();
            c.ccx(0, n - 2, n - 1).unwrap();
            c.rx(-0.71, 0).unwrap();
        }
        let cpu = HasAmplitudes::amplitudes(&run(&mut NaiveSvBackend::with_seed(564), &c).unwrap());
        for m in [1, n - 3, n - 1] {
            let sync = gpu.run_paged(&c, m).unwrap().amplitudes_vec();
            let resident = gpu.run_paged_batched(&c, m).unwrap().amplitudes_vec();
            for depth in [2, 3, 4, 7] {
                let combined = gpu
                    .run_paged_batched_overlapped(&c, m, depth)
                    .unwrap()
                    .amplitudes_vec();
                for (((a, b), d), e) in combined.iter().zip(&cpu).zip(&sync).zip(&resident) {
                    for delta in [(a - b).norm(), (a - d).norm(), (a - e).norm()] {
                        assert!(
                            delta.is_finite() && delta <= 1e-10,
                            "n={n} m={m} depth={depth}: {delta}"
                        );
                    }
                }
                let norm: f64 = combined.iter().map(|a| a.norm_sqr()).sum();
                assert!((norm - 1.0).abs() < 1e-10);
            }
        }
    }
}

#[test]
fn resident_pipeline_reversibility_and_rejected_inputs_leave_backend_usable() {
    let Some(mut gpu) = backend() else { return };
    let mut c = Circuit::new(8, 0);
    for q in 0..8 {
        c.h(q).unwrap();
        c.rx(0.37, q).unwrap();
    }
    c.cnot(0, 7).unwrap();
    c.cnot(6, 7).unwrap();
    c.cnot(6, 7).unwrap();
    c.cnot(0, 7).unwrap();
    for q in (0..8).rev() {
        c.rx(-0.37, q).unwrap();
        c.h(q).unwrap();
    }
    for depth in [2, 4] {
        let got = gpu
            .run_paged_batched_overlapped(&c, 3, depth)
            .unwrap()
            .amplitudes_vec();
        for (i, a) in got.iter().enumerate() {
            let delta = (a.re - if i == 0 { 1.0 } else { 0.0 }).abs() + a.im.abs();
            assert!(delta.is_finite() && delta < 1e-10);
        }
    }
    assert!(gpu.run_paged_batched_overlapped(&c, 0, 4).is_err());
    assert!(gpu.run_paged_batched_overlapped(&c, 8, 4).is_err());
    assert!(gpu.run_paged_batched_overlapped(&c, 3, u32::MAX).is_err());
    // Backend rejects k=6 only during apply, after a gather and H kernel have
    // already been enqueued. Exercise draining and context restoration on error.
    let mut late_error = Circuit::new(10, 0);
    late_error.h(0).unwrap();
    late_error
        .add_gate(GateInstance::new(
            Gate::UnitaryKq {
                k: 6,
                data: Vec::new().into(),
            },
            (0..6).collect::<Vec<_>>(),
        ))
        .unwrap();
    for m in [6, 8] {
        assert!(gpu.run_paged_batched_overlapped(&late_error, m, 4).is_err());
    }
    let huge = Circuit::new(64, 0);
    assert!(gpu.run_paged_batched_overlapped(&huge, 1, 4).is_err());
    let mut unsupported = Circuit::new(8, 1);
    unsupported.measure(0, 0).unwrap();
    assert!(gpu
        .run_paged_batched_overlapped(&unsupported, 3, 4)
        .is_err());
    let mut empty = Circuit::new(8, 0);
    empty.barrier([0]).unwrap();
    let st = gpu.run_paged_batched_overlapped(&empty, 3, 4).unwrap();
    assert!((st.norm_sqr() - 1.0).abs() < 1e-12);
}
