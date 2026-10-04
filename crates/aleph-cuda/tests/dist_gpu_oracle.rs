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

use aleph_cuda::{Exchange, LocalExchange};
use aleph_ir::dist::DistLayout;

fn rank_data(l: DistLayout) -> Vec<Vec<Complex<f64>>> {
    let size = 1usize << l.m();
    (0..l.ranks() as usize)
        .map(|r| {
            (0..size)
                .map(|i| Complex::new((r * size + i) as f64, 0.5 * i as f64 - r as f64))
                .collect()
        })
        .collect()
}

#[test]
fn local_exchange_matches_cpu_reference() {
    let Some(mut be) = gpu64() else { return };
    for (n, g) in [(9u32, 2u32), (10, 3)] {
        let l = DistLayout::new(n, g).unwrap();
        let m = l.m();
        let orders: Vec<Vec<u32>> = vec![
            vec![m],
            vec![m + 1],
            vec![m, m + 1],
            vec![m + 1, m],
            if g == 3 {
                vec![m + 2, m, m + 1]
            } else {
                vec![m + 1, m]
            },
        ];
        for bits in orders {
            for scratch in [4usize, 1 << 20] {
                let host = rank_data(l);
                let mut want = host.clone();
                aleph_sv::dist_ref::exchange_cpu(&mut want, l, &bits);
                let mut ranks: Vec<_> = host.iter().map(|v| be.upload(m, v).unwrap()).collect();
                let mut x = LocalExchange::<CudaSvBackend>::with_scratch_amps(scratch);
                x.exchange(&mut be, &mut ranks, l, &bits).unwrap();
                for (r, st) in ranks.iter().enumerate() {
                    let got = be.download(st).unwrap();
                    assert_eq!(
                        got, want[r],
                        "n={n} g={g} bits={bits:?} scratch={scratch} rank {r}"
                    );
                }
            }
        }
    }
}

use aleph_backend::run;
use aleph_core::{Gate, GateInstance, Param};
use aleph_cuda::DistSvBackend;
use aleph_ir::dist::Router;
use aleph_ir::{Circuit, Instruction};
use aleph_sv::NaiveSvBackend;

fn reference(c: &Circuit) -> Vec<Complex<f64>> {
    let mut b = NaiveSvBackend::with_seed(0);
    run(&mut b, c).unwrap().amplitudes().to_vec()
}

fn ghz(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    c.h(0).unwrap();
    for q in 0..n - 1 {
        c.cnot(q, q + 1).unwrap();
    }
    c
}

fn qft(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for j in (0..n).rev() {
        c.h(j).unwrap();
        for k in (0..j).rev() {
            let th = std::f64::consts::PI / f64::from(1u32 << (j - k));
            c.add_gate(GateInstance::controlled(
                Gate::Phase(Param::Concrete(th)),
                vec![j],
                vec![k],
            ))
            .unwrap();
        }
    }
    for q in 0..n / 2 {
        c.swap(q, n - 1 - q).unwrap();
    }
    c
}

fn brickwall(n: u32, depth: usize) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for d in 0..depth {
        for q in 0..n {
            c.rx(0.3 + 0.17 * f64::from(q), q).unwrap();
            c.rz(0.7 * d as f64 + 0.05 * f64::from(q), q).unwrap();
        }
        let mut q = (d % 2) as u32;
        while q + 1 < n {
            c.cnot(q, q + 1).unwrap();
            q += 2;
        }
    }
    c
}

fn grover8() -> Circuit {
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../scripts/qiskit-baseline/circuits/grover_n8_iters13.qasm"
    ))
    .unwrap();
    let parsed = aleph_parser::parse(&src).unwrap();
    let mut c = Circuit::new(parsed.num_qubits(), 0);
    for i in parsed.instructions() {
        if let Instruction::Gate(g) = i {
            c.add_gate(g.clone()).unwrap();
        }
    }
    c
}

fn all_diag_on_globals(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for q in 0..n {
        c.h(q).unwrap();
    }
    let top = n - 1;
    c.rz(0.9, top).unwrap();
    c.add_gate(GateInstance::new(Gate::Cz, vec![top - 1, top]))
        .unwrap();
    c.add_gate(GateInstance::new(
        Gate::CRz(Param::Concrete(1.1)),
        vec![0, top],
    ))
    .unwrap();
    c.add_gate(GateInstance::new(Gate::Ccz, vec![top - 1, 1, top]))
        .unwrap();
    c.add_gate(GateInstance::controlled(Gate::T, vec![2u32], vec![top]))
        .unwrap();
    c.add_gate(GateInstance::new(Gate::Toffoli, vec![top, top - 1, 0]))
        .unwrap();
    // Externally controlled diagonals touching globals: at g >= 1 the first
    // specialises to a `Unitary1qDiag` on q1 *with* local control q0; at g >= 2
    // the second leaves only a scalar phase gated on local control q0.
    c.add_gate(GateInstance::controlled(Gate::Cz, vec![1, top], vec![0]))
        .unwrap();
    c.add_gate(GateInstance::controlled(
        Gate::Cz,
        vec![top - 1, top],
        vec![0],
    ))
    .unwrap();
    c
}

fn cases() -> Vec<(&'static str, Circuit)> {
    vec![
        ("ghz10", ghz(10)),
        ("qft10", qft(10)),
        ("brick10", brickwall(10, 6)),
        ("grover8", grover8()),
        ("diag10", all_diag_on_globals(10)),
    ]
}

#[test]
fn dist_f64_matches_oracle_all_layouts_routers_fusion() {
    let Some(be) = gpu64() else { return };
    let mut d = DistSvBackend::new(be, LocalExchange::with_scratch_amps(8));
    for (name, c) in cases() {
        let want = reference(&c);
        for g in 0..=3u32 {
            for router in [Router::Naive, Router::Lookahead] {
                for fuse in [false, true] {
                    d = d.with_fusion(fuse);
                    let st = d.run(&c, g, router).unwrap();
                    let got = d.amplitudes(&st).unwrap();
                    for (i, (x, y)) in got.iter().zip(&want).enumerate() {
                        assert!(
                            (x - y).norm() < 1e-10,
                            "{name} g={g} {router:?} fuse={fuse} amp {i}: {x} vs {y}"
                        );
                    }
                    assert!((d.norm_sqr(&st).unwrap() - 1.0).abs() < 1e-10);
                }
            }
        }
    }
}

#[test]
fn dist_f32_matches_oracle() {
    let Some(be) = gpu32() else { return };
    let mut d = DistSvBackend::new(be, LocalExchange::with_scratch_amps(8));
    for (name, c) in cases() {
        let want = reference(&c);
        for g in [1u32, 2, 3] {
            let st = d.run(&c, g, Router::Lookahead).unwrap();
            let got = d.amplitudes(&st).unwrap();
            for (i, (x, y)) in got.iter().zip(&want).enumerate() {
                assert!((x - y).norm() < 1e-5, "{name} g={g} amp {i}: {x} vs {y}");
            }
        }
    }
}

#[test]
fn dist_matches_single_gpu_at_n20() {
    // Larger slices with the default scratch: equal to single-GPU CudaSvBackend.
    let Some(be) = gpu64() else { return };
    let Some(mut single) = gpu64() else { return };
    let c = brickwall(20, 8);
    let want = run(&mut single, &c).unwrap().amplitudes_vec();
    let mut d = DistSvBackend::new(be, LocalExchange::new());
    for g in [1u32, 2, 3] {
        let st = d.run(&c, g, Router::Lookahead).unwrap();
        let got = d.amplitudes(&st).unwrap();
        let worst = got
            .iter()
            .zip(&want)
            .map(|(x, y)| (x - y).norm())
            .fold(0.0, f64::max);
        assert!(worst < 1e-10, "g={g} worst {worst}");
    }
}

#[test]
fn dist_rejects_rank_slice_over_qubit_cap() {
    // m = 5 against a 4-qubit cap: must be TooManyQubits, not an allocation.
    let Some(be) = gpu64() else { return };
    let mut d = DistSvBackend::new(be.with_qubit_cap(4), LocalExchange::new());
    assert!(d.run(&ghz(6), 1, Router::Lookahead).is_err());
    // m = 64 would overflow `1 << m` (debug panic / release tiny buffer).
    let Some(be) = gpu64() else { return };
    let mut d = DistSvBackend::new(be, LocalExchange::new());
    assert!(d.run(&ghz(64), 0, Router::Naive).is_err());
}

#[test]
fn device_sv_rejects_oversized_slices() {
    let Some(mut be) = gpu64() else { return };
    assert!(be.alloc_rank(64, 1).is_err());
    assert!(be.alloc_rank(63, 0).is_err());
    let Some(mut be32) = gpu32() else { return };
    assert!(be32.alloc_rank(64, 1).is_err());
    assert!(be32.upload(64, &[]).is_err());
}

#[test]
fn local_exchange_rejects_duplicate_bits() {
    let Some(mut be) = gpu64() else { return };
    let l = DistLayout::new(8, 2).unwrap();
    let m = l.m();
    let mut ranks: Vec<_> = rank_data(l)
        .iter()
        .map(|v| be.upload(m, v).unwrap())
        .collect();
    let mut x = LocalExchange::<CudaSvBackend>::new();
    assert!(x.exchange(&mut be, &mut ranks, l, &[m, m]).is_err());
    assert!(x
        .exchange(&mut be, &mut ranks, l, &[m + 1, m, m + 1])
        .is_err());
}
