//! P6-01a: distributed SV on one GPU (DistSvBackend + LocalExchange) vs the
//! CPU oracle. Skips without a CUDA device.
#![cfg(all(target_os = "linux", feature = "cuda"))]

use aleph_core::Complex;
use aleph_cuda::{CudaSvBackend, CudaSvBackendF32, DeviceSv};

mod common;

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
                x.exchange(std::slice::from_mut(&mut be), &mut ranks, l, &bits)
                    .unwrap();
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
use aleph_cuda::DistSvBackend;
use aleph_ir::dist::Router;
use common::dist::*;

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
    assert!(x
        .exchange(std::slice::from_mut(&mut be), &mut ranks, l, &[m, m])
        .is_err());
    assert!(x
        .exchange(
            std::slice::from_mut(&mut be),
            &mut ranks,
            l,
            &[m + 1, m, m + 1]
        )
        .is_err());
}

#[test]
fn backends_open_on_explicit_ordinal() {
    let Ok(n) = aleph_cuda::device_count() else {
        return;
    };
    if n == 0 {
        return;
    }
    assert!(CudaSvBackend::on_device(0).is_ok());
    assert!(CudaSvBackendF32::on_device(0).is_ok());
    // One past the last device: an error, never a panic.
    assert!(CudaSvBackend::on_device(n).is_err());
    assert!(CudaSvBackendF32::on_device(n).is_err());
}

#[test]
fn device_sv_branch_norms_and_views() {
    let Some(mut be) = gpu64() else { return };
    let amps = ramp(16, 0.25);
    let st = be.upload(4, &amps).unwrap();
    let tot: f64 = amps.iter().map(|a| a.norm_sqr()).sum();
    let p1: f64 = amps
        .iter()
        .enumerate()
        .filter(|(i, _)| i & 4 != 0)
        .map(|(_, a)| a.norm_sqr())
        .sum();
    let (t, b) = be.branch_norms(&st, 4).unwrap();
    assert!((t - tot).abs() < 1e-9 && (b - p1).abs() < 1e-9, "{t} {b}");
    assert_eq!(be.ordinal(), 0);
    assert_eq!(CudaSvBackend::amps_view(&st, 8, 8).unwrap().len(), 16);
    assert!(CudaSvBackend::amps_view(&st, 9, 8).is_err());
    assert!(CudaSvBackend::amps_view(&st, usize::MAX / 2, 8).is_err());
    let Some(mut be32) = gpu32() else { return };
    let s32 = be32.upload(4, &amps).unwrap();
    let (t32, b32) = be32.branch_norms(&s32, 4).unwrap();
    assert!((t32 - tot).abs() < 1e-3 && (b32 - p1).abs() < 1e-3);
}

/// D backends that all live on GPU 0: exercises rank placement and per-device
/// scratch on one card (LocalExchange pairs by *ordinal*, so cross-"device"
/// pairs on the same GPU are plain D2D copies).
fn same_gpu_devs(d: usize) -> Option<Vec<CudaSvBackend>> {
    (0..d).map(|_| CudaSvBackend::on_device(0).ok()).collect()
}

#[test]
fn multi_backend_same_gpu_matches_oracle() {
    for d in [2usize, 4] {
        let Some(devs) = same_gpu_devs(d) else { return };
        let mut db = DistSvBackend::multi(devs, LocalExchange::with_scratch_amps(8)).unwrap();
        assert_eq!(db.devices(), d);
        for (name, c) in cases() {
            let want = reference(&c);
            for g in d.trailing_zeros()..=3u32 {
                for router in [Router::Naive, Router::Lookahead] {
                    let st = db.run(&c, g, router).unwrap();
                    let got = db.amplitudes(&st).unwrap();
                    for (i, (x, y)) in got.iter().zip(&want).enumerate() {
                        assert!(
                            (x - y).norm() < 1e-10,
                            "{name} D={d} g={g} {router:?} amp {i}"
                        );
                    }
                    assert!((db.norm_sqr(&st).unwrap() - 1.0).abs() < 1e-10);
                }
            }
        }
    }
}

#[test]
fn multi_rejects_bad_device_counts() {
    let Some(devs) = same_gpu_devs(3) else { return };
    assert!(DistSvBackend::multi(devs, LocalExchange::new()).is_err()); // D = 3
    assert!(DistSvBackend::<CudaSvBackend, _>::multi(vec![], LocalExchange::new()).is_err());
    let Some(devs) = same_gpu_devs(4) else { return };
    let mut db = DistSvBackend::multi(devs, LocalExchange::new()).unwrap();
    assert!(db.run(&ghz(8), 1, Router::Lookahead).is_err()); // R = 2 < D = 4
}

#[test]
fn prob_one_matches_reference_for_every_qubit() {
    let Some(be) = gpu64() else { return };
    let mut db = DistSvBackend::new(be, LocalExchange::with_scratch_amps(8));
    for (name, c) in [("qft10", qft(10)), ("brick10", brickwall(10, 6))] {
        let want = reference(&c);
        for g in [0u32, 2] {
            let st = db.run(&c, g, Router::Lookahead).unwrap();
            for q in 0..c.num_qubits() {
                let p: f64 = want
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| (i >> q) & 1 == 1)
                    .map(|(_, a)| a.norm_sqr())
                    .sum();
                let got = db.prob_one(&st, q).unwrap();
                assert!((got - p).abs() < 1e-10, "{name} g={g} q={q}: {got} vs {p}");
            }
            assert!(db.prob_one(&st, c.num_qubits()).is_err());
        }
    }
}

#[test]
fn run_plan_rejects_bad_final_map() {
    let Some(be) = gpu64() else { return };
    let mut db = DistSvBackend::new(be, LocalExchange::new());
    let c = ghz(6);
    let l = DistLayout::new(6, 1).unwrap();
    let mut p = aleph_ir::dist::plan(&c, l, Router::Naive).unwrap();
    p.final_map[0] = p.final_map[1]; // not a permutation
    assert!(db.run_plan(&p).is_err());
}

#[test]
fn rank_pass_count_counts_rank_zero_instructions() {
    let Some(be) = gpu64() else { return };
    let db = DistSvBackend::new(be, LocalExchange::new()).with_fusion(false);
    let c = ghz(6); // H + 5 CNOT, all local at g = 0
    let p = aleph_ir::dist::plan(&c, DistLayout::new(6, 0).unwrap(), Router::Naive).unwrap();
    assert_eq!(db.rank_pass_count(&p).unwrap(), 6);
}

/// #529: a diagonal that keeps two local qubits must stay diagonal after
/// `specialize`, so the fused rank program is one phase-polynomial pass.
#[test]
fn specialized_2local_diagonals_fuse_to_one_pass() {
    use aleph_core::{Gate, GateInstance};
    use aleph_cuda::{DistSvBackend, LocalExchange};
    use aleph_ir::dist::{DistLayout, Router};
    let Some(be) = gpu64() else { return };
    let db = DistSvBackend::new(be, LocalExchange::new());
    let mut c = aleph_ir::Circuit::new(6, 0);
    for q in 0..4 {
        c.add_gate(GateInstance::new(Gate::Ccz, vec![q, q + 1, 5]))
            .unwrap();
        c.add_gate(GateInstance::new(Gate::Cz, vec![q, q + 1]))
            .unwrap();
    }
    let p = aleph_ir::dist::plan(&c, DistLayout::new(6, 1).unwrap(), Router::Naive).unwrap();
    assert_eq!(db.rank_pass_count(&p).unwrap(), 1);
}

/// `Exchange` is public: a device count that is not a power of two (or
/// exceeds the rank count) must be an error, never an index panic.
#[test]
fn exchange_rejects_bad_device_counts() {
    let Some(mut devs) = same_gpu_devs(3) else {
        return;
    };
    let l = DistLayout::new(6, 2).unwrap(); // R = 4, m = 4
    let mut ranks: Vec<_> = (0..4).map(|r| devs[0].alloc_rank(4, r).unwrap()).collect();
    let mut x = LocalExchange::<CudaSvBackend>::new();
    assert!(x.exchange(&mut devs, &mut ranks, l, &[4]).is_err()); // D = 3
    let Some(mut devs8) = same_gpu_devs(8) else {
        return;
    };
    assert!(x.exchange(&mut devs8, &mut ranks, l, &[4]).is_err()); // D = 8 > R = 4
}
