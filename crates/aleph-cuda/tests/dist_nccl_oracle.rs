//! P6-01b NcclExchange oracle. On a 1-GPU host this runs D=1 with every pair
//! forced through NCCL self send/recv; on a multi-GPU host (AWS 4×L4) it also
//! runs one rank block per real GPU at D=2 and D=4.
//! Run: cargo test -p aleph-cuda --features nccl --test dist_nccl_oracle -- --nocapture
#![cfg(all(target_os = "linux", feature = "nccl"))]

mod common;

use aleph_cuda::{device_count, CudaSvBackend, CudaSvBackendF32, DistSvBackend, NcclExchange};
use aleph_ir::dist::Router;
use common::dist::*;

/// `Some(backend)`, `None` only when the device does not exist; any other
/// construction failure (NVRTC/driver missing, ...) panics, so a broken GPU
/// host cannot report a vacuous green.
fn or_skip<B>(r: Result<B, aleph_cuda::Error>) -> Option<B> {
    match r {
        Ok(b) => Some(b),
        Err(aleph_cuda::Error::NoDevice(_)) => None,
        Err(e) => panic!("GPU present but backend construction failed: {e}"),
    }
}

fn devs64(d: usize) -> Option<Vec<CudaSvBackend>> {
    (0..d)
        .map(|i| or_skip(CudaSvBackend::on_device(i)))
        .collect()
}

fn devs32(d: usize) -> Option<Vec<CudaSvBackendF32>> {
    (0..d)
        .map(|i| or_skip(CudaSvBackendF32::on_device(i)))
        .collect()
}

/// GPUs available for the real multi-GPU tests. `ALEPH_REQUIRE_GPUS=N` (set by
/// the AWS runbook) turns "fewer than N GPUs" into a failure instead of a skip.
fn gpus_for_multi() -> usize {
    let n = device_count().unwrap_or(0);
    if let Some(req) = std::env::var("ALEPH_REQUIRE_GPUS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        assert!(
            n >= req,
            "ALEPH_REQUIRE_GPUS={req} but only {n} GPU(s) visible"
        );
    }
    n
}

#[test]
fn nccl_new_without_lib_is_err_not_panic() {
    // Must not unwind whatever the host has (Err when libnccl is absent).
    let Some(devs) = devs64(1) else { return };
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        NcclExchange::new(&devs).is_ok()
    }));
    assert!(r.is_ok(), "NcclExchange::new panicked");
}

#[test]
fn nccl_rejects_duplicate_ordinals() {
    let Some(a) = devs64(1) else { return };
    let Some(b) = devs64(1) else { return };
    let devs: Vec<_> = a.into_iter().chain(b).collect(); // two backends, both GPU 0
    assert!(NcclExchange::new(&devs).is_err());
}

#[test]
fn nccl_self_routed_matches_oracle_f64() {
    let Some(devs) = devs64(1) else { return };
    // A GPU host that enables `nccl` must have the library: fail loudly
    // rather than skip (a silent skip hid a missing libnccl.so once).
    let x = NcclExchange::new(&devs).expect("libnccl.so must load on a GPU host");
    let x = x.with_scratch_amps(8).route_all_through_nccl(true);
    let mut db = DistSvBackend::multi(devs, x).unwrap();
    for (name, c) in cases() {
        let want = reference(&c);
        for g in 1..=3u32 {
            for router in [Router::Naive, Router::Lookahead] {
                let st = db.run(&c, g, router).unwrap();
                let got = db.amplitudes(&st).unwrap();
                for (i, (x, y)) in got.iter().zip(&want).enumerate() {
                    assert!((x - y).norm() < 1e-10, "{name} g={g} {router:?} amp {i}");
                }
            }
        }
    }
}

#[test]
fn nccl_self_routed_matches_oracle_f32() {
    let Some(devs) = devs32(1) else {
        return;
    };
    let x = NcclExchange::new(&devs).expect("libnccl.so must load on a GPU host");
    let x = x.with_scratch_amps(8).route_all_through_nccl(true);
    let mut db = DistSvBackend::multi(devs, x).unwrap();
    for (name, c) in cases() {
        let want = reference(&c);
        for g in 1..=3u32 {
            let st = db.run(&c, g, Router::Lookahead).unwrap();
            let got = db.amplitudes(&st).unwrap();
            for (i, (x, y)) in got.iter().zip(&want).enumerate() {
                assert!((x - y).norm() < 1e-5, "{name} g={g} amp {i}");
            }
        }
    }
}

#[test]
fn nccl_real_multi_gpu_matches_oracle() {
    let n_dev = gpus_for_multi();
    for d in [2usize, 4] {
        if n_dev < d {
            eprintln!("only {n_dev} GPU(s): skip D={d}");
            continue;
        }
        let devs = devs64(d).expect("device_count() >= d but a device is missing");
        let x = NcclExchange::new(&devs).unwrap().with_scratch_amps(8);
        let mut db = DistSvBackend::multi(devs, x).unwrap();
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
                    for q in [0, c.num_qubits() - 1] {
                        let p: f64 = want
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| (i >> q) & 1 == 1)
                            .map(|(_, a)| a.norm_sqr())
                            .sum();
                        let got = db.prob_one(&st, q).unwrap();
                        assert!((got - p).abs() < 1e-10, "{name} D={d} g={g} q={q}");
                    }
                }
            }
        }
    }
}

#[test]
fn nccl_real_multi_gpu_matches_oracle_f32_default_scratch() {
    let n_dev = gpus_for_multi();
    for d in [2usize, 4] {
        if n_dev < d {
            eprintln!("only {n_dev} GPU(s): skip D={d}");
            continue;
        }
        let devs = devs32(d).expect("device_count() >= d but a device is missing");
        let x = NcclExchange::new(&devs).unwrap(); // default scratch
        let mut db = DistSvBackend::multi(devs, x).unwrap();
        for (name, c) in cases() {
            let want = reference(&c);
            for g in d.trailing_zeros()..=3u32 {
                let st = db.run(&c, g, Router::Lookahead).unwrap();
                let got = db.amplitudes(&st).unwrap();
                for (i, (x, y)) in got.iter().zip(&want).enumerate() {
                    assert!((x - y).norm() < 1e-5, "{name} D={d} g={g} amp {i}");
                }
            }
        }
    }
}

/// The exchange alone, on known per-rank data, vs the CPU reference
/// (`exchange_cpu`): isolates the NCCL piece protocol from gate execution.
#[test]
fn nccl_exchange_matches_cpu_reference() {
    use aleph_core::Complex;
    use aleph_cuda::{DeviceSv, Exchange};
    use aleph_ir::dist::DistLayout;
    let Some(mut devs) = devs64(1) else { return };
    for (n, g) in [(9u32, 1u32), (9, 2), (10, 3)] {
        let l = DistLayout::new(n, g).unwrap();
        let m = l.m();
        let size = 1usize << m;
        let mut orders: Vec<Vec<u32>> = vec![vec![m], vec![n - 1]];
        if g >= 2 {
            orders.push(vec![m, m + 1]);
            orders.push(vec![m + 1, m]);
        }
        for bits in orders {
            for scratch in [2usize, 8, 1 << 20] {
                let host: Vec<Vec<Complex<f64>>> = (0..l.ranks() as usize)
                    .map(|r| {
                        (0..size)
                            .map(|i| Complex::new((r * size + i) as f64, 0.5 * i as f64 - r as f64))
                            .collect()
                    })
                    .collect();
                let mut want = host.clone();
                aleph_sv::dist_ref::exchange_cpu(&mut want, l, &bits);
                let mut ranks: Vec<_> =
                    host.iter().map(|v| devs[0].upload(m, v).unwrap()).collect();
                let mut x = NcclExchange::new(&devs)
                    .unwrap()
                    .with_scratch_amps(scratch)
                    .route_all_through_nccl(true);
                x.exchange(&mut devs, &mut ranks, l, &bits).unwrap();
                for (r, st) in ranks.iter().enumerate() {
                    let got = devs[0].download(st).unwrap();
                    let bad = got.iter().zip(&want[r]).position(|(a, b)| a != b);
                    assert!(
                        bad.is_none(),
                        "n={n} g={g} bits={bits:?} scratch={scratch} rank {r} first bad {bad:?}: got {:?} want {:?}",
                        bad.map(|i| got[i]),
                        bad.map(|i| want[r][i])
                    );
                }
            }
        }
    }
}
