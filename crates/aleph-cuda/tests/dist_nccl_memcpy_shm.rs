//! NcclExchange under NCCL's CUDA-memcpy SHM transport (`NCCL_SHM_USE_CUDA_MEMCPY=1`),
//! the fast path on P2P-less PCIe boxes (AWS g6.12xlarge: 7.2 vs 2.9 GB/s).
//!
//! In that mode the NCCL proxy thread issues CUDA copies. A host thread that
//! blocks inside a CUDA enqueue while an NCCL kernel is pending (the push
//! buffer fills when many rounds are enqueued unsynced) holds a driver lock the
//! proxy needs: deadlock. Seen on 4× L4 at n=31; `exchange` now bounds the
//! rounds in flight and drains before returning. Needs ≥2 GPUs; own binary
//! because the env var must be set before the first NCCL init in the process.
//! Run: cargo test -p aleph-cuda --features nccl --test dist_nccl_memcpy_shm -- --nocapture
#![cfg(all(target_os = "linux", feature = "nccl"))]

mod common;

use std::sync::mpsc;
use std::time::Duration;

use aleph_cuda::{device_count, CudaSvBackend, DistSvBackend, LocalExchange, NcclExchange};
use aleph_ir::dist::Router;
use common::dist::*;

#[test]
fn memcpy_shm_many_rounds_completes_and_matches_oracle() {
    let n_dev = device_count().unwrap_or(0);
    let required = std::env::var("ALEPH_REQUIRE_GPUS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    assert!(
        n_dev >= required,
        "ALEPH_REQUIRE_GPUS={required} but only {n_dev} GPU(s) visible"
    );
    if n_dev < 2 {
        eprintln!("only {n_dev} GPU(s): skip");
        return;
    }
    // Single test in this binary, before any NCCL init.
    std::env::set_var("NCCL_SHM_USE_CUDA_MEMCPY", "1");

    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        // 1 GiB FP64 slices, unfused: each exchange round is tens of ms of
        // NCCL traffic while the host races ahead enqueueing ~1 launch per
        // gate, which is what filled the push buffer at n=31.
        let (n, depth) = (27, 12);
        let c = brickwall(n, depth);
        // Reference: same plan, both ranks on GPU 0 (LocalExchange, oracle-tested).
        let one = CudaSvBackend::on_device(0).expect("GPU present");
        let mut one = DistSvBackend::new(one, LocalExchange::new()).with_fusion(false);
        let want = one.run(&c, 1, Router::Lookahead).unwrap();
        let want = one.amplitudes(&want).unwrap();
        drop(one);
        let devs: Vec<_> = (0..2)
            .map(|i| CudaSvBackend::on_device(i).expect("GPU present"))
            .collect();
        let x = NcclExchange::new(&devs).unwrap();
        let mut db = DistSvBackend::multi(devs, x).unwrap().with_fusion(false);
        let st = db.run(&c, 1, Router::Lookahead).unwrap();
        let got = db.amplitudes(&st).unwrap();
        let worst = got
            .iter()
            .zip(&want)
            .map(|(x, y)| (x - y).norm())
            .fold(0.0f64, f64::max);
        tx.send(worst).unwrap();
    });
    // A deadlock never returns, and a panic would not end the process either:
    // the harness's exit hangs in CUDA/NCCL teardown behind the stuck
    // threads. Abort instead, so CI fails rather than hangs.
    match rx.recv_timeout(Duration::from_secs(300)) {
        Ok(worst) => assert!(worst.is_finite() && worst < 1e-10, "max |Δamp| = {worst}"),
        Err(_) => {
            eprintln!("exchange did not finish in 300 s: NCCL/host deadlock");
            std::process::abort();
        }
    }
}
