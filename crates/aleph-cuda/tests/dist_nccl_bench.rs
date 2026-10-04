//! P6 multi-GPU scaling (P6-01b): exchange bandwidth, strong scaling (fixed n,
//! D = 1→2→4, one rank per GPU) and weak scaling (n = m + log2 D) against the
//! single-GPU in-core / `run_paged` baseline. Every multi-GPU cell prints the
//! time model's prediction next to the measurement:
//!
//!   pred = T_onecard(R=D) / D + exchange_bytes_per_gpu / xchg_bw
//!
//! `T_onecard(R=D)` is the same plan run with all D ranks on GPU 0
//! (`LocalExchange`): each of the D GPUs does exactly one of those ranks'
//! work, so this calibrates compute on the real (specialised, fused) kernels.
//! It includes the on-card exchange copies, so it is an upper bound. A
//! pass-count model was tried first and rejected: distributed ranks issue ~2×
//! the passes of the fused single-GPU circuit, but many are cheap (local
//! swaps, diagonals), so it over-predicts by ~1.6×. `xchg_bw` is the measured
//! one-bit NcclExchange bandwidth.
//!
//! On a 1-GPU host only the D=1 rows run (the exchange line is then NCCL
//! self-send through the same card, a protocol-overhead floor).
//! Run: cargo test --release -p aleph-cuda --features nccl --test dist_nccl_bench -- --ignored --nocapture --test-threads 1
#![cfg(all(target_os = "linux", feature = "nccl"))]

mod common;

use std::time::Instant;

use aleph_backend::run;
use aleph_cuda::{
    device_count, fuse_for_gpu, CudaContext, CudaSvBackend, CudaSvBackendF32, DeviceSv,
    DistSvBackend, Exchange, LocalExchange, NcclExchange, NcclType,
};
use aleph_ir::dist::{plan, DistLayout, DistPlan, Router};
use common::dist::{brickwall, ghz, qft};

/// `None` only for a missing device; any other construction failure panics
/// so a broken host cannot print a vacuous bench.
fn or_skip<B>(r: Result<B, aleph_cuda::Error>) -> Option<B> {
    match r {
        Ok(b) => Some(b),
        Err(aleph_cuda::Error::NoDevice(_)) => None,
        Err(e) => panic!("GPU present but backend construction failed: {e}"),
    }
}

/// Sync every device's default stream (all backends here submit to it).
fn sync_all(ctxs: &[CudaContext]) {
    for c in ctxs {
        c.synchronize().unwrap();
    }
}

/// Median wall time of `reps` runs after one warm-up, each closed by a sync
/// on every device (no host readback in the timed region, as in P6-01a).
fn median_of<T, F: FnMut() -> T>(ctxs: &[CudaContext], reps: usize, mut f: F) -> f64 {
    drop(f());
    sync_all(ctxs);
    let mut ts = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        let out = f();
        sync_all(ctxs);
        ts.push(t.elapsed().as_secs_f64());
        drop(out);
        sync_all(ctxs);
    }
    ts.sort_by(f64::total_cmp);
    ts[reps / 2]
}

/// Wall time of one run, no warm-up: for the paged baseline, where one run is
/// most of an hour and the JIT/pool warm-up is noise (`median_of` would
/// double it; it did on AWS).
fn once<T, F: FnOnce() -> T>(ctxs: &[CudaContext], f: F) -> f64 {
    let t = Instant::now();
    let out = f();
    sync_all(ctxs);
    let s = t.elapsed().as_secs_f64();
    drop(out);
    s
}

/// Bandwidth (GB/s, bytes moved per GPU / time) of one single-bit exchange of
/// 2^m-amplitude slices, one rank per device (D=1: two ranks, NCCL self-send).
fn xchg_bw<B: DeviceSv>(devs: &mut Vec<B>, ctxs: &[CudaContext], m: u32, amp_bytes: usize) -> f64
where
    B::Scalar: NcclType,
{
    xchg_bw_bits(devs, ctxs, m, amp_bytes, &[0])
}

/// [`xchg_bw`] over an arbitrary set of global (rank) bits: `rank_bits = [1]`
/// pairs devices 0↔2, `[0, 1]` sends to all three peers at once.
fn xchg_bw_bits<B: DeviceSv>(
    devs: &mut Vec<B>,
    ctxs: &[CudaContext],
    m: u32,
    amp_bytes: usize,
    rank_bits: &[u32],
) -> f64
where
    B::Scalar: NcclType,
{
    let d = devs.len();
    let g = d.max(2).trailing_zeros();
    let l = DistLayout::new(m + g, g).unwrap();
    let per = (l.ranks() as usize / d).max(1);
    let mut ranks: Vec<_> = (0..l.ranks())
        .map(|r| devs[r as usize / per].alloc_rank(m, r).unwrap())
        .collect();
    let mut x = NcclExchange::new(devs)
        .unwrap()
        .route_all_through_nccl(d == 1);
    let bits: Vec<u32> = rank_bits.iter().map(|b| m + b).collect();
    let t = median_of(ctxs, 3, || {
        x.exchange(devs.as_mut_slice(), &mut ranks, l, &bits)
            .unwrap()
    });
    // Swapping k global bits moves all but 2^-k of each slice (k = 1: half),
    // and each rank receives as much as it sends.
    let k = rank_bits.len();
    let moved = (1usize << m) - (1usize << (m as usize - k));
    let bytes = moved * amp_bytes * l.ranks() as usize / d;
    bytes as f64 / t / 1e9
}

/// `T_onecard(R=D)`: plan `pl` with all its ranks on GPU 0.
fn onecard<B: DeviceSv>(b0: B, ctx: &[CudaContext], pl: &DistPlan) -> (usize, f64) {
    let mut one = DistSvBackend::new(b0, LocalExchange::new());
    let passes = one.rank_pass_count(pl).unwrap();
    (passes, median_of(ctx, 3, || one.run_plan(pl).unwrap()))
}

struct Prec<B> {
    name: &'static str,
    amp_bytes: usize,
    make: fn(usize) -> Option<B>,
}

fn strong_and_weak<B>(p: &Prec<B>, n_strong: u32, m_weak: u32, ctxs: &[CudaContext])
where
    B: DeviceSv,
    B::Scalar: NcclType,
{
    let n_dev = ctxs.len();
    let ds: Vec<usize> = [1usize, 2, 4].into_iter().filter(|&d| d <= n_dev).collect();
    let Some(mut single) = (p.make)(0) else {
        return;
    };

    // Exchange bandwidth at the strong-scaling slice size of the widest D.
    let d_max = *ds.last().unwrap_or(&1);
    let m_x = n_strong - d_max.trailing_zeros().max(1);
    let mut devs: Vec<B> = (0..d_max)
        .map(|i| (p.make)(i).expect("GPU missing"))
        .collect();
    let bw = xchg_bw(&mut devs, &ctxs[..d_max], m_x, p.amp_bytes);
    drop(devs);
    println!("xchg,{},D={d_max},m={m_x},GBps={bw:.2}", p.name);

    println!("strong,prec,circuit,n,D,exchanges,MB_moved_per_gpu,passes,onecard_s,pred_s,meas_s,speedup_vs_1gpu");
    for (name, c) in [
        ("QFT", qft(n_strong)),
        ("random_d10", brickwall(n_strong, 10)),
        ("GHZ", ghz(n_strong)),
    ] {
        let fused = fuse_for_gpu(&c);
        let t1 = median_of(&ctxs[..1], 3, || run(&mut single, &fused).unwrap());
        println!(
            "strong,{},{name},{n_strong},1,0,0,-,-,-,{t1:.4},1.00",
            p.name
        );
        for &d in ds.iter().filter(|&&d| d > 1) {
            let g = d.trailing_zeros();
            let pl = plan(&c, DistLayout::new(n_strong, g).unwrap(), Router::Lookahead).unwrap();
            let Some(b0) = (p.make)(0) else {
                return;
            };
            let (passes, t_one) = onecard(b0, &ctxs[..1], &pl);
            let devs: Vec<B> = (0..d).map(|i| (p.make)(i).expect("GPU missing")).collect();
            let x = NcclExchange::new(&devs).unwrap();
            let mut db = DistSvBackend::multi(devs, x).unwrap();
            let mb = pl.stats.amps_moved_per_rank as f64 * p.amp_bytes as f64 / 1e6;
            let pred = t_one / d as f64 + mb * 1e6 / (bw * 1e9);
            let t = median_of(&ctxs[..d], 3, || db.run_plan(&pl).unwrap());
            println!(
                "strong,{},{name},{n_strong},{d},{},{mb:.0},{passes},{t_one:.4},{pred:.4},{t:.4},{:.2}",
                p.name,
                pl.stats.exchanges,
                t1 / t
            );
        }
    }

    println!("weak,prec,n,D,m,meas_s,baseline,baseline_s");
    let c1 = brickwall(m_weak, 10);
    let t1 = median_of(&ctxs[..1], 3, || {
        run(&mut single, &fuse_for_gpu(&c1)).unwrap()
    });
    println!(
        "weak,{},{m_weak},1,{m_weak},{t1:.4},in-core,{t1:.4}",
        p.name
    );
    drop(single);
    for &d in ds.iter().filter(|&&d| d > 1) {
        let g = d.trailing_zeros();
        let n = m_weak + g;
        let c = brickwall(n, 10);
        let devs: Vec<B> = (0..d).map(|i| (p.make)(i).expect("GPU missing")).collect();
        let x = NcclExchange::new(&devs).unwrap();
        let mut db = DistSvBackend::multi(devs, x).unwrap();
        let t = median_of(&ctxs[..d], 1, || db.run(&c, g, Router::Lookahead).unwrap());
        // Baseline (single-GPU run_paged at the same n) is run separately by
        // `weak_paged_baseline` — it needs the whole 2^n state in host RAM.
        println!(
            "weak,{},{n},{d},{m_weak},{t:.4},see weak_paged_baseline,-",
            p.name
        );
    }
}

#[test]
#[ignore]
fn dist_nccl_scaling() {
    let n_dev = device_count().unwrap_or(0).min(4);
    if n_dev == 0 {
        return;
    }
    let ctxs: Vec<CudaContext> = (0..n_dev).map(|i| CudaContext::new(i).unwrap()).collect();
    println!("gpus={n_dev}");
    let p64 = Prec {
        name: "FP64",
        amp_bytes: 16,
        make: |i| or_skip(CudaSvBackend::on_device(i)),
    };
    let p32 = Prec {
        name: "FP32",
        amp_bytes: 8,
        make: |i| or_skip(CudaSvBackendF32::on_device(i)),
    };
    // Weak: FP64 m=30 (16 GiB per GPU), FP32 m=31 (16 GiB per GPU).
    strong_and_weak(&p64, 28, 30, &ctxs);
    strong_and_weak(&p32, 28, 31, &ctxs);
}

/// Single-GPU out-of-core baseline for the weak-scaling n (2^n state in pinned
/// host RAM): n = 30 + log2 D FP64, 31 + log2 D FP32, for each D present.
///
/// The tile is one below the in-core ceiling: a brickwall 2q gate can touch two
/// high qubits, and the co-resident group `2^(tile+2)` must still fit the
/// device (`tile + g_max <= MAX_CUDA_QUBITS`; 16 GiB either precision).
#[test]
#[ignore]
fn weak_paged_baseline() {
    let n_dev = device_count().unwrap_or(0).min(4);
    if n_dev < 2 {
        eprintln!("weak_paged_baseline: needs the multi-GPU n; skip on {n_dev} GPU(s)");
        return;
    }
    let ctx = [CudaContext::new(0).unwrap()];
    println!("weak_paged,prec,n,tile_m,meas_s");
    for d in [2usize, 4].into_iter().filter(|&d| d <= n_dev) {
        let g = d.trailing_zeros();
        let n = 30 + g;
        let c = brickwall(n, 10);
        let mut be = CudaSvBackend::with_seed(0).unwrap();
        let t = once(&ctx, || be.run_paged(&c, 28).unwrap());
        println!("weak_paged,FP64,{n},28,{t:.3}");
        let n = 31 + g;
        let c = brickwall(n, 10);
        let mut be = CudaSvBackendF32::with_seed(0).unwrap();
        let t = once(&ctx, || be.run_paged(&c, 29).unwrap());
        println!("weak_paged,FP32,{n},29,{t:.3}");
    }
}

/// Exchange bandwidth alone, D = 2 and 4 (FP64, m = 26): a quick probe for
/// trying NCCL transport settings (`NCCL_*` env) without the full scaling run.
#[test]
#[ignore]
fn xchg_probe() {
    let n_dev = device_count().unwrap_or(0).min(4);
    let ctxs: Vec<CudaContext> = (0..n_dev).map(|i| CudaContext::new(i).unwrap()).collect();
    for d in [2usize, 4].into_iter().filter(|&d| d <= n_dev) {
        let Some(mut devs) = (0..d)
            .map(|i| or_skip(CudaSvBackend::on_device(i)))
            .collect::<Option<Vec<_>>>()
        else {
            return;
        };
        let sets: &[&[u32]] = if d == 4 {
            &[&[0], &[1], &[0, 1]]
        } else {
            &[&[0]]
        };
        for bits in sets {
            let bw = xchg_bw_bits(&mut devs, &ctxs[..d], 26, 16, bits);
            println!("xchg_probe,FP64,D={d},m=26,rank_bits={bits:?},GBps={bw:.2}");
        }
    }
}

/// Model inputs for D = 2, 4 (one rank per GPU), all measurable on ONE GPU:
/// passes per rank, exchanges, amplitudes moved per rank, and
/// `T_onecard(R=D)`. With the D=1 time and an assumed link bandwidth this
/// gives the prediction in docs/perf/p6-multi-gpu.md before any multi-GPU run.
#[test]
#[ignore]
fn model_inputs() {
    if device_count().unwrap_or(0) == 0 {
        return;
    }
    let ctx = [CudaContext::new(0).unwrap()];
    println!("model,prec,circuit,n,D,passes_per_rank,exchanges,amps_moved_per_rank,onecard_s");
    model_rows("FP64", &ctx, || or_skip(CudaSvBackend::on_device(0)));
    model_rows("FP32", &ctx, || or_skip(CudaSvBackendF32::on_device(0)));
}

fn model_rows<B: DeviceSv>(prec: &str, ctx: &[CudaContext], make: impl Fn() -> Option<B>) {
    for (name, c) in [
        ("QFT", qft(28)),
        ("random_d10", brickwall(28, 10)),
        ("GHZ", ghz(28)),
    ] {
        for d in [1usize, 2, 4] {
            let g = d.trailing_zeros();
            let pl = plan(&c, DistLayout::new(28, g).unwrap(), Router::Lookahead).unwrap();
            let Some(b0) = make() else {
                return;
            };
            let (passes, t) = onecard(b0, ctx, &pl);
            println!(
                "model,{prec},{name},28,{d},{passes},{},{},{t:.4}",
                pl.stats.exchanges, pl.stats.amps_moved_per_rank
            );
        }
    }
}
