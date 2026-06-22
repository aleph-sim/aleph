//! P5-06 benchmark: the diagonal-gate niche where the custom `apply_diag`
//! kernels beat cuStateVec's generic `custatevecApplyMatrix` (and our own dense
//! `apply_kq`). All `#[ignore]` — they need a real GPU and run for seconds.
//!
//! Run (RTX 4000 Ada box):
//! ```bash
//! ALEPH_DIAG_N=26 cargo test -p aleph-cuda --features cuquantum --release \
//!   -- --ignored --nocapture diag_bench
//! ```
//!
//! Each bench applies the *same* circuit under two routings and prints the
//! speedup, so the only variable is which kernel runs the diagonal gates.

#![cfg(all(target_os = "linux", feature = "cuda"))]

use std::time::Instant;

use aleph_backend::{run, Backend};
use aleph_core::{Gate, GateInstance, Param};
use aleph_cuda::CudaSvBackend;
use aleph_ir::Circuit;
use aleph_oracle::HasAmplitudes;

fn env_n(default: u32) -> u32 {
    std::env::var("ALEPH_DIAG_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn reps() -> u32 {
    std::env::var("ALEPH_DIAG_REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5)
}

fn env_depth(default: u32) -> u32 {
    std::env::var("ALEPH_DIAG_DEPTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// A deep diagonal-only workload: `depth` layers of (Rz on every qubit) + a Cz
/// brickwall. Every gate is diagonal, so this isolates the diagonal kernel.
fn diag_layers(n: u32, depth: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for d in 0..depth {
        for q in 0..n {
            c.rz(0.1 + 0.01 * (d as f64) + 0.001 * q as f64, q).unwrap();
        }
        let mut q = d % 2;
        while q + 1 < n {
            c.add_gate(GateInstance::new(Gate::Cz, vec![q, q + 1]))
                .unwrap();
            q += 2;
        }
    }
    c
}

/// QFT — the realistic controlled-Phase-dominated headline workload.
fn qft(n: u32) -> Circuit {
    let mut c = Circuit::new(n, 0);
    for j in 0..n {
        c.h(j).unwrap();
        for (offset, k) in ((j + 1)..n).enumerate() {
            let theta = std::f64::consts::PI / (1u64 << (offset + 1)) as f64;
            c.add_gate(GateInstance::controlled(
                Gate::Phase(Param::Concrete(theta)),
                vec![k],
                vec![j],
            ))
            .unwrap();
        }
    }
    c
}

/// Time `reps` warm runs of `circuit` on `backend`, forcing a device sync each
/// run by reading one amplitude. Returns the best (min) wall-clock seconds.
fn time_runs<B: Backend>(backend: &mut B, circuit: &Circuit, reps: u32) -> f64
where
    B::State: HasAmplitudes,
{
    // Warm-up (NVRTC/JIT, first-touch allocation, caches).
    let _ = run(backend, circuit).expect("warmup");
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        let st = run(backend, circuit).expect("timed run");
        let _ = HasAmplitudes::amplitudes(&st); // sync the stream
        best = best.min(t.elapsed().as_secs_f64());
    }
    best
}

fn bench_cuda_sv(name: &str, circuit: &Circuit, reps: u32) {
    let mut on = match CudaSvBackend::with_seed(0) {
        Ok(b) => b.with_custom_kernels(true),
        Err(e) => {
            eprintln!("skipping {name}: {e}");
            return;
        }
    };
    let mut off = CudaSvBackend::with_seed(0)
        .expect("device present")
        .with_custom_kernels(false);
    let t_on = time_runs(&mut on, circuit, reps);
    let t_off = time_runs(&mut off, circuit, reps);
    println!(
        "[hand-written SV] {name}: custom {:.4}s  dense {:.4}s  speedup {:.2}x",
        t_on,
        t_off,
        t_off / t_on
    );
}

#[cfg(feature = "cuquantum")]
fn bench_custatevec(name: &str, circuit: &Circuit, reps: u32) {
    use aleph_cuda::CuStateVecBackend;
    let mut on = match CuStateVecBackend::with_seed(0) {
        Ok(b) => b.with_custom_kernels(true),
        Err(e) => {
            eprintln!("skipping {name}: {e}");
            return;
        }
    };
    let mut off = CuStateVecBackend::with_seed(0)
        .expect("device present")
        .with_custom_kernels(false);
    let t_on = time_runs(&mut on, circuit, reps);
    let t_off = time_runs(&mut off, circuit, reps);
    println!(
        "[cuStateVec]      {name}: custom {:.4}s  cuQuantum {:.4}s  speedup {:.2}x",
        t_on,
        t_off,
        t_off / t_on
    );
}

#[test]
#[ignore]
fn diag_bench_layers() {
    let n = env_n(26);
    let depth = env_depth(30);
    let r = reps();
    let circuit = diag_layers(n, depth);
    println!("== diagonal layers n={n} depth={depth} reps={r} ==");
    bench_cuda_sv("diag-layers", &circuit, r);
    #[cfg(feature = "cuquantum")]
    bench_custatevec("diag-layers", &circuit, r);
}

/// The headline P5-06 sweep: a fixed-depth diagonal circuit across a range of
/// qubit counts. At large `n` the kernels are memory-bandwidth bound (custom ≈
/// dense ≈ cuStateVec); the custom kernel's win is at **small `n`**, where the
/// per-gate launch/dispatch overhead dominates and cuStateVec additionally pays a
/// `GetWorkspaceSize` query per gate that the bare custom launch skips.
#[test]
#[ignore]
fn diag_bench_sweep() {
    let depth = env_depth(200);
    let r = reps();
    // `ALEPH_DIAG_NS="4,8,12,..."` overrides the default sweep.
    let ns: Vec<u32> = std::env::var("ALEPH_DIAG_NS")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![4, 6, 8, 10, 12, 14, 16, 18, 20, 24]);
    println!("== diagonal sweep depth={depth} reps={r} ==");
    for n in ns {
        let circuit = diag_layers(n, depth);
        bench_cuda_sv(&format!("n={n:>2}"), &circuit, r);
        #[cfg(feature = "cuquantum")]
        bench_custatevec(&format!("n={n:>2}"), &circuit, r);
    }
}

#[test]
#[ignore]
fn diag_bench_qft() {
    let n = env_n(26);
    let r = reps();
    let circuit = qft(n);
    println!("== QFT n={n} reps={r} ==");
    bench_cuda_sv("qft", &circuit, r);
    #[cfg(feature = "cuquantum")]
    bench_custatevec("qft", &circuit, r);
}
