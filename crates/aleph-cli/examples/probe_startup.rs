//! Compare public CLI startup with/without the former eager SV construction.
//! Run separate interleaved processes: cargo run --release -p aleph-cli --features
//! metal (or cuda) --example probe_startup -- legacy|lazy [repeats].
use aleph_backend::{BackendKind, BackendRequest};
use aleph_cli::{cli::Precision, exec::run_circuit};
use std::{path::Path, time::Instant};

fn inits() -> u64 {
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
fn legacy_probe() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        let _ = aleph_metal::MetalSvBackend::new()?;
    }
    #[cfg(all(target_os = "linux", feature = "cuda"))]
    {
        let _ = aleph_cuda::CudaSvBackendF32::new()?;
    }
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if !cfg!(any(
        all(target_os = "macos", feature = "metal"),
        all(target_os = "linux", feature = "cuda")
    )) {
        return Err("benchmark requires the platform GPU feature (metal or cuda)".into());
    }
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "lazy".into());
    if mode != "legacy" && mode != "lazy" {
        return Err("expected legacy or lazy".into());
    }
    let repeats: usize = args.next().as_deref().unwrap_or("20").parse()?;
    if repeats == 0 {
        return Err("repeats must be positive".into());
    }
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../oracle/circuits/bell_phi_plus.qasm");
    let before = inits();
    let mut times = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        // Replay the old availability call, then the same public CPU workflow.
        // This isolates the removed setup work; it is not an old-revision binary.
        if mode == "legacy" {
            legacy_probe()?;
        }
        run_circuit(
            &path,
            Some(32),
            false,
            false,
            &[],
            Some(7),
            Precision::Auto,
            BackendRequest::Fixed(BackendKind::Statevector),
            128,
            None,
            &[],
            &mut Vec::new(),
        )?;
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let first = times[0];
    let rest = if times.len() > 1 {
        &mut times[1..]
    } else {
        &mut times[..]
    };
    rest.sort_by(f64::total_cmp);
    let median = rest[rest.len() / 2];
    println!("mode={mode},calls={repeats},first_ms={first:.6},warm_median_ms={median:.6},sv_backend_inits={}",inits()-before);
    Ok(())
}
