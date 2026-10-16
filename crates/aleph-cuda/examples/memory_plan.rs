//! Real-device allocation audit for #554. Each case runs in a fresh process:
//! cargo run --release -p aleph-cuda --features cuda --example memory_plan -- f32 sample
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use aleph_backend::{memory::estimate_dense_memory, run, Backend, BackendKind};
    use aleph_cuda::memory::{frontend_memory_request, runtime_memory_budget, CUDA_READOUT_LAYOUT};
    use aleph_cuda::{CudaContext, CudaSvBackend, CudaSvBackendF32};
    let args: Vec<_> = std::env::args().collect();
    let fp32 = args.get(1).is_some_and(|s| s == "f32");
    let view = args.get(2).map(String::as_str).unwrap_or("sample");
    let ctx = CudaContext::new(0)?;
    let budget = runtime_memory_budget(&ctx)?;
    let kind = if fp32 {
        BackendKind::CudaF32
    } else {
        BackendKind::Cuda
    };
    let mut c = aleph_ir::Circuit::new(18, 0);
    c.h(0)?;
    c.cnot(0, 17)?;
    let mut request = frontend_memory_request(
        &c,
        kind,
        if view == "sample" { 64 } else { 0 },
        view == "expectation",
        view == "export",
    )?;
    if view == "marginal" {
        request.marginal_qubits = Some(2);
    }
    let estimate = estimate_dense_memory(request, CUDA_READOUT_LAYOUT)?.check(budget)?;
    macro_rules! audit {($backend:expr)=>{{
        let mut be=$backend;
        let state=run(&mut be,&c)?;
        match view {
            "sample"=>{let samples=be.sample(&state,64)?;assert!(samples.iter().all(|&v|v==0||v==(1|(1<<17))));},
            "marginal"=>{let v=be.probabilities(&state,&[0,17])?;assert!((v[0]-0.5).abs()<1e-5&&(v[3]-0.5).abs()<1e-5);},
            "expectation"=>{let p=aleph_core::PauliString::new(1.0,vec![(0,aleph_core::Pauli::Z)])?;assert!(be.expectation_value(&state,&p)?.abs()<1e-5);},
            "export"=>{let v=state.try_amplitudes_vec()?;assert!((v[0].re-std::f64::consts::FRAC_1_SQRT_2).abs()<1e-5);},
            "execute"=>{},_=>return Err("unknown view".into()),
        }
        ctx.synchronize()?;
        let peak=ctx.pool_used_high_bytes().ok_or("pool peak counter unavailable")?;
        assert!(peak<=estimate.device_peak_bytes,"measured {peak} exceeds estimate {}",estimate.device_peak_bytes);
        assert!(estimate.device_peak_bytes - peak <= estimate.execution_scratch_bytes + estimate.reduction_bytes,
            "allocation layout drifted: update the memory model with the sampler/readout");
        println!("{{\"precision\":\"{}\",\"view\":\"{view}\",\"device_estimate\":{},\"pool_live_peak\":{peak},\"host_estimate\":{},\"device_free\":{}}}",if fp32 {"f32"} else {"f64"},estimate.device_peak_bytes,estimate.host_peak_bytes,budget.device_free_bytes);
    }}}
    if fp32 {
        audit!(CudaSvBackendF32::with_seed(554)?);
    } else {
        audit!(CudaSvBackend::with_seed(554)?);
    }
    // A budget rejection must happen before backend construction or state allocation.
    let before = aleph_cuda::device_alloc_count();
    let init = aleph_cuda::sv_backend_init_count();
    let huge = aleph_ir::Circuit::new(64, 0);
    assert!(
        aleph_cuda::memory::plan_frontend(&huge, BackendKind::CudaF32, 1024, false, false).is_err()
    );
    assert_eq!(aleph_cuda::device_alloc_count(), before);
    assert_eq!(aleph_cuda::sv_backend_init_count(), init);
    Ok(())
}
#[cfg(not(all(target_os = "linux", feature = "cuda")))]
fn main() {
    eprintln!("requires Linux + --features cuda and a real CUDA device");
    std::process::exit(1);
}
