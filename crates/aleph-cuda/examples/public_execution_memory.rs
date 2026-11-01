//! Fresh-process peak audit of the exact fused plan used by frontends (#552).
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use aleph_backend::{memory::estimate_dense_memory, Backend, BackendKind};
    use aleph_cuda::{
        prepare_cuda_execution, CudaContext, CudaExecutionOptions, CudaReadoutRequest,
        CudaSvBackend, CudaSvBackendF32,
    };
    let args: Vec<_> = std::env::args().collect();
    let f32 = args.get(1).is_some_and(|s| s == "f32");
    let optimize = args.get(2).is_none_or(|s| s != "raw");
    let ctx = CudaContext::new(0)?;
    let mut c = aleph_ir::Circuit::new(18, 0);
    for q in 0..18 {
        c.h(q)?;
        for j in q + 1..18 {
            c.add_gate(aleph_core::GateInstance::controlled(
                aleph_core::Gate::Phase(aleph_core::Param::Concrete(
                    0.3 / (2f64.powi((j - q) as i32)),
                )),
                vec![q],
                vec![j],
            ))?;
        }
    }
    macro_rules! audit {($backend:ty,$kind:expr)=>{{
        let (mut be,plan)=prepare_cuda_execution(&c,CudaExecutionOptions{optimize},CudaReadoutRequest{shots:32,..Default::default()},||<$backend>::with_seed(552))?;
        let request=aleph_cuda::memory::frontend_memory_request(plan.circuit(),$kind,32,false,false)?;
        let estimate=estimate_dense_memory(request,aleph_cuda::memory::CUDA_READOUT_LAYOUT)?;
        let state=plan.execute(&mut be)?;be.sample(&state,32)?;ctx.synchronize()?;
        let peak=ctx.pool_used_high_bytes().ok_or("missing live peak counter")?;
        assert!(peak<=estimate.device_peak_bytes,"planned scratch estimate underestimated peak");
        println!("{{\"fp32\":{f32},\"optimize\":{optimize},\"device_estimate\":{},\"device_live_peak\":{peak},\"execution_scratch\":{}}}",estimate.device_peak_bytes,estimate.execution_scratch_bytes);
    }}}
    if f32 {
        audit!(CudaSvBackendF32, BackendKind::CudaF32);
    } else {
        audit!(CudaSvBackend, BackendKind::Cuda);
    }
    Ok(())
}
#[cfg(not(all(target_os = "linux", feature = "cuda")))]
fn main() {
    eprintln!("requires Linux CUDA device");
    std::process::exit(1);
}
