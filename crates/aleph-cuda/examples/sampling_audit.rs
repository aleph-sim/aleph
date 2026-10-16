//! Bounded real-device sampler audit: precision mode qubits shots repetitions.
//! Preparation is excluded. Use profile=1 for synchronized stage attribution.
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use aleph_backend::{run, Backend};
    use aleph_cuda::{CudaContext, CudaSvBackend, CudaSvBackendF32, SamplingAlgorithm};
    use std::time::Instant;
    let a: Vec<_> = std::env::args().collect();
    let precision = a.get(1).map(String::as_str).unwrap_or("f64");
    let mode = match a.get(2).map(String::as_str).unwrap_or("auto") {
        "reference" => SamplingAlgorithm::Reference,
        "hierarchical" => SamplingAlgorithm::Hierarchical,
        "block" => SamplingAlgorithm::BlockCdf,
        "auto" => SamplingAlgorithm::Auto,
        _ => return Err("unknown mode".into()),
    };
    let n: u32 = a.get(3).map(String::as_str).unwrap_or("18").parse()?;
    let shots: u32 = a.get(4).map(String::as_str).unwrap_or("1024").parse()?;
    let reps: usize = a.get(5).map(String::as_str).unwrap_or("7").parse()?;
    let profile = a.get(6).is_some_and(|s| s == "1");
    if n > 22 || shots > 1_000_000 || reps == 0 {
        return Err("bounded audit: n<=22, shots<=1M, reps>0".into());
    }
    let ctx = CudaContext::new(0)?;
    let mut circuit = aleph_ir::Circuit::new(n, 0);
    for q in 0..n {
        circuit.h(q)?;
    }
    macro_rules! audit { ($backend:expr) => {{
        let mut be = $backend.with_sampling_algorithm(mode).with_sampling_profile(profile);
        let state = run(&mut be, &circuit)?;
        ctx.synchronize()?;
        be.sample(&state, shots)?;
        let mut times = Vec::new();
        for _ in 0..reps {
            let start=Instant::now();
            let result=be.sample(&state,shots)?;
            times.push(start.elapsed().as_secs_f64()*1000.);
            assert_eq!(result.len(), shots as usize);
            assert!(result.iter().all(|&i| i < (1 << n)));
        }
        times.sort_by(f64::total_cmp);
        let s=be.sampling_stats();
        let kind=if precision=="f32" {aleph_backend::BackendKind::CudaF32} else {aleph_backend::BackendKind::Cuda};
        let mut request=aleph_cuda::memory::frontend_memory_request(&circuit,kind,shots.into(),false,false)?;
        request.sampling_workspace_bytes=Some(aleph_cuda::sampling_workspace_bytes(n,shots,mode)?);
        let estimate=aleph_backend::memory::estimate_dense_memory(request,aleph_cuda::memory::CUDA_READOUT_LAYOUT)?;
        assert!(ctx.pool_used_high_bytes().unwrap_or(0)<=estimate.device_peak_bytes, "sampler allocation model underestimated live bytes");
        println!("{{\"precision\":\"{precision}\",\"mode\":\"{mode:?}\",\"qubits\":{n},\"shots\":{shots},\"profile\":{profile},\"median_ms\":{},\"min_ms\":{},\"max_ms\":{},\"norm_ms\":{},\"build_ms\":{},\"draw_ms\":{},\"download_ms\":{},\"workspace_bytes\":{},\"build_launches\":{},\"pool_peak\":{}}}",times[reps/2],times[0],times[reps-1],s.norm_ms,s.build_ms,s.draw_ms,s.download_ms,s.workspace_bytes,s.build_launches,ctx.pool_used_high_bytes().unwrap_or(0));
    }}}
    match precision {
        "f32" => audit!(CudaSvBackendF32::with_seed(555)?),
        "f64" => audit!(CudaSvBackend::with_seed(555)?),
        _ => return Err("precision must be f32/f64".into()),
    }
    Ok(())
}
#[cfg(not(all(target_os = "linux", feature = "cuda")))]
fn main() {
    eprintln!("requires Linux + --features cuda and a real device");
}
