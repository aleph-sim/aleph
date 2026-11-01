//! #520 device-limit and safe public-API reach probe; no cap override or OOM test.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use aleph_backend::{Backend, BackendError};
    use aleph_ir::Circuit;
    use aleph_metal::{MetalContext, MetalSvBackend};
    let context = MetalContext::new()?;
    let device = context.device();
    let before = device.current_allocated_size();
    let mut backend = MetalSvBackend::with_seed(520)?;
    let rejects_29 = matches!(
        backend.allocate(29),
        Err(BackendError::TooManyQubits {
            requested: 29,
            limit: 28
        })
    );
    assert!(
        rejects_29,
        "public Metal SV cap changed: update probe rather than claiming an untested reach"
    );
    let mut circuit = Circuit::new(28, 0);
    circuit.h(0)?;
    for q in 1..28 {
        circuit.cnot(q - 1, q)?;
    }
    let state = backend.run(&circuit)?;
    let amplitudes = state.amplitudes_f32();
    let norm: f64 = amplitudes
        .iter()
        .map(|z| {
            assert!(z.re.is_finite() && z.im.is_finite());
            (z.re as f64).powi(2) + (z.im as f64).powi(2)
        })
        .sum();
    assert!((norm - 1.0).abs() < 1e-5, "n28 GHZ norm {norm}");
    let expected = std::f64::consts::FRAC_1_SQRT_2;
    assert!((amplitudes[0].re as f64 - expected).abs() < 1e-5);
    assert!((amplitudes[amplitudes.len() - 1].re as f64 - expected).abs() < 1e-5);
    println!("{{\"max_buffer_length_bytes\":{},\"recommended_working_set_bytes\":{},\"process_device_bytes_before\":{},\"process_device_bytes_with_state\":{},\"measured_public_sv_qubits\":28,\"state_bytes\":1073741824,\"public_sv_n29_rejected_by_policy\":true,\"norm\":{},\"full_finite_scan\":true}}",device.max_buffer_length(),device.recommended_max_working_set_size(),before,device.current_allocated_size(),norm);
    Ok(())
}
#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn main() {
    panic!("metal_m5_audit requires macOS and --features metal");
}
