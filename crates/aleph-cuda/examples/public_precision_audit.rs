//! Bounded reproduction of the TF32 candidate rejected from public execution (#552).
//! This is a diagnostic, not an accuracy-pass test. Strict public plans use FP32 ALU.
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use aleph_backend::run;
    use aleph_cuda::{fuse_for_gpu_tf32, CudaSvBackendF32};
    let mut c = aleph_ir::Circuit::new(8, 0);
    for q in 0..8 {
        c.h(q)?;
    }
    c.barrier((0..8).collect::<Vec<_>>())?;
    for layer in 0..3 {
        for q in 0..8 {
            c.ry(0.13 + q as f64 * 0.07 + layer as f64 * 0.03, q)?;
        }
        for q in 0..7 {
            c.cnot(q, q + 1)?;
        }
        for q in 0..8 {
            c.rz(0.11 + q as f64 * 0.02, q)?;
        }
    }
    let ir = fuse_for_gpu_tf32(&c);
    let mut tf32 = CudaSvBackendF32::with_seed(552)?;
    if !tf32.tf32_available() {
        return Err("TF32 module unavailable; audit did not execute".into());
    }
    let candidate = tf32.run_layered(&ir)?.amplitudes_vec();
    let ordinary = CudaSvBackendF32::with_seed(552)?
        .with_tf32_kq(false)
        .run_layered(&ir)?
        .amplitudes_vec();
    let cpu = run(&mut aleph_sv::NaiveSvBackend::new(), &c)?;
    let vs_alu = candidate
        .iter()
        .zip(&ordinary)
        .map(|(a, b)| (*a - *b).norm())
        .fold(0f64, f64::max);
    let vs_cpu = candidate
        .iter()
        .zip(cpu.amplitudes())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0f64, f64::max);
    println!("{{\"executed\":true,\"qubits\":8,\"tf32_vs_same_ir_fp32_max_abs\":{vs_alu},\"tf32_vs_cpu_max_abs\":{vs_cpu},\"previous_empirical_budget\":0.0001,\"within_previous_budget\":{}}}",vs_alu<=1e-4);
    Ok(())
}
#[cfg(not(all(target_os = "linux", feature = "cuda")))]
fn main() {
    eprintln!("requires real CUDA TF32 hardware");
    std::process::exit(1);
}
