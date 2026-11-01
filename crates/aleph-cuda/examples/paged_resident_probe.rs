//! P8-14 bounded FP64 paging probe: MODEL N M sync|resident|overlap|combined [HOT].
//! Models: locality, alternating, short, qft. Backend JIT excluded; allocation/init,
//! execution and drain included. Norm/export excluded. All byte counts are models.
#[cfg(all(target_os = "linux", feature = "cuda"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use aleph_cuda::{paged_pass_counts, CudaSvBackend};
    use aleph_ir::Circuit;
    use std::time::Instant;
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(4..=5).contains(&args.len()) {
        return Err("usage: MODEL N M sync|resident|overlap|combined [HOT]".into());
    }
    let model = &args[0];
    let n: u32 = args[1].parse()?;
    let m: u32 = args[2].parse()?;
    // This experiment deliberately stays far below OOM and full-device reach.
    let hot: u32 = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(m);
    if !(4..=26).contains(&n)
        || m < 2
        || m + 2 > n
        || m > 24
        || hot < 2
        || hot > m
        || (m > 22 && args[3] != "resident")
    {
        return Err("outside bounded experiment".into());
    }
    let mut c = Circuit::new(n, 0);
    if model == "qft" {
        use aleph_core::{Gate, GateInstance, Param};
        for q in 0..n {
            c.h(q)?;
            for k in q + 1..n {
                c.add_gate(GateInstance::controlled(
                    Gate::Phase(Param::Concrete(
                        std::f64::consts::PI / 2f64.powi((k - q) as i32),
                    )),
                    vec![k],
                    vec![q],
                ))?;
            }
        }
    }
    for layer in 0..if model == "qft" { 0 } else { 4 } {
        match model.as_str() {
            "locality" => {
                for q in 0..hot {
                    c.ry(0.13 + (q + layer) as f64 * 0.017, q)?;
                }
                for q in 0..hot - 1 {
                    c.cnot(q, q + 1)?;
                }
                c.cnot(hot - 1, n - 1)?;
            }
            "alternating" => {
                for q in 0..4 {
                    c.rx(0.17, q % hot)?;
                    c.cnot(q % hot, n - 1 - q % 2)?;
                }
            }
            "short" => {
                c.ry(0.29, 0)?;
                c.rx(-0.19, 1)?;
                c.cnot(n - 2, n - 1)?;
            }
            _ => return Err("unknown model".into()),
        }
    }
    let (per_gate, passes) = paged_pass_counts(&c, m);
    let mut low_runs = Vec::new();
    let mut len = 0;
    let mut overlapped_segments = 0;
    let mut fallback_segments = 0;
    let mut launches = 0u64;
    let mut flush = |len: u64, hh: u32| {
        let groups = 1u64 << (n - m - hh);
        if groups >= 8 {
            overlapped_segments += 1;
        } else {
            fallback_segments += 1;
        }
        launches += groups * len;
    };
    for inst in c.instructions() {
        if let aleph_ir::Instruction::Gate(g) = inst {
            let mut high: Vec<_> = g
                .qubits
                .iter()
                .chain(&g.controls)
                .copied()
                .filter(|&q| q >= m)
                .collect();
            high.sort_unstable();
            high.dedup();
            if high.is_empty() {
                len += 1;
            } else {
                if len > 0 {
                    low_runs.push(len);
                    flush(len, 0);
                    len = 0;
                }
                flush(1, high.len() as u32);
            }
        }
    }
    if len > 0 {
        low_runs.push(len);
        flush(len, 0);
    }
    let mut gpu = CudaSvBackend::with_seed(564)?;
    let start = Instant::now();
    let st = match args[3].as_str() {
        "sync" => gpu.run_paged(&c, m)?,
        "resident" => gpu.run_paged_batched(&c, m)?,
        "overlap" => gpu.run_paged_overlapped(&c, m, 4)?,
        "combined" => gpu.run_paged_batched_overlapped(&c, m, 4)?,
        _ => return Err("unknown mode".into()),
    };
    let ns = start.elapsed().as_nanos();
    let norm = st.norm_sqr();
    if !norm.is_finite() || (norm - 1.0).abs() > 1e-9 {
        return Err("norm failed".into());
    }
    eprintln!("modelled=true state_bytes={} passes_per_gate={per_gate} residency_passes={passes} resident_transfer_bytes={} low_run_lengths={low_runs:?} combined_overlap_segments={overlapped_segments} combined_fallback_segments={fallback_segments} combined_gate_launches={launches}",16u64<<n, (32u64<<n)*passes);
    println!(
        "model,n,m,hot,mode,ns,norm\n{model},{n},{m},{hot},{},{ns},{norm:.15}",
        args[3]
    );
    Ok(())
}
#[cfg(not(all(target_os = "linux", feature = "cuda")))]
fn main() {
    eprintln!("requires Linux CUDA and real device");
    std::process::exit(1);
}
