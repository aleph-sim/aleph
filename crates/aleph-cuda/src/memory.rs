//! CUDA frontend preflight for the current operation-selected sampling workspace. Update this layout
//! and allocation tests when changing the sampler (#555). No large allocation occurs here.
use crate::CudaContext;
use aleph_backend::memory::{
    estimate_dense_memory, DenseMemoryEstimate, DenseMemoryLayout, DenseMemoryRequest, MemoryBudget,
};
use aleph_backend::BackendKind;
use aleph_ir::{Circuit, Instruction};

/// Reduction layout and legacy reference-sampler fallback (both precisions).
/// Frontend requests override sampling bytes with the selected #555 workspace.
pub const CUDA_READOUT_LAYOUT: DenseMemoryLayout = DenseMemoryLayout {
    sample_bytes_per_amplitude: 16, // two f64 arrays, each 2^n
    reduction_block: 256,
    reduction_bytes_per_block: 16, // two f64 partials per block
    accumulator_bytes: 16,
};

/// Linux MemAvailable, capped by the remaining cgroup v2 memory limit when present.
/// This excludes other processes, already retained results and resident pool pages.
pub fn host_available_bytes() -> Result<u64, String> {
    let text =
        std::fs::read_to_string("/proc/meminfo").map_err(|e| format!("query host memory: {e}"))?;
    let mut available = text
        .lines()
        .find_map(|l| {
            l.strip_prefix("MemAvailable:")
                .and_then(|s| s.split_whitespace().next())
                .and_then(|s| s.parse::<u64>().ok())
        })
        .and_then(|n| n.checked_mul(1024))
        .ok_or_else(|| "MemAvailable is unavailable".to_string())?;
    let root = std::path::Path::new("/sys/fs/cgroup");
    let membership = std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
    let group = membership
        .lines()
        .find_map(|l| l.strip_prefix("0::"))
        .unwrap_or("/");
    available = cgroup_available(root, group, available)?;
    Ok(available)
}

// Walk process/ancestor cgroup-v2 limits. In a cgroup namespace the mount may
// already be rooted at the process group; fall back to that visible root.
fn cgroup_available(
    root: &std::path::Path,
    group: &str,
    mut available: u64,
) -> Result<u64, String> {
    let relative = std::path::Path::new(group.trim_start_matches('/'));
    let safe = relative
        .components()
        .all(|c| matches!(c, std::path::Component::Normal(_)));
    let joined = root.join(relative);
    let mut dir = if safe && joined.is_dir() {
        joined
    } else {
        root.to_path_buf()
    };
    loop {
        if let Ok(text) = std::fs::read_to_string(dir.join("memory.max")) {
            if text.trim() != "max" {
                let limit = text
                    .trim()
                    .parse::<u64>()
                    .map_err(|e| format!("parse cgroup limit: {e}"))?;
                let used = std::fs::read_to_string(dir.join("memory.current"))
                    .map_err(|e| format!("query cgroup memory: {e}"))?
                    .trim()
                    .parse::<u64>()
                    .map_err(|e| format!("parse cgroup usage: {e}"))?;
                available = available.min(limit.saturating_sub(used));
            }
        }
        if dir == root || !dir.pop() {
            break;
        }
    }
    Ok(available)
}

/// Check an incremental host copy independently of device capacity (also used by lazy export).
pub fn ensure_host_bytes(required: u64) -> Result<(), String> {
    let available = host_available_bytes()?;
    let usable = available.saturating_sub((64 << 20).max(available / 20));
    if required > usable {
        Err(format!("CUDA host export needs {required} incremental bytes, available {usable} after headroom; reduce export size or release retained results"))
    } else {
        Ok(())
    }
}

/// Query actual device free/total memory. Cached pool blocks are deliberately not
/// added back: they may belong to another context/user. Headroom covers driver/JIT,
/// allocation granularity and fragmentation, not a reservation against concurrent tenants.
pub fn runtime_memory_budget(ctx: &CudaContext) -> Result<MemoryBudget, String> {
    let (free, total) = ctx
        .raw()
        .mem_get_info()
        .map_err(|e| format!("query CUDA memory: {e}"))?;
    let host = host_available_bytes()?;
    Ok(MemoryBudget {
        device_free_bytes: free as u64,
        device_total_bytes: total as u64,
        device_headroom_bytes: (256 << 20).max(total as u64 / 20),
        host_available_bytes: host,
        host_headroom_bytes: (64 << 20).max(host / 20),
    })
}

/// Requirements of the raw CUDA frontend executor, including matrix and phase scratch.
/// The pure planner remains independent of this CUDA/IR mapping.
pub fn frontend_memory_request(
    c: &Circuit,
    kind: BackendKind,
    shots: u64,
    expectation: bool,
    export: bool,
) -> Result<DenseMemoryRequest, String> {
    let amplitude_bytes = match kind {
        BackendKind::Cuda => 16,
        BackendKind::CudaF32 => 8,
        _ => return Err("CUDA memory plan requires a CUDA backend".into()),
    };
    let mut matrix = 1024u64; // at most 3-qubit ordinary matrix; f64 coefficients in both precisions
    let mut phases = 0u64;
    let mut measurements = shots == 0; // sample(0) still checks norm; conservative for export-only runs
    for inst in c.instructions() {
        match inst {
            Instruction::Gate(g) => {
                if let aleph_core::Gate::UnitaryKq { data, .. } = &g.gate {
                    matrix = matrix.max(
                        (data.len() as u64)
                            .checked_mul(16)
                            .ok_or("matrix scratch overflow")?,
                    );
                }
            }
            Instruction::DiagonalPhase(dp) => {
                let terms = dp.terms.len() as u64;
                let conds = dp
                    .terms
                    .iter()
                    .try_fold(0u64, |a, t| a.checked_add(t.conds.len() as u64))
                    .ok_or("phase scratch overflow")?
                    .max(1);
                let bytes = terms
                    .checked_mul(8)
                    .and_then(|a| conds.checked_mul(8).and_then(|b| a.checked_add(b)))
                    .and_then(|a| {
                        terms
                            .checked_add(1)
                            .and_then(|t| t.checked_mul(4))
                            .and_then(|b| a.checked_add(b))
                    })
                    .ok_or("phase scratch overflow")?;
                phases = phases.max(bytes);
            }
            Instruction::Measure { .. } | Instruction::Reset(_) => measurements = true,
            _ => {}
        }
    }
    // Include replacement/growth overlap and retaining-pool slack for gate scratch.
    let execution_scratch_bytes = matrix
        .checked_mul(2)
        .and_then(|a| a.checked_add(phases))
        .ok_or("execution scratch overflow")?;
    Ok(DenseMemoryRequest {
        qubits: c.num_qubits(),
        amplitude_bytes,
        shots,
        sampling_workspace_bytes: Some(crate::sampling_workspace_bytes(
            c.num_qubits(),
            u32::try_from(shots).map_err(|_| "CUDA shots exceed u32")?,
            crate::SamplingAlgorithm::Auto,
        )?),
        expectation,
        export_amplitudes: export,
        execution_scratch_bytes,
        measurements,
        marginal_qubits: None,
    })
}

/// Shared CLI/Python preflight before SV construction or state allocation. Returns an
/// actionable error instead of choosing the norm-only paged route for requested views.
/// Keeps the discovery context alive so backend construction reuses its primary context.
pub struct FrontendMemoryPlan {
    pub estimate: DenseMemoryEstimate,
    pub context: CudaContext,
}

/// Validate the current frontend request and retain the small discovery context.
/// No state or readout scratch is allocated by this function.
pub fn plan_frontend(
    c: &Circuit,
    kind: BackendKind,
    shots: u64,
    expectation: bool,
    export: bool,
) -> Result<FrontendMemoryPlan, String> {
    let request = frontend_memory_request(c, kind, shots, expectation, export)?;
    let estimate = estimate_dense_memory(request, CUDA_READOUT_LAYOUT)?;
    let ctx = CudaContext::new(0).map_err(|e| format!("CUDA memory discovery: {e}"))?;
    let estimate = estimate.check(runtime_memory_budget(&ctx)?)?;
    Ok(FrontendMemoryPlan {
        estimate,
        context: ctx,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selected_sampler_memory_drives_frontend_feasibility() {
        let c = Circuit::new(31, 0);
        let request =
            frontend_memory_request(&c, BackendKind::CudaF32, 1024, false, false).unwrap();
        let e = estimate_dense_memory(request, CUDA_READOUT_LAYOUT).unwrap();
        assert!(e.sampling_bytes < 65 << 20);
        let budget = MemoryBudget {
            device_free_bytes: 20 << 30,
            device_total_bytes: 20 << 30,
            device_headroom_bytes: 1 << 30,
            host_available_bytes: 64 << 30,
            host_headroom_bytes: 1 << 30,
        };
        assert!(e.check(budget).is_ok());
        let many =
            frontend_memory_request(&c, BackendKind::CudaF32, 1_000_000, false, false).unwrap();
        assert!(estimate_dense_memory(many, CUDA_READOUT_LAYOUT)
            .unwrap()
            .check(budget)
            .is_err());
        assert!(frontend_memory_request(&c, BackendKind::CudaF32, u64::MAX, false, false).is_err());
    }
    #[test]
    fn cgroup_leaf_and_parent_limits_are_both_enforced() {
        let root = std::env::temp_dir().join(format!("aleph-554-cgroup-{}", std::process::id()));
        std::fs::create_dir_all(root.join("parent/leaf")).unwrap();
        for (path, limit, used) in [
            ("", "max", "0"),
            ("parent", "1000", "800"),
            ("parent/leaf", "500", "50"),
        ] {
            std::fs::write(root.join(path).join("memory.max"), limit).unwrap();
            std::fs::write(root.join(path).join("memory.current"), used).unwrap();
        }
        assert_eq!(cgroup_available(&root, "/parent/leaf", 2000).unwrap(), 200);
        assert_eq!(cgroup_available(&root, "/parent/leaf", 100).unwrap(), 100);
        std::fs::write(root.join("parent/leaf/memory.current"), "600").unwrap();
        assert_eq!(cgroup_available(&root, "/parent/leaf", 2000).unwrap(), 0);
        std::fs::remove_dir_all(root).unwrap();
    }
}
