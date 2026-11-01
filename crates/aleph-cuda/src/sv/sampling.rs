//! Shared FP32/FP64 hierarchical sampling, with an explicit unchanged reference path.
use crate::{CudaContext, DeviceBuffer, Error};
use cudarc::driver::{CudaFunction, CudaModule, DeviceRepr, LaunchConfig, PushKernelArg};
use rand::{rngs::StdRng, Rng};
use std::{sync::Arc, time::Instant};

/// Seeded determinism holds for a fixed state, resolved algorithm, precision and
/// device/build. Auto may resolve differently when shots change. Regrouped FP64
/// sums need not preserve the reference's seed-to-outcome mapping.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SamplingAlgorithm {
    #[default]
    /// Choose from state size and shots using the measured conservative policy.
    Auto,
    /// Original two-buffer Hillis–Steele scan, retained for A/B comparisons.
    Reference,
    /// Reuse norm-block masses, then scan only selected amplitude blocks.
    Hierarchical,
    /// Build one block-local FP64 CDF and a small mass hierarchy.
    BlockCdf,
}
impl SamplingAlgorithm {
    /// Conservative crossover measured on RTX 4000 SFF Ada (see #555 report).
    /// Small states use BlockCdf; sparse-shot large states avoid its full-state
    /// workspace. Explicit variants support calibration on another device.
    pub fn resolve(self, n: u64, shots: u32) -> Self {
        if self != Self::Auto {
            return self;
        }
        if n <= 65536 || u64::from(shots).saturating_mul(4096) > n {
            Self::BlockCdf
        } else {
            Self::Hierarchical
        }
    }
}
/// Stage wall times in ms. With profiling enabled, synchronization attributes GPU
/// work to each stage; otherwise these measure host submissions and the download
/// includes queued GPU work. Build includes allocation; draw includes RNG/upload.
#[derive(Clone, Copy, Debug, Default)]
pub struct SamplingStats {
    /// Norm validation, including its scalar download.
    pub norm_ms: f64,
    /// Budget query, workspace allocation, and hierarchy/CDF construction.
    pub build_ms: f64,
    /// Uniform generation/upload, output allocation and inverse transforms.
    pub draw_ms: f64,
    /// Output download (and queued GPU work if profiling is disabled).
    pub download_ms: f64,
    /// Device scratch, excluding persistent norm buffers and 16 bytes per shot.
    pub workspace_bytes: u64,
    /// Construction kernels, excluding norm reduction and sampling search.
    pub build_launches: u32,
}

fn sizes(n: u64) -> Vec<u64> {
    let mut m = n.div_ceil(256);
    let mut out = vec![m];
    while m > 256 {
        m = m.div_ceil(256);
        out.push(m);
    }
    out
}
/// Exact sampler workspace (excludes persistent norm partials and 16 bytes/shot I/O).
/// No allocations proportional to the state are performed while computing the bound.
pub fn sampling_workspace_bytes(
    qubits: u32,
    shots: u32,
    mode: SamplingAlgorithm,
) -> Result<u64, String> {
    if shots == 0 {
        return Ok(0);
    }
    let n = 1u64
        .checked_shl(qubits)
        .ok_or("sampling dimension overflow")?;
    let mode = mode.resolve(n, shots);
    if mode == SamplingAlgorithm::Reference {
        return n
            .checked_mul(16)
            .ok_or_else(|| "sampling scratch overflow".into());
    }
    let entries = sizes(n)
        .into_iter()
        .try_fold(0u64, |sum, m| {
            sum.checked_add(m)?.checked_add(m.div_ceil(256))
        })
        .ok_or("sampling hierarchy overflow")?;
    let bytes = entries.checked_mul(8).ok_or("sampling scratch overflow")?;
    if mode == SamplingAlgorithm::BlockCdf {
        bytes
            .checked_add(n.checked_mul(8).ok_or("sampling CDF overflow")?)
            .ok_or_else(|| "sampling scratch overflow".into())
    } else {
        bytes
            .checked_add(8)
            .ok_or_else(|| "sampling scratch overflow".into())
    }
}

pub(crate) struct Sampler {
    local: CudaFunction,
    mass: CudaFunction,
    draw: CudaFunction,
}
impl Sampler {
    pub fn new(module: &Arc<CudaModule>) -> Result<Self, Error> {
        Ok(Self {
            local: module.load_function("sample_local_cdf")?,
            mass: module.load_function("sample_mass_cdf")?,
            draw: module.load_function("sample_hierarchy")?,
        })
    }
    #[allow(clippy::too_many_arguments)]
    pub fn sample<T: DeviceRepr>(
        &self,
        ctx: &CudaContext,
        rng: &mut StdRng,
        amps: &DeviceBuffer<T>,
        partials: &DeviceBuffer<f64>,
        qubits: u32,
        shots: u32,
        mode: SamplingAlgorithm,
        profile: bool,
    ) -> Result<(Vec<u64>, SamplingStats), Error> {
        let n = 1u64 << qubits;
        let mode = mode.resolve(n, shots);
        let start = Instant::now();
        let workspace = sampling_workspace_bytes(qubits, shots, mode).map_err(Error::Memory)?;
        let required = workspace
            .checked_add(u64::from(shots) * 16)
            .ok_or_else(|| Error::Memory("sampling allocation overflow".into()))?;
        let b = crate::memory::runtime_memory_budget(ctx).map_err(Error::Memory)?;
        if required > b.device_free_bytes.saturating_sub(b.device_headroom_bytes) {
            return Err(Error::Memory(format!(
                "sampling needs {required} incremental bytes; exceeds current free-memory budget"
            )));
        }
        let stream = ctx.stream();
        let dims = sizes(n);
        let flat_len = dims.iter().sum::<u64>();
        let mut flat = DeviceBuffer::<f64>::zeros(ctx, flat_len as usize)?;
        let mut leaf = DeviceBuffer::<f64>::zeros(
            ctx,
            if mode == SamplingAlgorithm::BlockCdf {
                n as usize
            } else {
                1
            },
        )?;
        let mut launches = 0;
        if mode == SamplingAlgorithm::BlockCdf {
            // SAFETY: amplitudes hold 2*N scalars, leaf holds N doubles; grid covers N.
            unsafe {
                stream
                    .launch_builder(&self.local)
                    .arg(amps.slice())
                    .arg(leaf.slice_mut())
                    .arg(&n)
                    .launch(config(n))?;
            }
            launches += 1;
        }
        let mut totals: Vec<DeviceBuffer<f64>> = Vec::new();
        let mut offset = 0u64;
        for (level, &m) in dims.iter().enumerate() {
            let mut next = DeviceBuffer::<f64>::zeros(ctx, m.div_ceil(256) as usize)?;
            let kind = if level == 0 {
                if mode == SamplingAlgorithm::BlockCdf {
                    1u32
                } else {
                    0u32
                }
            } else {
                2u32
            };
            let input = if level == 0 {
                if kind == 1 {
                    &leaf
                } else {
                    partials
                }
            } else {
                &totals[level - 1]
            };
            // SAFETY: input has 2*m norm entries, N local-CDF entries, or m totals;
            // flat[offset..offset+m] and next[ceil(m/256)] are disjoint owned buffers.
            unsafe {
                stream
                    .launch_builder(&self.mass)
                    .arg(input.slice())
                    .arg(flat.slice_mut())
                    .arg(next.slice_mut())
                    .arg(&m)
                    .arg(&offset)
                    .arg(&kind)
                    .arg(&n)
                    .launch(config(m))?;
            }
            totals.push(next);
            offset += m;
            launches += 1;
        }
        if profile {
            ctx.synchronize()?;
        }
        let built = Instant::now();
        let uniforms: Vec<f64> = (0..shots).map(|_| rng.gen::<f64>()).collect();
        let targets = DeviceBuffer::from_slice(ctx, &uniforms)?;
        let mut output = DeviceBuffer::<u64>::zeros(ctx, shots as usize)?;
        let s = u64::from(shots);
        let use_leaf = (mode == SamplingAlgorithm::BlockCdf) as u32;
        // SAFETY: both input precisions match this module's SampleAmplitude; hierarchy
        // layout follows sizes(N), inputs cover N / shots and output has shots entries.
        unsafe {
            stream
                .launch_builder(&self.draw)
                .arg(amps.slice())
                .arg(leaf.slice())
                .arg(flat.slice())
                .arg(&n)
                .arg(targets.slice())
                .arg(&s)
                .arg(&use_leaf)
                .arg(output.slice_mut())
                .launch(config(s))?;
        }
        if profile {
            ctx.synchronize()?;
        }
        let drawn = Instant::now();
        let result = output.to_vec(ctx)?;
        let end = Instant::now();
        Ok((
            result,
            SamplingStats {
                norm_ms: 0.,
                build_ms: (built - start).as_secs_f64() * 1000.,
                draw_ms: (drawn - built).as_secs_f64() * 1000.,
                download_ms: (end - drawn).as_secs_f64() * 1000.,
                workspace_bytes: workspace,
                build_launches: launches,
            },
        ))
    }
}
fn config(n: u64) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(256).max(1) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn workspace_bounds() {
        assert_eq!(
            sampling_workspace_bytes(0, 1, SamplingAlgorithm::Hierarchical).unwrap(),
            24
        );
        assert_eq!(
            sampling_workspace_bytes(22, 1024, SamplingAlgorithm::Hierarchical).unwrap(),
            132112
        );
        assert_eq!(
            sampling_workspace_bytes(22, 65536, SamplingAlgorithm::BlockCdf).unwrap(),
            33686536
        );
        assert_eq!(
            SamplingAlgorithm::Auto.resolve(1 << 18, 64),
            SamplingAlgorithm::Hierarchical
        );
        assert_eq!(
            SamplingAlgorithm::Auto.resolve(1 << 18, 65),
            SamplingAlgorithm::BlockCdf
        );
        assert_eq!(
            SamplingAlgorithm::Auto.resolve(1 << 16, 1),
            SamplingAlgorithm::BlockCdf
        );
        assert!(sampling_workspace_bytes(64, 1, SamplingAlgorithm::Auto).is_err());
        assert_eq!(
            sampling_workspace_bytes(22, 0, SamplingAlgorithm::Reference).unwrap(),
            0
        );
        assert!(
            sampling_workspace_bytes(22, 1, SamplingAlgorithm::Hierarchical).unwrap() < 1 << 18
        );
        assert!(
            sampling_workspace_bytes(22, 1, SamplingAlgorithm::BlockCdf).unwrap() < 9 * (1 << 22)
        );
    }
}

#[cfg(test)]
mod device_tests {
    use super::*;
    use rand::SeedableRng;

    // Fresh raw states allow testing zeros, phase, corruption and exact support,
    // without gate error obscuring the sampling oracle. Run with --test-threads=1
    // when asserting the process-wide transfer counter.
    #[test]
    fn raw_state_sampling_oracle() {
        let Ok(ctx) = CudaContext::new(0) else {
            return;
        };
        macro_rules! check_precision {
            ($scalar:ty, $state:ty, $readout:ty) => {{
                let mut readout = <$readout>::new(&ctx).unwrap();
                for mode in [
                    SamplingAlgorithm::Hierarchical,
                    SamplingAlgorithm::BlockCdf,
                    SamplingAlgorithm::Auto,
                ] {
                    readout.algorithm = mode;
                    for n in [0, 1, 7, 8, 9, 16, 17] {
                        let len = 1usize << n;
                        let mut amps = vec![0. as $scalar; 2 * len];
                        let indices = if len == 1 { vec![0] } else { vec![0, len - 1] };
                        let p = 1. / indices.len() as f64;
                        for &i in &indices {
                            amps[2 * i + 1] = p.sqrt() as $scalar;
                        }
                        let mut state = <$state>::allocate(&ctx, n).unwrap();
                        state.amps.write(&ctx, &amps).unwrap();
                        let before = crate::device_dtoh_bytes();
                        let samples = readout
                            .sample(&mut StdRng::seed_from_u64(555), &state, 4096)
                            .unwrap();
                        assert_eq!(crate::device_dtoh_bytes() - before, 16 + 4096 * 8);
                        assert!(
                            samples.iter().all(|i| indices.contains(&(*i as usize))),
                            "{mode:?} n={n}"
                        );
                        assert_eq!(
                            samples,
                            readout
                                .sample(&mut StdRng::seed_from_u64(555), &state, 4096)
                                .unwrap()
                        );
                        let first = samples.iter().filter(|&&i| i == 0).count() as f64;
                        assert!(
                            (first - 4096. * p).abs() <= 6. * (4096. * p * (1. - p)).sqrt() + 1.
                        );
                        assert!(readout
                            .sample(&mut StdRng::seed_from_u64(0), &state, 0)
                            .unwrap()
                            .is_empty());
                        // Concentrated final bin catches empty leading blocks/plateaus.
                        amps.fill(0. as $scalar);
                        amps[2 * len - 2] = 1. as $scalar;
                        state.amps.write(&ctx, &amps).unwrap();
                        assert!(readout
                            .sample(&mut StdRng::seed_from_u64(4), &state, 257)
                            .unwrap()
                            .iter()
                            .all(|&i| i == len as u64 - 1));
                    }
                    // Property-style independent random complex sparse distributions.
                    for seed in 0..8 {
                        let mut rng = StdRng::seed_from_u64(seed);
                        let mut amps = vec![0. as $scalar; 1024];
                        for i in 0..512 {
                            if i % 3 != 0 {
                                amps[2 * i] = rng.gen::<f64>() as $scalar;
                                amps[2 * i + 1] = (-rng.gen::<f64>()) as $scalar;
                            }
                        }
                        let norm = amps.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
                        for a in &mut amps {
                            *a = (*a as f64 / norm) as $scalar;
                        }
                        let mut state = <$state>::allocate(&ctx, 9).unwrap();
                        state.amps.write(&ctx, &amps).unwrap();
                        let mut oracle_rng = rng.clone();
                        let mut prefix = Vec::new();
                        let mut sum = 0.;
                        for pair in amps.chunks_exact(2) {
                            sum += (pair[0] as f64).powi(2) + (pair[1] as f64).powi(2);
                            prefix.push(sum);
                        }
                        let samples = readout.sample(&mut rng, &state, 100_000).unwrap();
                        for &actual in &samples {
                            let target = oracle_rng.gen::<f64>() * sum;
                            let expected = prefix.partition_point(|&p| p <= target);
                            assert_eq!(actual as usize, expected, "small exact CDF oracle");
                        }
                        let mut hist = vec![0u32; 512];
                        for i in samples {
                            hist[i as usize] += 1;
                        }
                        let total = amps.iter().map(|&v| (v as f64).powi(2)).sum::<f64>();
                        for i in 0..512 {
                            let p = ((amps[2 * i] as f64).powi(2)
                                + (amps[2 * i + 1] as f64).powi(2))
                                / total;
                            if p == 0. {
                                assert_eq!(hist[i], 0);
                            } else {
                                let mean = p * 100_000.;
                                assert!(
                                    (hist[i] as f64 - mean).abs()
                                        <= 6. * (mean * (1. - p)).sqrt() + 2.,
                                    "seed={seed} mode={mode:?} bin={i}"
                                );
                            }
                        }
                    }
                    // Dense uniform support, including a multi-level hierarchy.
                    for n in [9, 17] {
                        let dim = 1usize << n;
                        let mut amps = vec![0. as $scalar; 2 * dim];
                        for i in 0..dim {
                            amps[2 * i + (i % 2)] = (1. / (dim as f64).sqrt()) as $scalar;
                        }
                        let mut state = <$state>::allocate(&ctx, n).unwrap();
                        state.amps.write(&ctx, &amps).unwrap();
                        let samples = readout
                            .sample(&mut StdRng::seed_from_u64(73), &state, 100_000)
                            .unwrap();
                        let mut hist = vec![0u32; 512];
                        for i in samples {
                            hist[i as usize / (dim / 512)] += 1;
                        }
                        let mean = 100_000. / 512.;
                        assert!(
                            hist.iter()
                                .all(|&count| (count as f64 - mean).abs() <= 6. * mean.sqrt() + 2.),
                            "uniform mode={mode:?} n={n}"
                        );
                    }
                    for bad in [0., 2., f64::NAN, f64::INFINITY] {
                        let mut state = <$state>::allocate(&ctx, 1).unwrap();
                        state
                            .amps
                            .write(
                                &ctx,
                                &[bad as $scalar, 0. as $scalar, 0. as $scalar, 0. as $scalar],
                            )
                            .unwrap();
                        for shots in [0, 10] {
                            assert!(readout
                                .sample(&mut StdRng::seed_from_u64(0), &state, shots)
                                .is_err());
                        }
                    }
                }
            }};
        }
        check_precision!(
            f64,
            crate::sv::state::CudaSvState,
            crate::sv::readout::GpuReadout
        );
        check_precision!(
            f32,
            crate::sv::fp32::CudaSvStateF32,
            crate::sv::readout_f32::GpuReadoutF32
        );
    }
    #[test]
    fn exact_inverse_boundaries() {
        let Ok(ctx) = CudaContext::new(0) else {
            return;
        };
        let source = format!(
            "struct SampleAmplitude {{ double re, im; }};\n{}",
            include_str!("sampling.cu")
        );
        let ptx = cudarc::nvrtc::compile_ptx(source).unwrap();
        let module = ctx.raw().load_module(ptx).unwrap();
        let sampler = Sampler::new(&module).unwrap();
        let n = 512u64;
        let mut amplitudes = vec![0f64; 1024];
        for (i, p) in [(1usize, 0.125f64), (255, 0.125), (256, 0.25), (511, 0.5)] {
            amplitudes[2 * i] = p.sqrt();
        }
        let amps = DeviceBuffer::from_slice(&ctx, &amplitudes).unwrap();
        let mut cdf = vec![0.; 512];
        for start in [0, 256] {
            let mut sum = 0.;
            for i in start..start + 256 {
                sum += amplitudes[2 * i].powi(2);
                cdf[i] = sum;
            }
        }
        let leaf = DeviceBuffer::from_slice(&ctx, &cdf).unwrap();
        let flat = DeviceBuffer::from_slice(&ctx, &[cdf[255], cdf[255] + cdf[511]]).unwrap();
        let uniforms = DeviceBuffer::from_slice(
            &ctx,
            &[
                0.,
                0.125,
                f64::from_bits(0.25f64.to_bits() - 1),
                0.25,
                0.5,
                f64::from_bits(1f64.to_bits() - 1),
            ],
        )
        .unwrap();
        let shots = 6u64;
        for use_leaf in [0u32, 1] {
            let mut out = DeviceBuffer::<u64>::zeros(&ctx, 6).unwrap();
            // SAFETY: all arrays match sample_hierarchy's concrete double ABI;
            // N=512 has a two-entry one-level hierarchy and six bounded targets.
            unsafe {
                ctx.stream()
                    .launch_builder(&sampler.draw)
                    .arg(amps.slice())
                    .arg(leaf.slice())
                    .arg(flat.slice())
                    .arg(&n)
                    .arg(uniforms.slice())
                    .arg(&shots)
                    .arg(&use_leaf)
                    .arg(out.slice_mut())
                    .launch(config(shots))
                    .unwrap();
            }
            assert_eq!(out.to_vec(&ctx).unwrap(), [1, 255, 255, 256, 511, 511]);
        }
    }
}
