//! Budgeted trajectory chunks. The shared Rayon pool may also run kernel work;
//! a chunk owns its state across suspension, so chunks (not active OS threads)
//! are the memory/concurrency bound.

use super::{apply, Counts, NoiseError, NoiseModel, QuantumError};
use aleph_backend::Backend;
use aleph_core::{Complex, Gate};
use aleph_ir::{Circuit, Instruction};
use rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;
use std::mem::size_of;

/// Ascending `(basis_state, count)` pairs, containing only observed outcomes.
pub type SparseCounts = Vec<(u64, u64)>;

/// Incremental heap-buffer policy for one noisy call.
///
/// The default is 1 GiB with at most the current Rayon pool's thread count
/// of trajectory chunks. Caller-owned circuit/model, allocator bookkeeping,
/// stacks, runtime task allocations and other concurrent calls are excluded.
/// This is an allocation plan, not a reservation of OS memory or an RSS cap.
#[derive(Clone, Copy, Debug)]
pub struct NoiseOptions {
    /// Maximum planned incremental buffer bytes; checked before allocating.
    pub memory_budget_bytes: usize,
    /// Optional upper bound on simultaneously owned trajectory contexts.
    /// Zero is invalid. Does not create or resize the shared Rayon pool.
    pub max_workers: Option<usize>,
}
impl Default for NoiseOptions {
    fn default() -> Self {
        Self {
            memory_budget_bytes: 1 << 30,
            max_workers: None,
        }
    }
}

/// Explainable upper bound on owned data buffers (not allocator/runtime overhead).
#[derive(Clone, Copy, Debug)]
pub struct NoisePlan {
    /// Maximum number of trajectory contexts alive, including suspended tasks.
    pub workers: usize,
    /// Outcomes, result and resolved attachment storage.
    pub fixed_bytes: usize,
    /// State, Kraus probabilities, sample, validated matrix snapshots and kernel scratch.
    pub per_worker_bytes: usize,
    /// `fixed_bytes + workers * per_worker_bytes`.
    pub peak_buffer_bytes: usize,
}
fn overflow() -> NoiseError {
    NoiseError::Configuration {
        reason: "buffer size exceeds platform allocation limits",
    }
}
fn add(a: usize, b: usize) -> Result<usize, NoiseError> {
    a.checked_add(b)
        .filter(|v| *v <= isize::MAX as usize)
        .ok_or_else(overflow)
}
fn mul(a: usize, b: usize) -> Result<usize, NoiseError> {
    a.checked_mul(b)
        .filter(|v| *v <= isize::MAX as usize)
        .ok_or_else(overflow)
}
fn dimension(circuit: &Circuit) -> Result<usize, NoiseError> {
    let n = circuit.num_qubits();
    if n > crate::backend::MAX_NAIVE_QUBITS {
        return Err(aleph_backend::BackendError::TooManyQubits {
            requested: n,
            limit: crate::backend::MAX_NAIVE_QUBITS,
        }
        .into());
    }
    1usize.checked_shl(n).ok_or_else(overflow)
}

/// Plan dense (`sparse = false`) or sparse counts without allocating trajectory
/// or output buffers. Use in the same Rayon pool that will execute the run.
/// Zero-shot plans need only the requested result and no trajectory context.
pub fn plan_noisy(
    circuit: &Circuit,
    model: &NoiseModel,
    shots: u32,
    options: NoiseOptions,
    sparse: bool,
) -> Result<NoisePlan, NoiseError> {
    if options.max_workers == Some(0) {
        return Err(NoiseError::Configuration {
            reason: "max_workers must be positive",
        });
    }
    let dim = dimension(circuit)?;
    let output = if sparse {
        mul((shots as usize).min(dim), 16)?
    } else {
        mul(dim, 8)?
    };
    let mut fixed = add(mul(shots as usize, 8)?, output)?;
    if shots == 0 {
        return fit(fixed, 0, 0, options.memory_budget_bytes);
    }
    fixed = add(
        fixed,
        mul(
            circuit.instructions().len(),
            size_of::<Vec<&QuantumError>>(),
        )?,
    )?;
    let mut branches = 0;
    let mut kernel_scratch = 0;
    let mut matrix_cache_bytes = 0;
    for inst in circuit.instructions() {
        if let Instruction::Gate(gi) = inst {
            let mut count = 0;
            for error in model.errors_iter(gi.gate.name(), &gi.qubits) {
                count += 1;
                if let QuantumError::Kraus(channel) = error {
                    branches = branches.max(channel.kraus.len());
                }
            }
            fixed = add(fixed, mul(count, size_of::<&QuantumError>())?)?;
            // apply_gate's u32 seen list and kernels' (u32, bool) fixed
            // lists can spill simultaneously. Allow rounded-up capacities.
            let args = add(gi.qubits.len(), gi.controls.len())?;
            let seen = if args > 6 { mul(args, 8)? } else { 0 };
            let fixed_len = add(gi.controls.len(), 3)?;
            let fixed_spill = if fixed_len > 8 {
                mul(fixed_len, 16)?
            } else {
                0
            };
            let scratch = if let Gate::UnitaryKq { k, .. } = gi.gate {
                let width = 1usize.checked_shl(k as u32).ok_or_else(overflow)?;
                // Replacing the backend's single validated snapshot briefly owns
                // both matrices. Budget two maximum matrices per context.
                matrix_cache_bytes = matrix_cache_bytes.max(mul(mul(width, width)?, 32)?);
                // AVX512 gathers eight blocks; scalar uses one. Kernel tasks
                // share the Rayon pool but can suspend, so conservatively allow
                // eight blocks per pool thread per trajectory, plus offsets.
                // AVX512 also broadcasts TWO full matrices into 64-byte
                // vectors. Include them even on scalar-only machines.
                add(
                    mul(mul(width, width)?, 128)?,
                    add(
                        mul(mul(width, 128)?, rayon::current_num_threads())?,
                        add(mul(width, 8)?, mul(k as usize, 16)?)?,
                    )?,
                )?
            } else {
                fixed_spill
            };
            kernel_scratch = kernel_scratch.max(add(seen, scratch)?);
        }
    }
    let per_worker = add(
        mul(dim, size_of::<Complex>())?,
        add(
            mul(branches, 8)?,
            add(8, add(kernel_scratch, matrix_cache_bytes)?)?,
        )?,
    )?;
    let max_workers = options
        .max_workers
        .unwrap_or(rayon::current_num_threads())
        .min(rayon::current_num_threads())
        .min(shots as usize);
    fit(fixed, per_worker, max_workers, options.memory_budget_bytes)
}
fn fit(
    fixed: usize,
    per_worker: usize,
    maximum: usize,
    budget: usize,
) -> Result<NoisePlan, NoiseError> {
    let required = add(fixed, if maximum == 0 { 0 } else { per_worker })?;
    if required > budget {
        return Err(NoiseError::MemoryBudget { required, budget });
    }
    let workers = if maximum == 0 {
        0
    } else {
        maximum.min((budget - fixed) / per_worker)
    };
    Ok(NoisePlan {
        workers,
        fixed_bytes: fixed,
        per_worker_bytes: per_worker,
        peak_buffer_bytes: add(fixed, mul(workers, per_worker)?)?,
    })
}

// Keep the allocation-free noisy preflight consistent with all CPU SV backends.
fn validate_fused_gate(gi: &aleph_core::GateInstance) -> Result<(), NoiseError> {
    if let Gate::UnitaryKq { k, data } = &gi.gate {
        crate::validation::validate_unitary_kq(*k, data, &gi.controls).map_err(|error| {
            // Preserve the noisy API's existing fused-input error categories.
            match error {
                aleph_backend::BackendError::InvalidState { reason } => {
                    NoiseError::Configuration { reason }
                }
                aleph_backend::BackendError::UnsupportedGate { kind } => {
                    NoiseError::Unsupported { kind }
                }
                other => NoiseError::Backend(other),
            }
        })?;
    }
    Ok(())
}

// Validate public-field channel inputs before scratch allocation/indexing.
// Gate/unitarity validation stays in apply_gate on every trajectory.
fn validate_channel(error: &QuantumError, arity: usize) -> Result<(), NoiseError> {
    let bad = |reason| NoiseError::InvalidChannel { reason };
    if error.arity() != arity {
        return Err(bad("attachment arity does not match gate"));
    }
    match error {
        QuantumError::Pauli(c) => {
            let mut total = 0.0;
            for (p, term) in &c.terms {
                if !p.is_finite() || *p < 0.0 || term.len() != arity {
                    return Err(bad("invalid Pauli probability or term arity"));
                }
                total += p;
            }
            if !total.is_finite() || (total - 1.0).abs() > 1e-9 {
                return Err(bad("Pauli probabilities must sum to one"));
            }
        }
        QuantumError::Kraus(c) => {
            let mut completeness = [[Complex::new(0.0, 0.0); 2]; 2];
            for k in &c.kraus {
                if k.iter()
                    .flatten()
                    .any(|z| !z.re.is_finite() || !z.im.is_finite())
                {
                    return Err(bad("non-finite Kraus matrix"));
                }
                for (i, row) in completeness.iter_mut().enumerate() {
                    for (j, sum) in row.iter_mut().enumerate() {
                        for krow in k {
                            *sum += krow[i].conj() * krow[j];
                        }
                    }
                }
            }
            for (i, row) in completeness.iter().enumerate() {
                for (j, sum) in row.iter().enumerate() {
                    let deviation =
                        (*sum - Complex::new(if i == j { 1.0 } else { 0.0 }, 0.0)).norm();
                    if !deviation.is_finite() || deviation > 1e-9 {
                        return Err(bad("Kraus operators must be trace preserving"));
                    }
                }
            }
        }
    }
    Ok(())
}
fn outcomes(
    circuit: &Circuit,
    model: &NoiseModel,
    shots: u32,
    seed: u64,
    plan: NoisePlan,
) -> Result<Vec<u64>, NoiseError> {
    if shots == 0 {
        return Ok(Vec::new());
    }
    // Validation and size planning precede all trajectory/result allocations.
    for inst in circuit.instructions() {
        match inst {
            Instruction::Gate(gi) => {
                validate_fused_gate(gi)?;
                for error in model.errors_iter(gi.gate.name(), &gi.qubits) {
                    validate_channel(error, gi.qubits.len())?;
                }
            }
            Instruction::Measure { .. } => return Err(NoiseError::MidCircuit { kind: "measure" }),
            Instruction::Reset(_) => return Err(NoiseError::MidCircuit { kind: "reset" }),
            Instruction::TiledBlock(_) => {
                return Err(NoiseError::Unsupported {
                    kind: "tiled-block",
                })
            }
            _ => {}
        }
    }
    for (&q, ro) in model.readout_map() {
        if q >= circuit.num_qubits()
            || ro.m.iter().any(|row| {
                row.iter().any(|p| !p.is_finite() || *p < 0.0)
                    || (row[0] + row[1] - 1.0).abs() > 1e-9
            })
        {
            return Err(NoiseError::InvalidChannel {
                reason: "invalid readout probability or qubit",
            });
        }
    }
    let resolved: Vec<Vec<&QuantumError>> = circuit
        .instructions()
        .iter()
        .map(|inst| match inst {
            Instruction::Gate(gi) => {
                let count = model.errors_iter(gi.gate.name(), &gi.qubits).count();
                let mut errors = Vec::with_capacity(count);
                errors.extend(model.errors_iter(gi.gate.name(), &gi.qubits));
                errors
            }
            _ => Vec::new(),
        })
        .collect();
    let max_branches = resolved
        .iter()
        .flatten()
        .filter_map(|error| match error {
            QuantumError::Kraus(c) => Some(c.kraus.len()),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    let mut outcomes = vec![0; shots as usize];
    let chunk_len = (shots as usize).div_ceil(plan.workers);
    outcomes
        .par_chunks_mut(chunk_len)
        .enumerate()
        .try_for_each(|(chunk, output)| {
            let mut backend = crate::NaiveSvBackend::with_seed(0);
            let mut state = backend.allocate(circuit.num_qubits())?;
            let mut scratch = Vec::with_capacity(max_branches);
            for (offset, outcome) in output.iter_mut().enumerate() {
                // Absolute index makes the RNG and readout independent of budget,
                // chunk length, scheduling and the state left by the previous shot.
                backend.rng = StdRng::seed_from_u64(apply::shot_seed(
                    seed,
                    (chunk * chunk_len + offset) as u64,
                ));
                if offset != 0 {
                    state.amps.fill(Complex::new(0.0, 0.0));
                    state.amps[0] = Complex::new(1.0, 0.0);
                }
                *outcome = super::run_trajectory(
                    circuit,
                    &resolved,
                    model.readout_map(),
                    &mut backend,
                    &mut state,
                    &mut scratch,
                )?;
            }
            Ok::<_, NoiseError>(())
        })?;
    Ok(outcomes)
}

/// Dense compatibility API with explicit incremental buffer budget.
pub fn run_noisy_with_options(
    circuit: &Circuit,
    model: &NoiseModel,
    shots: u32,
    seed: u64,
    options: NoiseOptions,
) -> Result<Counts, NoiseError> {
    let plan = plan_noisy(circuit, model, shots, options, false)?;
    let outcomes = outcomes(circuit, model, shots, seed, plan)?;
    let mut hist = vec![0; dimension(circuit)?];
    for index in outcomes {
        hist[index as usize] += 1;
    }
    Ok(hist)
}

/// Sparse counts under an explicit buffer budget, sorted by basis index.
/// Avoids the dense `2^n` result; worst-case output holds `min(shots, 2^n)` pairs.
/// Planning reserves that worst case even if the actual distribution is sparse.
pub fn run_noisy_sparse(
    circuit: &Circuit,
    model: &NoiseModel,
    shots: u32,
    seed: u64,
    options: NoiseOptions,
) -> Result<SparseCounts, NoiseError> {
    let plan = plan_noisy(circuit, model, shots, options, true)?;
    let mut outcomes = outcomes(circuit, model, shots, seed, plan)?;
    outcomes.sort_unstable(); // In-place, no heap scratch.
    let mut counts: SparseCounts = Vec::with_capacity((shots as usize).min(dimension(circuit)?));
    for index in outcomes {
        if let Some((_, count)) = counts.last_mut().filter(|(last, _)| *last == index) {
            *count += 1;
        } else {
            counts.push((index, 1));
        }
    }
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::noise::{
        amplitude_damping_error, depolarizing_error, phase_damping_error, ReadoutError,
    };
    use aleph_parser::parse;

    // Independent driver reference: exactly the previous fresh-per-shot path,
    // including fresh channel scratch. Does not call the reusable driver.
    fn fresh_reference(circuit: &Circuit, model: &NoiseModel, shots: u32, seed: u64) -> Counts {
        let n = circuit.num_qubits();
        let mut hist = vec![0; 1 << n];
        for shot in 0..shots {
            let mut backend = crate::NaiveSvBackend::with_seed(apply::shot_seed(seed, shot as u64));
            let mut state = backend.allocate(n).unwrap();
            for inst in circuit.instructions() {
                match inst {
                    Instruction::Gate(gi) => {
                        backend.apply_gate(&mut state, gi).unwrap();
                        for error in model.errors_for(gi.gate.name(), &gi.qubits) {
                            apply::apply_channel(
                                &mut state.amps,
                                n,
                                error,
                                &gi.qubits,
                                &mut backend.rng,
                            );
                        }
                    }
                    Instruction::DiagonalPhase(dp) => {
                        backend.apply_diagonal_phase(&mut state, dp).unwrap()
                    }
                    Instruction::Barrier(_) => {}
                    _ => panic!("reference uses supported gate circuits"),
                }
            }
            let index = backend.sample(&state, 1).unwrap()[0];
            let index = apply::apply_readout(index, n, model.readout_map(), &mut backend.rng);
            hist[index as usize] += 1;
        }
        hist
    }
    fn mixed_model() -> NoiseModel {
        let mut model = NoiseModel::new();
        model.add_all_qubit_quantum_error(depolarizing_error(0.17, 1), &["H"]);
        // Alternate scratch lengths (4 -> 2 -> 4) to catch stale probabilities.
        let identity = [
            [Complex::new(0.5, 0.0), Complex::new(0.0, 0.0)],
            [Complex::new(0.0, 0.0), Complex::new(0.5, 0.0)],
        ];
        let four = QuantumError::Kraus(super::super::KrausChannel {
            kraus: vec![identity; 4],
        });
        model.add_all_qubit_quantum_error(four.clone(), &["Ry"]);
        model.add_all_qubit_quantum_error(amplitude_damping_error(0.31), &["Ry"]);
        model.add_all_qubit_quantum_error(four, &["Ry"]);
        model.add_quantum_error(phase_damping_error(0.27), &["H"], &[1]);
        model.add_all_qubit_quantum_error(depolarizing_error(0.12, 2), &["Cnot"]);
        model.add_readout_error(ReadoutError::new([[0.93, 0.07], [0.21, 0.79]]), 0);
        model
    }
    #[test]
    fn reused_chunks_match_fresh_seeded_reference() {
        let circuit = parse("OPENQASM 3.0; include \"stdgates.inc\"; qubit[3] q; h q[0]; h q[1]; cx q[0], q[2]; ry(1.13) q[1]; barrier q; h q[2];").unwrap();
        for model in [NoiseModel::new(), mixed_model()] {
            for seed in [0, 7, u64::MAX] {
                let expected = fresh_reference(&circuit, &model, 257, seed);
                for threads in [1, 2, 4] {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .unwrap();
                    pool.install(|| {
                        for limit in [1, 2, 3, 4] {
                            let mut options = NoiseOptions {
                                max_workers: Some(limit),
                                ..Default::default()
                            };
                            let plan = plan_noisy(&circuit, &model, 257, options, false).unwrap();
                            // Exact one-worker budget exercises memory-based throttling
                            // even while more shared pool threads execute kernel work.
                            for budget in [plan.fixed_bytes + plan.per_worker_bytes, 1 << 20] {
                                options.memory_budget_bytes = budget;
                                let actual =
                                    run_noisy_with_options(&circuit, &model, 257, seed, options)
                                        .unwrap();
                                assert_eq!(
                                    actual, expected,
                                    "seed {seed}, threads {threads}, limit {limit}"
                                );
                                let sparse_plan = plan_noisy(
                                    &circuit,
                                    &model,
                                    257,
                                    NoiseOptions::default(),
                                    true,
                                )
                                .unwrap();
                                let sparse_options = NoiseOptions {
                                    memory_budget_bytes: budget.max(
                                        sparse_plan.fixed_bytes + sparse_plan.per_worker_bytes,
                                    ),
                                    ..options
                                };
                                let sparse =
                                    run_noisy_sparse(&circuit, &model, 257, seed, sparse_options)
                                        .unwrap();
                                let mut dense = vec![0; 8];
                                for (index, count) in sparse {
                                    dense[index as usize] = count;
                                }
                                assert_eq!(dense, expected);
                            }
                        }
                    });
                }
            }
        }
    }
    #[test]
    fn zero_shots_and_budget_boundaries() {
        let circuit = Circuit::new(20, 0);
        let model = NoiseModel::new();
        let options = NoiseOptions {
            max_workers: Some(4),
            ..Default::default()
        };
        let plan = plan_noisy(&circuit, &model, 19, options, false).unwrap();
        let one = plan.fixed_bytes + plan.per_worker_bytes;
        let exact = NoiseOptions {
            memory_budget_bytes: one,
            ..options
        };
        assert_eq!(
            plan_noisy(&circuit, &model, 19, exact, false)
                .unwrap()
                .workers,
            1
        );
        assert!(matches!(
            plan_noisy(
                &circuit,
                &model,
                19,
                NoiseOptions {
                    memory_budget_bytes: one - 1,
                    ..options
                },
                false
            ),
            Err(NoiseError::MemoryBudget { .. })
        ));
        let zero = NoiseOptions {
            memory_budget_bytes: 0,
            ..options
        };
        assert!(run_noisy_sparse(&circuit, &model, 0, 7, zero)
            .unwrap()
            .is_empty());
        assert!(matches!(
            run_noisy_with_options(&circuit, &model, 0, 7, zero),
            Err(NoiseError::MemoryBudget { .. })
        ));
        let circuit = Circuit::new(0, 0);
        assert_eq!(
            super::super::run_noisy(&circuit, &model, 3, 7).unwrap(),
            vec![3]
        );
        assert!(matches!(
            plan_noisy(
                &circuit,
                &model,
                1,
                NoiseOptions {
                    max_workers: Some(0),
                    ..options
                },
                false
            ),
            Err(NoiseError::Configuration { .. })
        ));
        assert!(dimension(&Circuit::new(usize::BITS, 0)).is_err());
        assert!(mul(usize::MAX, 16).is_err());
        assert!(add(isize::MAX as usize, 1).is_err());
    }
    #[test]
    fn large_state_budget_can_be_raised_without_allocating_the_state() {
        let model = NoiseModel::new();
        for n in [26, 27, 28] {
            let circuit = Circuit::new(n, 0);
            for sparse in [false, true] {
                assert!(matches!(
                    plan_noisy(&circuit, &model, 1, NoiseOptions::default(), sparse),
                    Err(NoiseError::MemoryBudget { .. })
                ));
                let options = NoiseOptions {
                    memory_budget_bytes: 8usize << 30,
                    max_workers: Some(1),
                };
                assert_eq!(
                    plan_noisy(&circuit, &model, 1, options, sparse)
                        .unwrap()
                        .workers,
                    1
                );
            }
        }
    }
    #[test]
    fn sparse_result_avoids_dense_output_limit() {
        let circuit = Circuit::new(20, 0);
        let model = NoiseModel::new();
        let sparse = plan_noisy(&circuit, &model, 3, NoiseOptions::default(), true).unwrap();
        let options = NoiseOptions {
            memory_budget_bytes: sparse.fixed_bytes + sparse.per_worker_bytes,
            max_workers: Some(1),
        };
        assert!(plan_noisy(&circuit, &model, 3, options, false).is_err());
        assert_eq!(
            run_noisy_sparse(&circuit, &model, 3, 9, options).unwrap(),
            vec![(0, 3)]
        );
    }
    #[test]
    fn fused_matrix_inputs_cannot_bypass_validation() {
        use aleph_core::GateInstance;
        let mut identity = vec![Complex::new(0.0, 0.0); 16];
        for i in 0..4 {
            identity[i * 4 + i] = Complex::new(1.0, 0.0);
        }
        let execute = |k, data: Vec<Complex>, controlled| {
            let mut circuit = Circuit::new(3, 0);
            let gate = Gate::UnitaryKq {
                k,
                data: data.into_boxed_slice(),
            };
            let qubits: Vec<u32> = (0..k as u32).collect();
            let gi = if controlled {
                GateInstance::controlled(gate, &qubits[..], &[2][..])
            } else {
                GateInstance::new(gate, &qubits[..])
            };
            circuit.add_gate(gi).unwrap();
            run_noisy_sparse(
                &circuit,
                &NoiseModel::new(),
                3,
                557,
                NoiseOptions::default(),
            )
        };
        assert_eq!(execute(2, identity.clone(), false).unwrap(), vec![(0, 3)]);
        let mut x_first = vec![Complex::new(0.0, 0.0); 16];
        for col in 0..4 {
            x_first[(col ^ 2) * 4 + col] = Complex::new(1.0, 0.0);
        }
        assert_eq!(execute(2, x_first.clone(), false).unwrap(), vec![(1, 3)]);
        assert!(execute(2, vec![Complex::new(1e308, 0.0); 16], false).is_err());
        assert!(execute(2, vec![], false).is_err());
        assert!(execute(1, vec![Complex::new(1.0, 0.0); 4], false).is_err());
        assert!(execute(2, vec![Complex::new(0.0, 0.0); 16], false).is_err());
        let mut nonfinite = identity.clone();
        nonfinite[0].re = f64::NAN;
        assert!(execute(2, nonfinite, false).is_err());
        let mut reversed = Circuit::new(2, 0);
        reversed
            .add_gate(GateInstance::new(
                Gate::UnitaryKq {
                    k: 2,
                    data: x_first.into_boxed_slice(),
                },
                &[1, 0][..],
            ))
            .unwrap();
        assert_eq!(
            run_noisy_sparse(
                &reversed,
                &NoiseModel::new(),
                1,
                557,
                NoiseOptions::default()
            )
            .unwrap(),
            vec![(2, 1)]
        );
        assert!(matches!(
            execute(2, identity, true),
            Err(NoiseError::Unsupported { .. })
        ));
    }
    #[test]
    fn plan_includes_vectorized_matrix_and_control_spills() {
        use aleph_core::GateInstance;
        let mut circuit = Circuit::new(12, 0);
        let mut data = vec![Complex::new(0.0, 0.0); 16];
        for i in 0..4 {
            data[i * 4 + i] = Complex::new(1.0, 0.0);
        }
        circuit
            .add_gate(GateInstance::new(
                Gate::UnitaryKq {
                    k: 2,
                    data: data.into_boxed_slice(),
                },
                &[0, 1][..],
            ))
            .unwrap();
        let plan = plan_noisy(
            &circuit,
            &NoiseModel::new(),
            1,
            NoiseOptions::default(),
            false,
        )
        .unwrap();
        assert!(
            plan.per_worker_bytes
                >= (1 << 12) * 16 + 128 * 16 + 128 * 4 * rayon::current_num_threads()
        );
        let mut circuit = Circuit::new(12, 0);
        circuit
            .add_gate(GateInstance::controlled(
                Gate::X,
                &[0][..],
                &(1..9).collect::<Vec<_>>()[..],
            ))
            .unwrap();
        let plan = plan_noisy(
            &circuit,
            &NoiseModel::new(),
            1,
            NoiseOptions::default(),
            false,
        )
        .unwrap();
        assert!(plan.per_worker_bytes >= (1 << 12) * 16 + 9 * 8 + 11 * 16);
    }

    #[test]
    fn malformed_channels_rejected_before_trajectory() {
        let circuit = parse("OPENQASM 3.0; include \"stdgates.inc\"; qubit[1] q; h q[0];").unwrap();
        let mut model = NoiseModel::new();
        model.add_all_qubit_quantum_error(
            QuantumError::Kraus(super::super::KrausChannel { kraus: vec![] }),
            &["H"],
        );
        assert!(matches!(
            super::super::run_noisy(&circuit, &model, 1, 0),
            Err(NoiseError::InvalidChannel { .. })
        ));
        let mut model = NoiseModel::new();
        model.add_all_qubit_quantum_error(depolarizing_error(0.1, 2), &["H"]);
        assert!(matches!(
            super::super::run_noisy(&circuit, &model, 1, 0),
            Err(NoiseError::InvalidChannel { .. })
        ));
        let mut model = NoiseModel::new();
        model.add_readout_error(
            ReadoutError {
                m: [[f64::NAN, 0.0], [0.0, 1.0]],
            },
            0,
        );
        assert!(matches!(
            super::super::run_noisy(&circuit, &model, 1, 0),
            Err(NoiseError::InvalidChannel { .. })
        ));
    }
}
