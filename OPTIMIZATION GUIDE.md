# Optimization Guide

> **The methodology for optimizing this simulator.** Read this before any performance work. The companion document `OPTIMIZATION CYCLE.md` walks through one full iteration; per-algorithm playbooks (at repo root) apply this framework to specific algorithms.

-----

## 1. Philosophy

Optimization is a science, not a craft. Anyone can make code faster by guessing; doing it reliably, reproducibly, and without introducing bugs requires discipline.

Four principles govern this project:

1. **Measure, don’t guess.** Every claim of “faster” requires numbers from `criterion` with a defined baseline. Intuition about performance is wrong more often than right, especially with modern CPUs.
1. **Correctness gates everything.** A 10× speedup that produces results 1e-8 off from the oracle is a bug, not an optimization. Oracle tests (vs. Qiskit / Stim) run before and after every change.
1. **Optimize what matters.** 80% of execution time lives in 20% of the code. Optimizing the wrong 80% is wasted work. Profile first.
1. **One variable at a time.** Bundling three changes makes it impossible to know which gave the speedup (or regression). Each PR isolates one optimization.

The opposite of these — guessing, hoping correctness holds, optimizing whatever’s interesting, mixing changes — is how most performance projects fail.

-----

## 2. The Four Questions

Before any optimization, answer all four. Write them down in the issue.

**Q1: What are we measuring?**
Specific metric: wall-clock time of a specific benchmark on specific hardware. Not “speed of QFT” — `qft_25_qubits_avx2_threadcount_1` on `Ryzen 9 7950X, RUSTFLAGS=-C target-cpu=native`.

**Q2: What’s the current value?**
Baseline number from `cargo bench` on the agreed reference hardware. Recorded with date and commit hash.

**Q3: What’s the target?**
A specific value, not “faster”. Examples:

- “Beat Qiskit Aer single-thread for `qft_20` by 10%.”
- “Within 2× of cuQuantum for `random_circuit_30_depth_20`.”
- “Reduce L3 cache misses by 50%.”

The target must be **derived from an external reference** (a competitor, a hardware ceiling, a published number). Targets out of thin air encourage premature stopping or endless tuning.

**Q4: What’s the hypothesis?**
A statement of cause and effect. “Switching to SoA layout will reduce L1 misses, which will increase apply_1q_gate throughput by ~2× on AVX2-vectorizable kernels.”

If the hypothesis is “this should be faster because it’s more elegant” — stop. That’s guessing.

-----

## 3. The Optimization Hierarchy

Optimizations sorted by **ROI (return on investment)** — speedup per unit effort. Always work from the top.

|Rank|Lever                            |Typical speedup             |Notes                                                                    |
|----|---------------------------------|----------------------------|-------------------------------------------------------------------------|
|1   |Choose the right **algorithm**   |10–1000×                    |Stabilizer vs. SV for Clifford; MPS vs. SV for shallow. Biggest lever.   |
|2   |Pick the right **backend**       |5–100×                      |Automatic backend selection. Backend mismatch = wasted everything else.  |
|3   |**IR-level** optimizations       |2–10×                       |Gate fusion, cancellation, commutation. Cheap to implement, broad impact.|
|4   |**Memory layout** (SoA, blocking)|2–5×                        |SoA, cache-friendly access, alignment.                                   |
|5   |**SIMD** (AVX2 / AVX-512)        |2–4×                        |Hand-written intrinsics on hot kernels.                                  |
|6   |**Multi-threading** (CPU)        |~N cores at ≤80% efficiency |Embarrassingly parallel; bottleneck is memory bandwidth.                 |
|7   |**GPU** acceleration             |10–50× over CPU             |Major investment; ROI depends on workload.                               |
|8   |**Multi-GPU / distributed**      |~N GPUs at 50–80% efficiency|Communication-bound; research-level.                                     |

**Rules of engagement**:

- Don’t skip ranks. SIMD before fusion is wrong — fusion changes what you’d vectorize.
- A win at rank 1 can make all lower ranks irrelevant. Always recheck.
- Each rank’s ceiling is a hard limit unless you go down a rank.

-----

## 4. Performance Ceilings — Know Your Limits

Every optimization runs into a ceiling. Knowing the ceiling tells you when to stop.

### Roofline model (single CPU)

For each kernel, compute:

- **Arithmetic intensity (AI)** = FLOPs / bytes accessed.
- **Peak FLOPs** of the CPU (e.g., 1.5 TFLOP/s on Ryzen 9 7950X for FP64 with AVX2).
- **Peak memory bandwidth** (e.g., 80 GB/s for DDR5-6000 dual-channel).
- **Roofline ceiling** = min(peak FLOPs, AI × peak bandwidth).

State vector kernels are **memory-bound**: AI ≈ 1–4 FLOPs/byte. A `apply_1q_gate` reads 16 bytes (2 complex amplitudes) and does ~8 FLOPs, AI = 0.5. On 80 GB/s memory, ceiling = 40 GFLOP/s. If you’re at 30 GFLOP/s, you have 33% headroom. If you’re at 38 GFLOP/s, **stop** — further work won’t pay back.

### Communication ceilings (multi-GPU / distributed)

Inter-GPU: NVLink at ~600 GB/s, PCIe at ~30 GB/s. Bytes exchanged per “global qubit” gate ≈ state vector size / 2. For 30 qubits = 8 GB, exchange = 4 GB; over PCIe = 130 ms per global gate. That’s your ceiling.

### What this means practically

- Compute the ceiling **before** optimizing. Knowing you’re 40% of ceiling vs. 90% of ceiling changes everything.
- If you’re near ceiling at rank N, drop to rank N+1 (e.g., from CPU-SIMD to GPU).
- Publish ceilings in performance reports. “We hit 85% of memory bandwidth” is a stronger claim than “we’re faster than X.”

-----

## 5. Identifying Bottlenecks

Symptoms → likely causes:

|Symptom                             |Likely bottleneck                    |Tool to confirm                                        |
|------------------------------------|-------------------------------------|-------------------------------------------------------|
|CPU usage <100% on single-thread    |I/O, syscalls, memory stall          |`perf stat` → look at `stalled-cycles-frontend/backend`|
|L3 cache miss rate >5%              |Cache-unfriendly access              |`perf stat -e cache-misses,cache-references`           |
|Branch misprediction >2%            |Data-dependent branches in hot loops |`perf stat -e branch-misses`                           |
|IPC <1.5                            |Pipeline stalls, dependency chains   |`perf stat` → `instructions/cycle`                     |
|Scaling efficiency <70% at N threads|False sharing, lock contention       |`perf c2c` for false sharing                           |
|Multi-GPU efficiency <50%           |Communication-bound                  |NCCL profiling, Nsight Systems                         |
|GPU utilization <80%                |Kernel launch overhead, small kernels|Nsight Compute                                         |

**Workflow**:

1. Run criterion benchmark to get baseline.
1. Run `perf stat` to get high-level counters.
1. Run `cargo flamegraph` to identify hot functions.
1. For hot functions: zoom in with `perf record -e cycles --call-graph dwarf`.
1. Form hypothesis. Test by optimizing or by ad-hoc microbenchmark.

Never skip step 1–3. Optimizing without profile data has roughly 50/50 odds of making things worse.

-----

## 6. Benchmark Discipline

### Anatomy of a good benchmark

A criterion benchmark for this project must:

1. **Be deterministic**: fixed RNG seed for random circuits.
1. **Be reproducible**: documented hardware, RUSTFLAGS, thread count.
1. **Be focused**: one kernel or one algorithm at a time.
1. **Have a baseline**: record results vs. `--baseline main` for diffs.
1. **Include warmup**: criterion handles this by default.
1. **Measure end-to-end and per-component**: macro (full QFT) and micro (apply_h kernel) benchmarks both.

### Hardware reference

The project tracks performance on three reference systems. Every benchmark report mentions which.

- **Workstation**: Ryzen 9 7950X, 64 GB DDR5-6000, no GPU.
- **Server**: 2× Xeon Platinum 8480+, 512 GB DDR5, NVIDIA H100 80GB.
- **Consumer GPU box**: Ryzen 9 7950X + RTX 4090.

If you only have one machine, benchmark on it consistently. Don’t compare results across machines without a calibration run.

### Comparing against external simulators

For Qiskit Aer / Stim / cuQuantum comparisons:

- Run the same circuit (load from the same OpenQASM file).
- Same thread count (1 for single-thread comparisons; max physical cores for multi).
- Same precision (FP64).
- Exclude setup time (compilation, transpilation) — measure only the simulation phase.
- Run ≥10 iterations; report median.

If our number is suspiciously good, double-check: is the comparison fair? Is the work the same? Is there missing functionality? “Too good to be true” usually is.

-----

## 7. Stopping Criteria

The single hardest skill in optimization is knowing when to stop. The default is to keep going forever because you can always find another 5%.

**Stop when any of these is true**:

1. Hit the **target** (Q3 above). Move to the next bottleneck.
1. At **>85% of the roofline ceiling**. Diminishing returns; drop to the next rank.
1. Time invested exceeds 2× the original estimate, with <10% improvement so far. The lever isn’t paying off; reconsider.
1. Optimization requires **>3× code complexity** for **<2× speedup**. Maintenance burden exceeds value.
1. Correctness is becoming hard to validate (e.g., floating-point error accumulating, edge cases multiplying).

When stopping, document:

- What was tried.
- What worked, what didn’t.
- The new baseline.
- The next bottleneck (handoff to the next issue).

-----

## 8. Trade-offs

Every optimization trades one resource for another. Be explicit about which.

|You gain          |You may lose                           |
|------------------|---------------------------------------|
|Speed             |Memory (e.g., precomputed tables)      |
|Speed             |Code clarity (e.g., SIMD intrinsics)   |
|Speed             |Portability (e.g., AVX-512 only)       |
|Speed             |Precision (e.g., FP32 instead of FP64) |
|Memory            |Speed (e.g., recomputing vs. caching)  |
|Generality        |Speed (specialized vs. generic kernels)|
|Numerical accuracy|Speed (e.g., FMA reordering)           |

Document trade-offs in the PR. Reviewers should know what they’re approving.

Forbidden trade-offs in this project:

- **Speed for correctness.** Never. Unitarity / oracle equivalence is non-negotiable.
- **Speed for memory safety.** No raw pointers, no UB. SIMD intrinsics that are technically safe-by-construction must include a `// SAFETY:` block.

-----

## 9. Common Antipatterns

Specific failure modes that have killed performance projects elsewhere.

**1. Premature optimization.** Optimizing code that runs once during setup. ROI = 0.

**2. Optimization without profiling.** “I think this is the bottleneck” — usually wrong. Always profile.

**3. Microbenchmark mismatch.** A kernel benchmark that runs in L1 cache shows speedups invisible at full scale because in reality the data spills to L3 or memory. Always confirm with full-circuit benchmarks.

**4. The “elegant” trap.** Rewriting working code to be prettier without measurable improvement. If the diff is large and the speedup small, revert.

**5. Cargo-cult SIMD.** Reaching for intrinsics before fusion, before SoA, before commutation. Skipping ranks 3-4.

**6. Multi-threading too early.** Adding parallelism to a slow single-thread code amplifies the slowness. Optimize single-thread first.

**7. Benchmark-driven overfitting.** Tuning specifically to the benchmark fixture instead of the general case. A QFT specifically tuned to n=20 may regress at n=22.

**8. Ignoring memory.** Halving runtime by doubling memory is sometimes great, sometimes catastrophic. Track both.

**9. Ignoring tail performance.** Optimizing the median while making p99 worse. Report tail metrics for benchmarks with variance.

**10. “It works on my machine.”** Optimizations sensitive to CPU model, RAM speed, page size. Benchmark on the reference hardware.

-----

## 10. Checklists

### Before opening a PR

- [ ] Issue is selected from `BACKLOG.md` and assigned.
- [ ] Four Questions answered in the issue or PR description.
- [ ] Roofline ceiling computed (where relevant).
- [ ] Baseline measured and recorded (commit hash + machine).
- [ ] Hypothesis stated.

### Before merging an optimization PR

- [ ] All oracle tests pass (vs. Qiskit / Stim).
- [ ] All property tests pass.
- [ ] Criterion benchmark numbers: before/after, same machine, same RUSTFLAGS.
- [ ] Improvement is ≥5% on the target metric (smaller = noise).
- [ ] No regression >2% on any other benchmark in the suite.
- [ ] Code review at least self-review with fresh eyes.
- [ ] Documentation updated if APIs changed.
- [ ] If accepting trade-off: documented in PR.

### When investigating a regression

- [ ] Bisect to the introducing commit (`git bisect`).
- [ ] Reproduce on the reference machine.
- [ ] Profile to identify the new bottleneck.
- [ ] Decide: revert, fix-forward, or accept (with justification).

-----

## 11. Tooling Quick Reference

### Profiling

```bash
# Flamegraph from a benchmark
cargo flamegraph --bench qft -- --bench

# perf stat on a benchmark binary
cargo bench --bench qft --no-run                       # compile only
perf stat -e cycles,instructions,cache-misses,cache-references,branch-misses \
  ./target/release/deps/qft-<hash> --bench

# perf record + report for hot functions
perf record -e cycles --call-graph dwarf ./target/release/deps/qft-<hash> --bench
perf report

# Cache analysis
perf stat -e L1-dcache-load-misses,L1-dcache-loads,LLC-load-misses,LLC-loads \
  ./target/release/deps/qft-<hash> --bench

# False sharing detection
perf c2c record ./target/release/deps/...
perf c2c report
```

### Benchmarking

```bash
# Standard benchmark run
cargo bench --workspace

# Specific benchmark
cargo bench --bench qft

# With baseline comparison
cargo bench --bench qft -- --save-baseline main
# (make changes, then)
cargo bench --bench qft -- --baseline main

# With optimal compile flags
RUSTFLAGS="-C target-cpu=native -C opt-level=3 -C lto=fat" cargo bench

# Disable CPU frequency scaling (Linux) for stable numbers
sudo cpupower frequency-set -g performance
# (run benchmarks)
sudo cpupower frequency-set -g powersave
```

### Comparison vs. external simulators

```bash
# Qiskit Aer (Python)
python scripts/bench_qiskit.py --circuit qft_20.qasm --threads 1

# Stim (Python)
python scripts/bench_stim.py --circuit surface_d3.stim

# Our simulator
./target/release/aleph bench qft_20.qasm --threads 1
```

Reference scripts live under `scripts/bench_external/`.

-----

## 12. Reporting Format

Every optimization PR includes a “Benchmark Results” section with this format:

```markdown
### Benchmark Results

**Machine**: Ryzen 9 7950X, 64 GB DDR5-6000, Rust 1.75
**Flags**: `RUSTFLAGS="-C target-cpu=native"`
**Date**: 2025-XX-XX
**Baseline**: commit abc1234

| Benchmark               | Baseline (ms) | This PR (ms) | Speedup | Notes |
|-------------------------|---------------|--------------|---------|-------|
| qft_20_singlethread     | 142.3 ± 1.2   | 87.4 ± 0.9   | 1.63×   | Target |
| qft_25_singlethread     | 5210 ± 18     | 3180 ± 15    | 1.64×   | Generalizes |
| grover_16_singlethread  | 89.1 ± 0.7    | 88.4 ± 0.8   | 1.01×   | No regression |
| ghz_20_singlethread     | 12.4 ± 0.2    | 12.5 ± 0.2   | 0.99×   | Within noise |

**Roofline analysis**: kernel achieves 32 GFLOP/s = 80% of peak memory bandwidth (40 GFLOP/s ceiling for FP64 on this hardware).

**Hypothesis confirmed**: SoA layout reduced L1 misses from 8% to 1.2%; expected speedup ~1.5×, observed 1.63×.
```

Every phase exit produces a consolidated report under `docs/perf/phase{n}.md` aggregating all PRs from that phase.

-----

## 13. The Per-Algorithm Playbooks

Each algorithm has unique characteristics that drive specific optimization opportunities. The playbooks at the repo root apply this framework algorithm by algorithm:

- `QFT.md` — Quantum Fourier Transform. Diagonal-gate heavy; phase-precomputation wins.
- `GROVER.md` — Grover’s algorithm. Oracle + diffusion repeated; loop-unroll potential.
- `VQE.md` — Variational Quantum Eigensolver. Many short circuits with shared structure; expectation values dominate.
- `QAOA.md` — Quantum Approximate Optimization. Layered structure; MPS-friendly.
- `RANDOM CIRCUIT.md` — Sycamore-style random. Worst-case for state vector; tests every kernel.
- `STABILIZER CIRCUITS.md` — Clifford circuits (surface code). Wrong backend = wrong universe.

Read the relevant playbook before working on an issue that targets a specific algorithm.

-----

## 14. Further Reading

- Hennessy & Patterson, *Computer Architecture: A Quantitative Approach*, 6th ed. — roofline, memory hierarchy.
- Drepper, *What Every Programmer Should Know About Memory* — <https://lwn.net/Articles/250967/>
- Agner Fog’s optimization manuals — <https://www.agner.org/optimize/>
- Williams, Waterman, Patterson, *Roofline: An Insightful Visual Performance Model* (2009).
- Hager & Wellein, *Introduction to High Performance Computing for Scientists and Engineers* (2010).