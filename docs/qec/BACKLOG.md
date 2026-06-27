# QEC Decoder Track — Detailed Backlog

> **Source of truth for the QEC decoder track issues.**
> North Star and strategy: `docs/qec/ROADMAP.md`. Mainline simulator backlog: `../../BACKLOG.md`.
> Issue ID convention: `Q{phase}-{nn}` (e.g. `Q0-01`). Each issue uses the same template as
> the mainline `BACKLOG.md` so it can be turned into a GitHub Issue directly.

-----

## How to Read This Document

```
### [Q{phase}-{nn}] {Title}

**Labels:** `area:*`, `type:*`, `priority:*`
**Milestone:** Phase Q{n}
**Estimate:** S / M / L / XL  (S ≈ <1 day, M ≈ 1–3 days, L ≈ 3–7 days, XL ≈ >1 week)
**Depends on:** Q{phase}-{nn}, ...

**Description** — short summary.
**Context** — why this matters.
**Technical Details** — implementation guidance, algorithms, references.
**Acceptance Criteria** — testable bullet points.
**Testing Requirements** — unit, property, integration, benchmark.
**References** — papers, implementations.
```

## New Labels (extend the mainline label system)

- **Area**: `area:qec` (QEC circuits, codes, noise, DEM), `area:decoder` (matching/UF/BP),
  `area:fpga`, `area:asic`
- Reuse existing: `area:backend-stab`, `area:backend-gpu`, `area:bench`, `area:docs`.

## Milestones

- Phase Q0 — Experiment Loop Foundation
- Phase Q1 — MWPM Decoder
- Phase Q2 — Union-Find Decoder
- Phase Q3 — GPU Decoder
- Phase Q4 — Real-Time / Streaming
- Phase Q5 — qLDPC Frontier
- Phase Q6 — FPGA
- Phase Q7 — ASIC (North Star)

## Track-level Golden Rules (in addition to CLAUDE.md)

1. **A decoder is only as good as its experiment loop.** Every decoder claim must be backed
   by a logical-error-rate Monte-Carlo, not a single-shot demo.
2. **Cross-check every DEM and every decode against Stim + PyMatching** before reporting
   numbers. Same noise model, same DEM, same shots.
3. **Report latency *and* accuracy.** A faster decoder that raises logical error rate is a
   different trade-off, not a win — say which.
4. **Co-design for hardware from Q1.** Prefer data structures and control flow that map to
   FPGA/ASIC (bounded memory, no dynamic allocation in the hot loop, integer arithmetic).

-----

# Phase Q0 — Experiment Loop Foundation

Goal: close the loop `noise → syndrome → (no decoder yet) → logical error rate` and
reproduce the surface-code threshold. This is the instrument that gates every later phase.

**Exit metric:** reproduce surface-code memory threshold p_th ≈ 0.5–1.0% (rotated code,
circuit-level or phenomenological noise), with logical error rate curves for d ∈ {3,5,7,9}
crossing at a single point, cross-checked against a Stim-generated DEM on the same circuit.

-----

### [Q0-01] Create `aleph-qec` crate + core QEC types

**Labels:** `area:qec`, `type:feature`, `priority:critical`
**Milestone:** Phase Q0
**Estimate:** M
**Depends on:** —

**Description**
Stand up a new workspace crate `crates/aleph-qec` that owns codes, noise, syndromes, the
Detector Error Model (DEM), and the decoder trait. No decoder implementation yet.

**Context**
Decoders, DEMs, and experiment harnesses do not belong in `aleph-stab` (a backend) or
`benches` (test fixtures). They need a home that depends on `aleph-stab` / `aleph-core` but
is depended on by `aleph-cli` and future GPU decoders. Keep `aleph-core`/`aleph-ir`
backend-agnostic — QEC types live here, not there.

**Technical Details**
- Add `crates/aleph-qec` to the workspace `Cargo.toml`.
- Core types:
  - `DetectorErrorModel { detectors: usize, observables: usize, errors: Vec<DemError> }`
    where `DemError { prob: f64, dets: Vec<u32>, obs: Vec<u32> }` (Stim-DEM shape).
  - `Syndrome` (bit-vector of fired detectors), `Correction` (observable flips applied).
  - `trait Decoder { fn decode(&self, syndrome: &Syndrome) -> Correction; }`
  - `LogicalErrorResult { shots, logical_errors, rate, ci95 }`.
- Crate-local `Error` enum via `thiserror`. No `unwrap` in lib code (CLAUDE.md).

**Acceptance Criteria**
- [ ] `cargo build -p aleph-qec` and `cargo test -p aleph-qec` pass.
- [ ] `DetectorErrorModel` round-trips to/from Stim `.dem` text format (parse + emit).
- [ ] `clippy -D warnings` and `fmt --check` clean.

**Testing Requirements**
- Unit: parse a known Stim `.dem` snippet, assert error count / probs / detector indices.
- Property: emit→parse round-trip is identity for random DEMs.

**References**
- Stim DEM format: https://github.com/quantumlib/Stim/blob/main/doc/file_format_dem.md

-----

### [Q0-02] Stochastic Pauli-frame noise injection into the stabilizer backend

**Labels:** `area:backend-stab`, `area:qec`, `type:feature`, `priority:critical`
**Milestone:** Phase Q0
**Estimate:** L
**Depends on:** Q0-01

**Description**
Add stochastic Pauli noise (depolarizing, X/Z flip, measurement flip) to the stabilizer
backend so it can produce **noisy syndromes**. Today noise lives only in the state-vector
backend, which cannot scale to QEC sizes.

**Context**
Surface-code Monte-Carlo needs thousands of qubits × thousands of shots — only the
stabilizer backend reaches that. The existing Pauli-frame batched sampler (P4.6-02) is the
natural insertion point: a stochastic Pauli channel is just a random sign/Pauli flip on the
frame, O(1) per qubit per shot, and stays Clifford.

**Technical Details**
- Channels: 1q/2q depolarizing (p), bit-flip, phase-flip, readout/measurement flip.
- Insert Pauli errors as random gates between circuit layers, seeded RNG for reproducibility.
- Reuse the 64-shot Pauli-frame batching: each shot gets independent error draws sharing one
  tableau. Per-shot only the Pauli frame differs (as in P4.6-02 sampling).
- Keep this separate from the state-vector `NoiseModel`; share the `error.rs` probability
  presets where types allow, but stabilizer noise is Pauli-only by construction.
- Update `aleph-backend::select` so the stabilizer backend reports `allows_noise() == true`
  for the Pauli-only subset (currently it rejects all noise).

**Acceptance Criteria**
- [ ] Stabilizer backend runs a circuit with depolarizing + measurement noise and emits
      per-shot syndrome bits.
- [ ] Single-qubit X error before a Z-ancilla deterministically fires that ancilla (matches
      the existing `surface_code_logical` tests but now via the noise channel).
- [ ] Sampled error frequencies match injected `p` within statistical tolerance (1e-2 at 1e5
      shots).
- [ ] n=1000, depth=100, 1000 shots with noise runs in < 1 s (no per-shot tableau rebuild).

**Testing Requirements**
- Unit: each channel on a Bell pair / single ancilla gives textbook syndrome statistics.
- Property: zero noise (p=0) reproduces the noiseless syndrome exactly.
- Oracle: detector firing statistics match Stim on the same circuit + noise (1e-2 at 1e5 shots).

**References**
- Stim measurement/noise sampling model.
- `docs/perf/surface_code.md` (Pauli-frame batched sampler, P4.6-02).

-----

### [Q0-03] Detector Error Model construction from a surface-code memory circuit

**Labels:** `area:qec`, `type:feature`, `priority:critical`
**Milestone:** Phase Q0
**Estimate:** L
**Depends on:** Q0-01, Q0-02

**Description**
Generate a multi-round surface-code **memory experiment** circuit and derive its Detector
Error Model: detectors (differences of consecutive stabilizer measurements) and logical
observable, with per-error probabilities and detector/observable supports.

**Context**
Decoders consume a DEM, not raw gates. The DEM encodes the matching graph weights. Building
it correctly (including space-time detectors across rounds and boundary edges) is the crux
of getting threshold behavior right.

**Technical Details**
- Extend the rotated surface code in `benches/src/lib.rs` (or port into `aleph-qec`) to emit
  a `d`-round memory experiment: init → (syndrome cycle × d) → final data readout.
- Detectors = XOR of the same stabilizer across consecutive rounds (and vs reset/readout at
  the boundaries in time).
- Logical observable = product of data qubits along a logical line.
- Two DEM sources, must agree:
  1. **Analytic**: enumerate each noise location → which detectors/observables it flips.
  2. **Empirical/oracle**: emit the equivalent Stim circuit, run `stim` to produce `.dem`,
     parse via Q0-01. Use as the cross-check.
- Start with phenomenological noise (data + measurement), then circuit-level (each gate).

**Acceptance Criteria**
- [ ] Generates a valid `DetectorErrorModel` for rotated surface code d ∈ {3,5,7,9}, any rounds.
- [ ] Analytic DEM matches the Stim-emitted DEM (same edges, probs within 1e-9) for d=3,5.
- [ ] Each detector has the expected degree (bulk = 2 error mechanisms per edge; boundary edges present).

**Testing Requirements**
- Unit: d=3 single-round DEM has the known number of detectors and a hand-checkable edge set.
- Oracle: full DEM equality vs `stim.Circuit.detector_error_model()` for d=3,5,7.

**References**
- Fowler et al., ArXiv:1208.0928 (surface codes, detectors).
- Stim circuit → DEM: `stim.Circuit.detector_error_model(decompose_errors=True)`.

-----

### [Q0-04] Surface-code memory experiment harness (logical error rate Monte-Carlo)

**Labels:** `area:qec`, `area:bench`, `type:feature`, `priority:critical`
**Milestone:** Phase Q0
**Estimate:** M
**Depends on:** Q0-02, Q0-03

**Description**
A harness that runs many noisy shots of the memory experiment, collects syndromes + true
observable flips, feeds them to a `Decoder`, and computes logical error rate with confidence
intervals. With no real decoder yet, ship a trivial `NullDecoder` (predicts no correction)
and a `PerfectMatchingOracle` placeholder (calls PyMatching via subprocess) to validate the loop.

**Context**
This is the reusable measurement instrument for the whole track. Every later decoder plugs
into it unchanged. It must be correct before any decoder is trusted.

**Technical Details**
- API: `run_memory_experiment(code, noise, rounds, shots, &dyn Decoder) -> LogicalErrorResult`.
- Logical error = (decoder-predicted observable flip) XOR (true observable flip).
- Wilson or normal-approx 95% CI on the rate.
- Parallelize over shots with rayon; reuse the batched sampler.
- Provide `NullDecoder` and an external `PyMatchingOracle` (subprocess to `pymatching`) so
  the loop produces a real threshold *before* Q1 exists.

**Acceptance Criteria**
- [ ] `run_memory_experiment` returns rate + CI for any (d, p, rounds, shots).
- [ ] With `PyMatchingOracle`, logical error rate decreases with d below threshold and
      increases with d above it (qualitatively correct).
- [ ] Deterministic given a fixed seed.

**Testing Requirements**
- Unit: p=0 → logical error rate exactly 0 for any decoder.
- Integration: small d=3, p=0.05, 1e4 shots produces a sane rate vs PyMatching baseline.

**References**
- PyMatching `Matching.from_detector_error_model`.

-----

### [Q0-05] Reproduce the surface-code threshold + Phase Q0 report

**Labels:** `area:qec`, `area:docs`, `type:test`, `priority:high`
**Milestone:** Phase Q0
**Estimate:** M
**Depends on:** Q0-04

**Description**
Run the full d-sweep × p-sweep, plot logical error rate vs p for each distance, locate the
crossing (threshold), and write `docs/perf/qec-q0-threshold.md`.

**Context**
The threshold plot is the "Hello World" of QEC and the proof that Q0-02/03/04 are correct.
It is also the first publishable artifact for the portfolio (ROADMAP Phase D).

**Technical Details**
- d ∈ {3,5,7,9}, p across ~8 points bracketing the expected threshold.
- Use `PyMatchingOracle` (real MWPM) so the threshold is meaningful before Q1.
- Plot via a small Python script under `scripts/`; commit the data CSV.

**Acceptance Criteria**
- [ ] Threshold p_th ≈ 0.5–1.0% for circuit-level noise (or the known phenomenological value
      ~3% if phenomenological), matching literature within the curves' resolution.
- [ ] Report committed with plot, CSV, exact noise model, shot counts, seeds.
- [ ] Stim cross-check: same threshold (within CI) when decoding the Stim-emitted DEM.

**Testing Requirements**
- Regression test (ignored/nightly): threshold within a tolerance band of the recorded value.

**References**
- Fowler et al., ArXiv:1208.0928 (threshold ~1% circuit-level).

-----

# Phase Q1 — MWPM Decoder

Goal: a from-scratch minimum-weight perfect matching decoder in Rust, plugged into the Q0
harness, benchmarked against PyMatching for correctness and speed.

**Exit metric:** aleph-MWPM logical error rate equals PyMatching's (within CI) on shared
DEMs for d ∈ {3,5,7,9}; speed reported (target: same order of magnitude as Sparse Blossom).

-----

### [Q1-01] Matching graph builder from a DEM

**Labels:** `area:decoder`, `type:feature`, `priority:critical`
**Milestone:** Phase Q1
**Estimate:** M
**Depends on:** Q0-03

**Description**
Convert a `DetectorErrorModel` into a weighted matching graph: nodes = detectors (+ a virtual
boundary node), edges = error mechanisms with weight `w = log((1-p)/p)`.

**Context**
MWPM and Union-Find both operate on this graph. Edges that flip a single detector connect it
to the boundary; edges flipping two detectors connect them. Edges flipping the observable
carry an observable-parity flag used to reconstruct the correction.

**Technical Details**
- Decompose multi-detector errors into edge pairs (the DEM from Q0-03 should already be
  graph-like for surface code; reject hyperedges with a clear error for now — qLDPC is Q5).
- Edge weight `log((1-p)/p)`; combine parallel edges by probability addition before weighting.
- Track per-edge observable-flip parity.

**Acceptance Criteria**
- [ ] Builds a graph with correct node/edge counts for d=3,5 (hand-checkable).
- [ ] Boundary edges present for detectors on the spatial/temporal boundary.
- [ ] Rejects non-graphlike (hyperedge) DEMs with a typed error.

**Testing Requirements**
- Unit: d=3 graph matches a hand-drawn adjacency.
- Property: every edge weight ≥ 0; observable parity preserved under parallel-edge merge.

**References**
- Higgott, PyMatching (ArXiv:2105.13082); Dennis et al. on matching weights.

-----

### [Q1-02] Blossom / Edmonds MWPM core

**Labels:** `area:decoder`, `type:feature`, `priority:critical`
**Milestone:** Phase Q1
**Estimate:** XL
**Depends on:** Q1-01

**Description**
Implement minimum-weight perfect matching on the syndrome graph (fired detectors + boundary)
via Edmonds' blossom algorithm, producing the set of edges → the correction.

**Context**
This is the fundamental decoder exercise (ROADMAP Phase B). Doing it from scratch (not
wrapping PyMatching) is the point — it builds the understanding needed for UF, GPU, and HW.

**Technical Details**
- Shortest-path distances between fired detectors on the weighted graph (Dijkstra; cache).
- Edmonds' blossom for minimum-weight perfect matching on the complete graph of syndrome
  defects, with the boundary as an always-matchable sink.
- Reconstruct correction = XOR of observable-parity along matched paths.
- Keep allocation out of the inner loop where feasible (hardware co-design rule).

**Acceptance Criteria**
- [ ] Correct minimum-weight matching on hand-verifiable small graphs.
- [ ] Integrated as a `Decoder`; runs end-to-end in the Q0 harness.
- [ ] Logical error rate matches PyMatching within CI on d=3,5 shared DEMs.

**Testing Requirements**
- Unit: known graphs with known optimal matchings.
- Property: matching weight ≤ any greedy matching's weight.
- Oracle: same corrections as PyMatching on ≥1e4 random syndromes per d.

**References**
- Edmonds (1965), "Paths, trees, and flowers."
- Kolmogorov, "Blossom V" (concepts only — do not copy code).

-----

### [Q1-03] Sparse / localized matching optimization

**Labels:** `area:decoder`, `type:optimization`, `priority:high`
**Milestone:** Phase Q1
**Estimate:** L
**Depends on:** Q1-02

**Description**
Optimize the MWPM core toward Sparse-Blossom-style locality: avoid the full all-pairs
shortest-path blowup by exploiting that defects only match locally on a geometric graph.

**Context**
Naïve blossom is O(n³) and won't scale. Sparse Blossom achieves ~1M errors/core-second by
growing regions locally. This is where aleph-MWPM either becomes competitive or stays a toy.

**Technical Details**
- Region-growing / local Dijkstra bounded by current matching radius.
- Lazy graph exploration; stop when a perfect matching is found.
- Benchmark each optimization with criterion; record before/after (CLAUDE.md golden rule).

**Acceptance Criteria**
- [ ] ≥10× faster than the Q1-02 baseline on d=11 syndromes at threshold density.
- [ ] Identical corrections to Q1-02 (optimization preserves correctness).

**Testing Requirements**
- Differential: Q1-03 output == Q1-02 output on 1e5 random syndromes.
- Benchmark: criterion d ∈ {7,9,11,13}, errors/second reported.

**References**
- Higgott, Gidney, "Sparse Blossom" (ArXiv:2303.15933).

-----

### [Q1-04] MWPM integration + decode→correct→verify in the harness

**Labels:** `area:qec`, `area:decoder`, `type:feature`, `priority:high`
**Milestone:** Phase Q1
**Estimate:** S
**Depends on:** Q1-02, Q0-04

**Description**
Wire aleph-MWPM as the default `Decoder` in the Q0 harness and regenerate the threshold plot
using the native decoder instead of the PyMatching oracle.

**Acceptance Criteria**
- [ ] Threshold plot reproduced with aleph-MWPM, matching the PyMatching-oracle plot within CI.
- [ ] `docs/perf/qec-q0-threshold.md` updated with the native-decoder curve overlaid.

**Testing Requirements**
- Regression: native-decoder threshold within tolerance of the oracle threshold.

-----

### [Q1-05] Benchmark + correctness vs PyMatching (Phase Q1 report)

**Labels:** `area:decoder`, `area:bench`, `area:docs`, `type:test`, `priority:high`
**Milestone:** Phase Q1
**Estimate:** M
**Depends on:** Q1-03, Q1-04

**Description**
Head-to-head report: aleph-MWPM vs PyMatching on shared DEMs — accuracy parity + decode
throughput (errors/second) across distances.

**Acceptance Criteria**
- [ ] Accuracy: logical error rate equal within CI for d ∈ {3,5,7,9,11}.
- [ ] Speed: errors/second reported per d; honest verdict on where aleph wins/loses.
- [ ] `docs/perf/qec-q1-mwpm.md` committed with methodology, machine, seeds.

**Testing Requirements**
- Oracle: 1e6 random syndromes per d, corrections compared (allow ties — equal weight, diff path).

**References**
- PyMatching benchmarks (ArXiv:2105.13082, 2303.15933).

-----

# Phase Q2 — Union-Find Decoder

Goal: an almost-linear-time Union-Find decoder (Delfosse-Nickerson), the natural precursor to
hardware. Faster than MWPM, slightly less accurate — characterize the trade-off.

**Exit metric:** UF decoder faster than aleph-MWPM on ≥1 regime, with logical error rate
within a documented factor of MWPM; both plugged into the same harness.

-----

### [Q2-01] Union-Find decoder core (Delfosse-Nickerson)

**Labels:** `area:decoder`, `type:feature`, `priority:high`
**Milestone:** Phase Q2
**Estimate:** L
**Depends on:** Q1-01

**Description**
Implement the Union-Find / cluster-growth decoder: grow clusters around defects until each
contains an even number of defects (or touches the boundary), then peel to a correction.

**Context**
UF is O(n α(n)) ≈ almost linear, with a simple, bounded-memory, integer-arithmetic control
flow — exactly what maps to FPGA/ASIC (Q6/Q7). This is the algorithm most likely to reach
the North Star.

**Technical Details**
- Weighted/unweighted cluster growth on the matching graph (Q1-01).
- Union-Find with path compression for cluster membership.
- Peeling decoder to extract the correction from the spanning forest.
- Keep all hot-loop data in fixed-size arrays (hardware co-design).

**Acceptance Criteria**
- [ ] Integrated as a `Decoder`; runs in the Q0 harness.
- [ ] Produces a valid correction (syndrome-consistent) for every input.
- [ ] Threshold reproduced (slightly below MWPM's, as expected).

**Testing Requirements**
- Property: every returned correction reproduces the input syndrome.
- Oracle: threshold within known UF-vs-MWPM gap of the MWPM result.

**References**
- Delfosse, Nickerson, ArXiv:1709.06218.

-----

### [Q2-02] Weighted-growth accuracy improvements

**Labels:** `area:decoder`, `type:optimization`, `priority:medium`
**Milestone:** Phase Q2
**Estimate:** M
**Depends on:** Q2-01

**Description**
Add weighted cluster growth / edge-weight awareness to close part of the accuracy gap to MWPM
while keeping near-linear runtime.

**Acceptance Criteria**
- [ ] Logical error rate improves vs Q2-01 at fixed d (documented).
- [ ] Runtime stays within 2× of Q2-01.

**Testing Requirements**
- Benchmark + accuracy comparison vs Q2-01 and MWPM.

**References**
- Huang, Newman, Brown, weighted Union-Find variants.

-----

### [Q2-03] UF vs MWPM trade-off report

**Labels:** `area:decoder`, `area:docs`, `type:docs`, `priority:medium`
**Milestone:** Phase Q2
**Estimate:** S
**Depends on:** Q2-02, Q1-05

**Description**
Document the speed/accuracy Pareto front: UF vs MWPM across distances and noise strengths.

**Acceptance Criteria**
- [ ] `docs/perf/qec-q2-unionfind.md` with errors/second + logical error rate for both, per d.
- [ ] Clear recommendation on when to use which (and which goes to hardware).

-----

# Phase Q3 — GPU Decoder (the differentiator)

Goal: exploit aleph's CUDA depth — a GPU decoder + massive end-to-end Monte-Carlo on GPU.
This is the unique angle (ROADMAP §2.4): nobody else has a co-designed GPU sim + GPU decoder.

**Exit metric:** GPU decoder beats CPU MWPM/UF on decode throughput at large d / high shot
counts; end-to-end (simulate noisy syndromes + decode) runs entirely on GPU.

-----

### [Q3-01] GPU Union-Find decoder

**Labels:** `area:backend-gpu`, `area:decoder`, `type:feature`, `priority:high`
**Milestone:** Phase Q3
**Estimate:** XL
**Depends on:** Q2-01

**Description**
Port the Union-Find decoder to CUDA in `crates/aleph-cuda`: parallel cluster growth across
many syndromes (batch decoding), device-resident graph.

**Context**
GPUs are bad at irregular pointer-chasing UF but great at batch-parallel decoding (one
syndrome per warp/block) — which is exactly the Monte-Carlo workload. Co-design with the GPU
stabilizer (P5-07) keeps syndromes on-device, avoiding PCIe round-trips.

**Technical Details**
- One block per syndrome shot; cluster growth in shared memory.
- Parallel Union-Find (e.g. lock-free / atomic merge) within a block.
- Reuse `aleph-cuda` NVRTC plumbing + stream-ordered pool from Phase 5.
- Keep graph in constant/texture memory (fixed per code).

**Acceptance Criteria**
- [ ] Bit-identical corrections to CPU UF (Q2-01) on the same syndromes.
- [ ] Decodes a batch of ≥1e4 syndromes; throughput beats CPU UF at d ≥ 9.

**Testing Requirements**
- Oracle: GPU vs CPU UF corrections match on 1e5 syndromes.
- Benchmark: syndromes/second GPU vs CPU vs PyMatching.

**References**
- Liyanage et al. (FPGA UF, ArXiv:2301.08419) for the parallel structure.

-----

### [Q3-02] GPU belief-propagation kernel (qLDPC-ready)

**Labels:** `area:backend-gpu`, `area:decoder`, `type:feature`, `priority:medium`
**Milestone:** Phase Q3
**Estimate:** XL
**Depends on:** Q3-01

**Description**
Implement min-sum / sum-product belief propagation on the GPU over a parity-check (Tanner)
graph — the workhorse for qLDPC codes (Phase Q5).

**Context**
BP is dense, regular, matrix-like → ideal for GPU (unlike UF). This is the bridge from
surface-code matching to qLDPC, the actual frontier (ROADMAP §2.3).

**Technical Details**
- Tanner graph as sparse check/variable adjacency in device memory.
- Min-sum message passing, fixed iteration count, batched over shots.
- Designed so BP+OSD (Q5-02) can layer on top.

**Acceptance Criteria**
- [ ] Converges on small surface/repetition codes to the correct correction.
- [ ] Batched throughput reported; matches a CPU BP reference within numerical tolerance.

**Testing Requirements**
- Unit: repetition code BP recovers known errors.
- Oracle: vs a CPU BP reference (e.g. `ldpc` library) on shared checks.

**References**
- Roffe et al., `ldpc`/BP+OSD; Panteleev-Kalachev BP+OSD.

-----

### [Q3-03] GPU end-to-end Monte-Carlo harness (simulate + decode on device)

**Labels:** `area:backend-gpu`, `area:qec`, `area:bench`, `type:feature`, `priority:high`
**Milestone:** Phase Q3
**Estimate:** L
**Depends on:** Q3-01, Q0-04

**Description**
Fuse the GPU stabilizer (noisy syndrome generation, P5-07 + Q0-02) with the GPU decoder so a
full threshold sweep runs without leaving the device.

**Context**
This is the headline capability: massive logical-error-rate Monte-Carlo (the bottleneck of
all decoder research) at GPU throughput. A genuine portfolio differentiator.

**Acceptance Criteria**
- [ ] End-to-end threshold sweep (d ∈ {3..13}) runs entirely on GPU.
- [ ] ≥10× faster wall-clock than the CPU harness (Q0-04) at matched shot counts.
- [ ] Threshold agrees with the CPU result within CI.

**Testing Requirements**
- Oracle: GPU threshold == CPU threshold within CI.
- Benchmark: shots/second end-to-end, GPU vs CPU.

-----

### [Q3-04] GPU decoder report (Phase Q3 verdict)

**Labels:** `area:backend-gpu`, `area:docs`, `type:docs`, `priority:high`
**Milestone:** Phase Q3
**Estimate:** S
**Depends on:** Q3-01, Q3-03

**Description**
Publish `docs/perf/qec-q3-gpu.md`: GPU decoder throughput/latency vs CPU MWPM/UF and
PyMatching; end-to-end Monte-Carlo speedup; honest where-it-wins/loses verdict.

**Acceptance Criteria**
- [ ] Throughput + per-shot latency tables, methodology, hardware (RTX 4000 Ada), seeds.
- [ ] Clear statement of the differentiator and its limits (latency vs FPGA still open → Q4/Q6).

-----

# Phase Q4 — Real-Time / Streaming

Goal: the shift from offline batch decoding to real-time streaming — the actual industrial
problem (ROADMAP §2.3). Latency, not throughput.

**Exit metric:** a sliding-window decoder that keeps up with a syndrome stream without backlog
growth, with a measured per-round latency budget.

-----

### [Q4-01] Sliding-window decoding

**Labels:** `area:decoder`, `type:feature`, `priority:high`
**Milestone:** Phase Q4
**Estimate:** L
**Depends on:** Q1-04 (or Q2-01)

**Description**
Decode a continuous syndrome stream in overlapping time windows, committing corrections for
the "core" of each window and carrying the boundary forward.

**Context**
Real devices produce syndromes forever; you cannot wait for the end. Sliding window is the
standard real-time approach. Correctness near window seams is the hard part.

**Technical Details**
- Window of W rounds, commit region of C < W, overlap for boundary continuity.
- Carry unmatched defects across window boundaries.
- Validate logical error rate vs full-batch decoding (should match for adequate W).

**Acceptance Criteria**
- [ ] Logical error rate within CI of full-batch decoding for W ≥ some d-dependent bound.
- [ ] Decodes an unbounded stream with bounded memory.

**Testing Requirements**
- Oracle: sliding-window vs batch corrections on long streams.

**References**
- Dennis et al.; Skoric et al. / Tan et al. on parallel-window decoding.

-----

### [Q4-02] Parallel-window decoding + backlog handling

**Labels:** `area:decoder`, `type:feature`, `priority:medium`
**Milestone:** Phase Q4
**Estimate:** L
**Depends on:** Q4-01

**Description**
Decode multiple windows concurrently and address the "backlog problem": if decode is slower
than syndrome arrival, the queue grows unboundedly and fault tolerance breaks.

**Acceptance Criteria**
- [ ] Throughput keeps pace with a configurable syndrome arrival rate (no unbounded backlog).
- [ ] Measured sustained syndrome-bits/second.

**References**
- Battistel et al., ArXiv:2303.00054 (real-time decoding survey, backlog problem).

-----

### [Q4-03] Latency-budget instrumentation

**Labels:** `area:decoder`, `area:bench`, `type:test`, `priority:high`
**Milestone:** Phase Q4
**Estimate:** M
**Depends on:** Q4-01

**Description**
Instrument per-round decode latency and produce a budget breakdown against the < 1 µs/round
target (the fault-tolerance threshold-breach constraint from the roadmap).

**Acceptance Criteria**
- [ ] Per-stage latency histogram (graph build, growth, peel, commit) for d ∈ {5,7,9,11}.
- [ ] `docs/perf/qec-q4-realtime.md` with the budget and the gap to 1 µs (motivates Q6 FPGA).

-----

# Phase Q5 — qLDPC Frontier

Goal: move past surface code to qLDPC (bivariate-bicycle / gross codes) — the genuine open
research frontier where decoders are *not* solved (ROADMAP §2.3).

**Exit metric:** BP+OSD on a gross code reaching a threshold within range of published results.

-----

### [Q5-01] Bivariate-bicycle (gross) code construction + DEM

**Labels:** `area:qec`, `type:feature`, `priority:medium`
**Milestone:** Phase Q5
**Estimate:** L
**Depends on:** Q0-03

**Description**
Construct IBM-style bivariate-bicycle (gross) codes, their stabilizers/Tanner graph, and a
DEM under circuit-level noise.

**Acceptance Criteria**
- [ ] [[144,12,12]] gross code constructed; parameters verified.
- [ ] Tanner graph + DEM emitted for decoding.

**References**
- Bravyi et al., ArXiv:2308.07915 (gross codes).

-----

### [Q5-02] BP+OSD decoder

**Labels:** `area:decoder`, `type:feature`, `priority:medium`
**Milestone:** Phase Q5
**Estimate:** XL
**Depends on:** Q5-01, Q3-02

**Description**
Belief propagation with ordered-statistics decoding post-processing — the standard qLDPC
decoder. Reuse the GPU BP kernel (Q3-02) for the BP stage.

**Acceptance Criteria**
- [ ] Decodes the gross code; logical error rate vs physical error rate curve produced.
- [ ] Threshold within range of published BP+OSD results for the same code/noise.

**References**
- Panteleev-Kalachev BP+OSD; Roffe `ldpc` library (concepts only).

-----

### [Q5-03] relay-BP / improvements + literature benchmark

**Labels:** `area:decoder`, `area:docs`, `type:optimization`, `priority:low`
**Milestone:** Phase Q5
**Estimate:** L
**Depends on:** Q5-02

**Description**
Explore recent BP improvements (relay-BP, memory/ambiguity-clustering variants) and benchmark
against the literature. Track ArXiv for the latest (≤12 months).

**Acceptance Criteria**
- [ ] ≥1 improvement implemented and measured vs Q5-02.
- [ ] `docs/perf/qec-q5-qldpc.md` with results positioned against published numbers.

-----

### [Q5-04] Circuit-level DEM for the gross code (depth-7 syndrome extraction)

**Labels:** `area:qec`, `type:feature`, `priority:medium`
**Milestone:** Phase Q5
**Depends on:** Q5-01

**Description**
Q5-01 shipped only a *code-capacity* DEM (single round, perfect measurements, independent `Z`
noise), despite its brief naming circuit-level noise; Q5-02/Q5-03 decoded that. Close the gap:
build the **depth-7 syndrome-extraction circuit** of Bravyi et al. (the exact CNOT schedule
`sX=[idle,1,4,3,5,0,2]`, `sZ=[3,5,0,1,2,4,idle]` and qubit-labelling convention from the
authors' reference implementation), run a `rounds`-cycle **memory-X** experiment, and emit a
**circuit-level** DEM under depolarizing noise (faulty CNOTs `4/15·p`, idle `2/3·p`, init/measure
flips). The DEM is a hypergraph (errors flip up to 6 detectors), consumed by BP+OSD (Q5-02) /
relay-BP (Q5-03) exactly like the code-capacity DEM.

**Acceptance Criteria**
- [ ] `BBCode::circuit_level_dem(rounds, noise)` builds the gross-code circuit-level DEM; the
  depth-7 schedule is verified conflict-free and measures both stabiliser types without disturbance.
- [ ] DEM cross-checked **edge-for-edge against Stim** for the same circuit (the determinism +
  correctness gate).
- [ ] `docs/perf/qec-q5-circuit-dem.md` with the DEM structure, a circuit-level logical-rate curve,
  and an honest comparison to the published ~0.7% threshold.

**References**
- Bravyi et al., ArXiv:2308.07915 (gross codes); `sbravyi/BivariateBicycleCodes` (reference circuit).

-----

### [Q5-05] relay-BP + OSD decoder; circuit-level threshold

**Labels:** `area:decoder`, `area:docs`, `type:optimization`, `priority:medium`
**Milestone:** Phase Q5
**Depends on:** Q5-02, Q5-03, Q5-04

**Description**
Q5-04 gave a Stim-exact circuit-level DEM but a modest BP+OSD reached only a ~0.1% threshold (per-
shot metric) vs the published ~0.7%. Improve the decoder and the methodology: (1) **relay-BP + OSD**
— feed relay-BP's (Q5-03) disordered-memory soft output into OSD's combination sweep (Q5-02), the
strongest qLDPC decoder; (2) use the correct **per-cycle** logical-error metric (a d=12 memory runs
2× the rounds of d=6) for the threshold crossing.

**Acceptance Criteria**
- [ ] `RelayBpOsdDecoder` implemented; beats BP+OSD and standalone relay-BP on the circuit-level DEM
  with CI separation.
- [ ] Circuit-level threshold re-measured with the per-cycle metric; honest positioning vs ~0.7% in
  `docs/perf/qec-q5-circuit-dem.md`.

**References**
- Panteleev-Kalachev BP+OSD; Müller et al. relay-BP; Bravyi et al. ArXiv:2308.07915.

-----

# Phase Q6 — FPGA

Goal: the real hardware milestone (ROADMAP §2 truth #1). Put a decoder on an FPGA and measure
latency. ASIC stays out of scope until this proves competitive.

**Exit metric:** Union-Find decoder on Arty A7 with measured per-round latency; GPU-vs-FPGA
comparison report.

-----

### [Q6-01] FPGA toolchain + Arty A7 bring-up

**Labels:** `area:fpga`, `type:infra`, `priority:medium`
**Milestone:** Phase Q6
**Estimate:** M
**Depends on:** Q2-01

**Description**
Acquire an Arty A7 (~$200), set up Vivado WebPack, and get a blinky + UART syndrome-in /
correction-out skeleton working.

**Acceptance Criteria**
- [ ] Toolchain builds and flashes the board.
- [ ] Host ↔ FPGA syndrome/correction round-trip over UART/PCIe verified on a trivial passthrough.

**References**
- Digilent Arty A7 docs; Vivado tutorials.

-----

### [Q6-02] Union-Find decoder on FPGA with measured latency

**Labels:** `area:fpga`, `area:decoder`, `type:feature`, `priority:medium`
**Milestone:** Phase Q6
**Estimate:** XL
**Depends on:** Q6-01, Q2-01

**Description**
Implement the Union-Find decoder in RTL (Verilog/SystemVerilog or Chisel/Amaranth) for a
small distance, and measure decode latency in cycles → ns.

**Context**
UF's bounded-memory integer control flow (designed in for since Q2) is what makes this
feasible. This is the first hard datapoint on the < 1 µs question.

**Acceptance Criteria**
- [ ] Decodes d=3,5 syndromes correctly on-board (vs CPU UF golden vectors).
- [ ] Measured per-round latency reported in ns.

**Testing Requirements**
- Co-simulation: RTL output == CPU UF (Q2-01) on a test-vector suite.

**References**
- Liyanage et al., ArXiv:2301.08419 (FPGA surface-code decoding).

-----

### [Q6-03] GPU vs FPGA comparison report

**Labels:** `area:fpga`, `area:docs`, `type:docs`, `priority:medium`
**Milestone:** Phase Q6
**Estimate:** S
**Depends on:** Q6-02, Q3-04

**Description**
Compare the same UF algorithm on GPU (Q3-01) vs FPGA (Q6-02): latency, throughput, power.
This report is the decision input for whether ASIC (Q7) is worth pursuing.

**Acceptance Criteria**
- [ ] `docs/perf/qec-q6-fpga.md` with latency/throughput/power for GPU and FPGA.
- [ ] Explicit recommendation: does an ASIC close a gap that FPGA cannot? Go/no-go for Q7.

-----

# Phase Q7 — ASIC (North Star)

Goal: the endgame from ROADMAP §0. **Gated** on Q6 proving FPGA competitive AND a business
gate (funding + real QPU-company customer). Kept deliberately high-level — detail this only
when Q6 results justify it.

**Exit metric:** an architecture spec + RTL core + tape-out feasibility study with a funding
and customer commitment in place. (Tape-out itself is a separate, funded program.)

-----

### [Q7-01] Decoder ASIC architecture spec + HW/SW partitioning

**Labels:** `area:asic`, `area:docs`, `type:docs`, `priority:low`
**Milestone:** Phase Q7
**Estimate:** L
**Depends on:** Q6-03

**Description**
Specify the decoder ASIC: which operations are ASIC vs FPGA vs CPU, target node, latency/power
budget, cryogenic vs room-temperature placement, I/O to the QPU control stack.

**Acceptance Criteria**
- [ ] Architecture document with block diagram, budgets, and partitioning rationale.
- [ ] Cost/timeline estimate for an MPW-shuttle prototype.

**References**
- Battistel et al. (real-time constraints); Riverlane Deltaflow architecture posts.

-----

### [Q7-02] RTL implementation of the core decoder block

**Labels:** `area:asic`, `area:fpga`, `type:feature`, `priority:low`
**Milestone:** Phase Q7
**Estimate:** XL
**Depends on:** Q7-01

**Description**
Synthesizable RTL of the core decode datapath, verified in simulation and (where possible) on
FPGA as an ASIC prototype, targeting the spec's latency/power.

**Acceptance Criteria**
- [ ] RTL passes full co-simulation against the software decoder golden model.
- [ ] Synthesis reports meet (or quantify the gap to) the Q7-01 budgets.

-----

### [Q7-03] MPW tape-out feasibility + funding/customer gate

**Labels:** `area:asic`, `type:docs`, `priority:low`
**Milestone:** Phase Q7
**Estimate:** L
**Depends on:** Q7-02

**Description**
Feasibility study for a multi-project-wafer tape-out: foundry/shuttle options, NRE cost,
timeline — AND the business gate: funding secured + a committed QPU-company customer.

**Context**
This is the honest go/no-go. ASIC only makes sense with money and a customer (ROADMAP §0).
Without both, stop at FPGA — that is still a complete, valuable outcome.

**Acceptance Criteria**
- [ ] Tape-out cost/timeline documented (shuttle vs full mask).
- [ ] Explicit gate decision recorded: funding + customer present → proceed; else → stop at FPGA.

**References**
- Riverlane as precedent (decoder ASIC company founded by non-physicist).
