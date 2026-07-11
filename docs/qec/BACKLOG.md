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

**Exit metric:** Union-Find decoder on the FPGA with measured per-round latency; GPU-vs-FPGA
comparison report. Targeted at **two boards** (see Hardware) — synthesis is board-independent, so
the whole pre-silicon path proceeds now regardless of shipping.

**Hardware (both ordered; ETA unknown — design for both):**
- **Digilent Zybo Z7-20** — Zynq-7020 (`xc7z020clg400-1`): dual Cortex-A9 PS + Artix-class PL
  (~53k LUT, 140 BRAM36, 220 DSP). The smaller, cheaper, likely-sooner board. Fit headroom is tight
  → the d=5 scaling risk lives here.
- **Xilinx Kria KV260** — K26 SOM (`xck26-sfvc784-2LV-c`): Zynq UltraScale+ PS + much larger PL
  (~256k LUT). The headroom board; self-programs from its on-board ARM Linux.

Sim is Mac-native (Verilator); synthesis (Vivado, x86-Linux box `openwebgui`) builds **both** parts
from one RTL base; only the final on-board bring-up needs the physical board.

**Execution order:** Q6-04 → Q6-09 are the board-independent pre-silicon path (synthesizable rewrite,
dual-target synth, gate-level sign-off, PS↔PL integration, host software, d=5 scaling). Run them in
order now; do the board ACs of Q6-01/Q6-02 when hardware lands; **Q6-03** (the GPU-vs-FPGA report)
is last because it needs measured on-board numbers.

-----

### [Q6-01] FPGA toolchain bring-up (sim-first; KV260)

**Labels:** `area:fpga`, `type:infra`, `priority:medium`
**Milestone:** Phase Q6
**Depends on:** Q2-01

**Description**
Stand up the FPGA flow **simulation-first** so RTL is verified before the board arrives. Build a
synthesisable syndrome-in / correction-out decoder skeleton, verify it bit-exactly in Verilator
against the Rust decoder, then (when the KV260 lands) build the bitstream in Vivado and verify the
host↔board round-trip.

**Acceptance Criteria**
- [x] **(sim)** A real-decoder RTL skeleton (d=3 surface-code LUT, generated by the Rust Union-Find
  decoder) passes a Verilator testbench over all syndromes; per-decode latency reported. (`hw/`)
- [ ] **(board, pending hardware)** Vivado builds + flashes the KV260; host ↔ board
  syndrome/correction round-trip verified on the passthrough.

**References**
- Kria KV260 / Vivado docs; Verilator manual.

-----

### [Q6-02] Union-Find decoder on FPGA with measured latency

**Labels:** `area:fpga`, `area:decoder`, `type:feature`, `priority:medium`
**Milestone:** Phase Q6
**Estimate:** XL
**Depends on:** Q6-01, Q2-01

**Description**
Implement the Union-Find decoder in RTL (SystemVerilog) and measure decode latency in cycles → ns.
Built sim-first: a 1-D **repetition-code** UF core lands first (the line specialisation of UF =
minimum-weight matching), then the 2-D **surface-code cluster-growth** UF (grow / union / peel).

**Context**
UF's bounded-memory integer control flow (designed in for since Q2) is what makes this
feasible. This is the first hard datapoint on the < 1 µs question.

**Acceptance Criteria**
- [x] **(sim)** Repetition-code UF datapath in RTL (`hw/uf_rep_decoder.sv`): a Verilator testbench
  over all syndromes confirms the correction reproduces the syndrome and the logical flip matches the
  CPU `UnionFindDecoder` (Q2-01); 1-cycle latency.
- [x] **(sim)** 2-D surface-code cluster-growth UF datapath (`hw/uf_surface_decoder.sv`: growth →
  spanning forest → peeling on the d=3 matching graph). Verilator TB: valid on all syndromes,
  distance-3 correct (all weight-1 errors), weight-≤2 logical-error-rate matches/beats the CPU UF
  (40 vs 50). Tie-breaks differ from CPU UF on degenerate syndromes (both valid min-weight).
- [ ] **(sim)** Parametrise to d=5; **pipeline the growth iterations** for timing (currently a
  single-cycle combinational decode). → **carried by Q6-04 (sequential rewrite) + Q6-09 (d=5 scaling).**
- [ ] **(board, pending hardware)** Decodes on the board; measured per-round latency in ns.
  → **carried by Q6-08** (host bring-up on Zybo / KV260).

**Testing Requirements**
- Co-simulation: RTL output == CPU UF (Q2-01) on a test-vector suite. ✅ (repetition code)

**References**
- Liyanage et al., ArXiv:2301.08419 (FPGA surface-code decoding).

-----

### [Q6-03] GPU vs FPGA comparison report — ✅ DONE (#391, on post-route estimates)

**Labels:** `area:fpga`, `area:docs`, `type:docs`, `priority:medium`
**Milestone:** Phase Q6
**Estimate:** S
**Depends on:** Q6-08, Q6-09, Q3-04 (runs **last** — needs measured on-board numbers)

**Description**
Compare the same UF algorithm on GPU (Q3-01) vs FPGA (Q6-08/Q6-09): latency, throughput, power.
This report is the decision input for whether ASIC (Q7) is worth pursuing.

**Acceptance Criteria**
- [x] `docs/perf/qec-q6-fpga.md` with latency/throughput/power for GPU and FPGA (both boards).
- [x] Explicit recommendation: does an ASIC close a gap that FPGA cannot? Go/no-go for Q7.

**Outcome (`docs/perf/qec-q6-fpga.md` §Q6-03).** FPGA wins single-decode **latency** by ~10–50×
(deterministic sub-µs vs GPU's launch+PCIe-bound batch device) and **energy/decode** by 150–600×
(~0.1–0.22 µJ vs ~29–120 µJ); GPU wins raw batch throughput but a single FPGA instance is already in
its class and replicates across spare fabric. **Verdict: conditional GO for Q7** — the FPGA latency is
76–82 % routing, a tax a std-cell ASIC removes (→ sub-100 ns, d≥9 real-time, µW-class, one
decoder/qubit, cryo); gate tape-out on funding + customer per Q7. **Caveat:** built on Vivado
post-route *estimates*; Q6-08 on-board bring-up must confirm with measured silicon before any tape-out
commitment. (Done ahead of Q6-08 because the synth numbers were decisive enough to make the Q7 call.)

-----

### [Q6-04] Synthesizable sequential UF decoder (de-combinationalize growth/forest/peel)

**Labels:** `area:fpga`, `area:decoder`, `type:refactor`, `priority:high`
**Milestone:** Phase Q6
**Estimate:** L
**Depends on:** Q6-02

**Description**
`hw/uf_surface_decoder.sv` currently does the *entire* decode (growth → spanning forest → peeling)
inside one `always_comb`: a triple-nested fixpoint (`for gi … for p … for e`) unrolled into a single
giant combinational cloud, with only the result registered to a 1-cycle handshake. Verilator-correct,
but for synthesis this will not close timing at any useful clock and blows up area. Rewrite the engine
as a **clocked FSM** with bounded per-cycle work (one growth step / one connected-components pass /
one peel step per cycle), all working state in registers/arrays, and a multi-cycle valid handshake
with a latency counter. Keep it parametrised by the generated graph (`uf_surface_graph.svh`) so d=5
follows in Q6-09.

**Context**
This is the precondition for *every* downstream task: synthesis (Q6-05), gate-level sign-off (Q6-06),
and the on-board latency number (Q6-08) are all meaningless on a combinational-cloud netlist. It
supersedes the open "pipeline the growth iterations" criterion of Q6-02.

**Acceptance Criteria**
- [ ] Clocked FSM replaces the single `always_comb` decode; per-cycle combinational depth bounded
  (no unrolled fixpoint cloud); cluster/forest/peel state held in registers or BRAM-inferable arrays.
- [ ] Multi-cycle `in_valid → out_valid` handshake with `busy`/`done`; decode latency exposed as a
  cycle count.
- [ ] **Bit-for-bit identical** `correction` and `obs_flip` vs the current combinational decode on all
  256 d=3 syndromes (regression-locked in the Verilator TB).
- [ ] Verilator TB green: validity on all syndromes, distance-3 correctness, quality unchanged
  (RTL 40 vs CPU UF 50).
- [ ] No inferred latches; single synchronous-reset style; `verilator -Wall` lint-clean.

**Testing Requirements**
- Co-simulation against the frozen combinational golden output (snapshot the current 256-row
  {obs_flip, correction} table, assert equality).

-----

### [Q6-05] Vivado dual-target synth/impl flow (XC7Z020 + XCK26)

**Labels:** `area:fpga`, `type:infra`, `priority:high`
**Milestone:** Phase Q6
**Estimate:** M
**Depends on:** Q6-04

**Description**
Stand up a non-project Tcl Vivado flow on the x86 Linux box (`openwebgui`) and build the decoder for
**both** parts — no board required. Synthesis answers the two questions that decide the whole
prototype: **does it fit**, and **how fast** (Fmax). Free Vivado edition covers both the Zynq-7020 and
the Kria K26.

**Acceptance Criteria**
- [ ] `hw/syn/` non-project Tcl flow: read RTL → `synth_design` → `opt/place/route` →
  `report_utilization` + `report_timing_summary`, parameterised by `-part`.
- [ ] Per-board XDC (clock + reset, PL clock from PS): Zybo Z7-20 (`xc7z020clg400-1`) and KV260
  (`xck26-sfvc784-2LV-c`).
- [ ] Utilization (LUT/FF/BRAM/DSP) and Fmax (WNS at a target clock) for d=3 on **both** parts,
  committed to `docs/perf/qec-q6-fpga.md`.
- [ ] Explicit fit verdict for XC7Z020 (53k LUT / 140 BRAM / 220 DSP) and headroom for XCK26.

-----

### [Q6-06] Post-synth / post-impl gate-level sign-off (xsim)

**Labels:** `area:fpga`, `type:test`, `priority:medium`
**Milestone:** Phase Q6
**Estimate:** S
**Depends on:** Q6-05

**Description**
Re-run the verification vectors on the synthesized and implemented netlist in Vivado `xsim` to catch
what Verilator RTL sim cannot: synth/Verilator semantic mismatch, inferred latches, X-propagation,
and reset behaviour. Board-independent.

**Acceptance Criteria**
- [ ] Post-synthesis functional sim of the netlist passes the full 256-syndrome vector suite.
- [ ] Post-implementation timing sim (SDF back-annotated) passes at the closed clock; no `X` on
  outputs after reset deassertion.
- [ ] Any RTL fixes for sim/synth parity folded back; Verilator TB still green.

-----

### [Q6-07] PS↔PL integration: AXI4-Lite control + AXI4-Stream syndrome wrapper

**Labels:** `area:fpga`, `type:feature`, `priority:medium`
**Milestone:** Phase Q6
**Estimate:** L
**Depends on:** Q6-04

**Description**
Wrap the decoder for the Zynq PS so the ARM can drive it: an **AXI4-Lite** slave for control/status +
register map, and **AXI4-Stream** for syndrome ingress / correction egress (this is where the Q4
sliding-window streaming model maps onto hardware). The wrapper is PS-agnostic — identical for the
Zynq-7020 (Zybo) and Zynq UltraScale+ (KV260) PS. Fully simulatable; no board.

**Acceptance Criteria**
- [ ] `hw/uf_axi_wrap.sv`: AXI4-Lite slave (start, busy/done, latency, obs_flip, correction read-back)
  + AXI4-Stream syndrome ingress.
- [ ] Register map documented in `hw/README.md`.
- [ ] Cocotb (or Verilog-AXI) testbench drives a syndrome frame in and reads the correction out,
  matching the bare decoder on the vector suite. No board.
- [ ] Wrapper builds for both PS variants (Zynq-7020 and Zynq UltraScale+).

-----

### [Q6-08] PS-side host software + on-board bring-up + latency wiring

**Labels:** `area:fpga`, `area:decoder`, `type:feature`, `priority:medium`
**Milestone:** Phase Q6
**Estimate:** M
**Depends on:** Q6-07

**Description**
Bare-metal C (Vitis) host that configures the AXI-Lite registers, streams a syndrome, polls `done`,
and reads `correction`/`obs_flip`/`latency`. Wire the PL cycle-counter latency into the Q4-03
latency-budget instrumentation so on-board numbers land in the same table. Software dev is
board-independent (against the block design); the on-board round-trip closes when hardware arrives.

**Acceptance Criteria**
- [x] `hw/sw/` host driver: configure AXI-Lite, stream a syndrome, poll done, read results;
  portable across Zynq-7020 / UltraScale+ PS. (Bare-metal C `hw/sw/uf_decoder.c` + a PYNQ/Python
  twin `hw/sw/uf_pynq.py` for the Linux/LAN bring-up path — same regmap, verified vs golden.)
- [x] PL-reported latency surfaced to the host and folded into the latency-budget instrumentation
  (LATENCY register → `uf_latency_ns`; measured on-board below).
- [x] **End-to-end on hardware (Arty Z7-20, `xc7z020clg400-1` — same PL part as the Zybo Z7-20):**
  built the board bitstream (`hw/syn/arty_z7_bd.tcl`, Zynq7 PS + `uf_axi_top` on AXI GP0, FCLK
  50 MHz, WNS +7.29 ns), loaded via PYNQ overlay, host↔PL round-trip verified — **256/256 syndromes
  bit-identical to golden, IDCODE ok, worst decode latency 30 clk = 600 ns @ 50 MHz (< 1 µs
  round budget → real-time on silicon)**. **Closes the board ACs of Q6-01 and Q6-02.**

-----

### [Q6-09] d=5 parametrization + scaling/fit study (both boards)

**Labels:** `area:fpga`, `area:decoder`, `type:feature`, `priority:medium`
**Milestone:** Phase Q6
**Estimate:** L
**Depends on:** Q6-04, Q6-05

**Description**
Drive the generated-graph parametrisation to **d=5** (and project d=7), re-verify in Verilator, and
re-synth on both parts. This is where the XC7Z020 fit risk gets resolved: report where the decoder
stops fitting the small part and how much headroom the K26 has — deciding which board carries which
code distance, and feeding Q6-03 and the ASIC question.

**Acceptance Criteria**
- [ ] d=5 matching graph generated (`qec_surface_uf_graph`); UF FSM verified in Verilator (validity +
  distance-5 correctness vs the CPU UF).
- [ ] Synth/impl for d=3 and d=5 on `xc7z020clg400-1` and `xck26-sfvc784-2LV-c`: utilization + Fmax
  table in `docs/perf/qec-q6-fpga.md`.
- [ ] Scaling verdict: max code distance per board; decode latency (cycles → ns at Fmax) vs the 1 µs
  per-round budget, per distance and board.

-----

## Phase Q6.5 — from code-capacity toy to a *real* decoder (no hardware needed)

Q6-04…Q6-18 built and optimized a synthesizable UF decoder that is **real-time at d=5 (both boards)
and d=7 (KV260)** — but on a **single-round, code-capacity** matching graph (2D, space only). A *real*
QEC decoder must handle the **time dimension**: many measurement rounds, with measurement errors, on
the 3D space-time matching graph, decoded as a bounded-latency **stream** (you cannot wait for the
experiment to end). That gap — not more Fmax — is what separates this from the field
(Riverlane/Google decode circuit-level, streaming, at distance). These three issues close it, all
**board-free** (Verilator + Vivado + the existing simulator); board bring-up (Q6-01/02/08) waits on
hardware. The ASIC (Q7) stays parked until this lands and we are at/near the frontier (see Q7 note).

### [Q6-19] Multi-round (3D space-time) decoding on the FPGA decoder

**Labels:** `area:fpga`, `area:decoder`, `type:feature`, `priority:high`
**Milestone:** Phase Q6
**Estimate:** M
**Depends on:** Q6-17

**Description**
Move the FPGA decoder from a single-round code-capacity graph to a **multi-round phenomenological**
3D space-time graph: `T` measurement rounds with **measurement-error (time-like) edges** between
consecutive rounds, plus the usual data edges. The decoder RTL (`hw/uf_surface_decoder.sv`) is
**graph-agnostic** (parametric in `UF_N`/`UF_M`/edge tables) and was verified to decode a 3D graph
**with zero RTL changes** during scoping — so this is graph-generation + verification + a synth/scaling
verdict, not an RTL rewrite. The generator (`qec_surface_uf_graph`) already gained a `rounds` argument
(`graph <d> <rounds>`, built on `SurfaceCode::memory_z_experiment(rounds)` + the existing
`phenomenological_mechanisms`, whose `p_meas` creates the time-like edges).

**Scoping results (validated):** d=5 × 3 rounds → 48 detectors / 120 edges, **0/7140 weight-≤2
logical errors** (d=5 corrects ≤2), 85 clk worst-case; d=3 × 5 rounds → 24 det / 62 edges, validity +
distance(weight-1) clean. Note: at d=5 × 5 rounds the graph hits 72 detectors → the **syndrome
exceeds 64 bits**, which the scale TB's 64-bit syndrome can't hold (the input-side analogue of the
d=7 wide-`correction` fix) — extend the TB with a wide-syndrome setter to reach the GPU-comparable
d=5×5 graph (72 det / 186 edges — *identical size to the Q3 GPU bench*, enabling a true apples-to-apples
GPU-vs-FPGA latency number for Q6-03).

**Acceptance Criteria**
- [ ] `surf-3d` make target generating + Verilator-verifying ≥1 clean 3D case (d=5×3: validity 0,
  distance(weight-1) 0, weight-≤2 = 0 logical errors).
- [ ] Dual-target synth (Zybo + KV260) of a representative 3D case: fit / Fmax / latency vs the budget
  in `docs/perf/qec-q6-fpga.md`, and the cycle/area scaling vs round count.
- [ ] Wide-syndrome TB extension so >64-detector graphs verify (unlocks d=5×5 = the GPU-comparable size).
- [ ] **Follow-up noted:** *phenomenological* ≠ *circuit-level*. Full circuit-level (gate/CX-level
  noise) needs a surface-code `circuit_level_mechanisms()` mirroring `BBCode::circuit_level_dem`
  (`bivariate_bicycle.rs`) — track as a refinement once phenomenological 3D ships.

### [Q6-20] Sliding-window streaming decoder on FPGA (bounded-latency real-time)

**Labels:** `area:fpga`, `area:decoder`, `type:feature`, `priority:high`
**Milestone:** Phase Q6
**Estimate:** L
**Depends on:** Q6-19

**Description**
A real decoder consumes an **unbounded stream** of rounds at the QPU's rate; it cannot store the whole
history or it falls behind (the backlog problem). Transcribe the **already-complete, tested** software
`SlidingWindowDecoder` (`crates/aleph-qec/src/sliding.rs`) to an RTL streaming wrapper around the
existing per-window UF core: decode a window of `W` rounds, **commit** the oldest `C` rounds, carry the
**residual** syndrome forward, and handle the **temporal-sink** seam nodes (out-of-window detectors
drain to a separate sink, *not* the spatial boundary, so time-cut edges don't spuriously flip the
logical). Per-window working set is `O(W)` → fixed on-chip RAM, independent of stream length.

**Hard parts (flagged in scoping):** residual read-modify-write on on-chip memory between windows;
per-window temporal-sink set differs (pre-allocate a sink pool); DMA sequencing so window latency
stays under the syndrome arrival rate (else backlog). Software window params for reference (Q4-03):
`W = 3d`, `C = d`, buffer `= 2d`.

**Acceptance Criteria**
- [ ] RTL streaming wrapper: windowed decode + commit + residual carry + temporal sinks, Verilator-
  verified bit-equal to the software `SlidingWindowDecoder` on long streams.
- [ ] Bounded memory (per-window working set independent of total rounds) demonstrated.
- [ ] Per-window latency vs the per-round budget (cycles → ns at Fmax), synth on ≥1 board.

### [Q6-21] Sim↔RTL co-simulation: simulator as a QPU emulator driving the decoder (board-free HiL) — ✅ DONE (sim)

**Labels:** `area:fpga`, `area:decoder`, `type:test`, `priority:medium`
**Milestone:** Phase Q6
**Estimate:** M
**Depends on:** Q6-19

**Description**
The board-free form of hardware-in-the-loop, and the ROADMAP §2.4 co-design differentiator made
concrete: drive the **Verilated decoder model** from the simulator's Monte-Carlo syndrome stream
(`qec_threshold` / on-device sampling), feed corrections back, and measure the **logical error rate /
threshold using the actual RTL decoder** — confirming it matches the software decoder's threshold
(Q0). Closes the whole verification chain on realistic noise: noise model → syndromes → **RTL** decode
→ logical error rate. When hardware lands (Q6-08) the same harness swaps the Verilated model for the
real board over the Q6-07 AXI link.

**Acceptance Criteria**
- [x] Harness streams simulator syndromes (≥ phenomenological, ideally 3D from Q6-19) into the
  Verilated `uf_surface_decoder`, collects `obs_flip`, accumulates logical error rate.
  *(`qec_q6_cosim.rs` + `hw/tb_uf_cosim.cpp`; `make -C hw cosim` d=3 + `cosim-3d` d=5×3 3-D graph.)*
- [x] RTL-decoder threshold/LER curve matches the software UF decoder within Monte-Carlo CI (a plot or
  table in `docs/perf/`). *(`docs/perf/qec-q6-cosim.md`: d=3 within CI at every p; d=5×3 within CI
  sub-threshold (p≤0.02). The supra-threshold gap — RTL UF is unweighted/bounded-depth so it
  tie-breaks degenerate cosets more crudely — is reported honestly as a true HiL-surfaced quality
  difference, not noise.)*
- [x] Documented as the board-free HiL path; note the AXI swap-in for Q6-08.
  *(`docs/perf/qec-q6-cosim.md` § "The AXI swap-in"; same `.vec` stream over `uf_axi_wrap.sv`.)*

**Result:** the verification chain noise→syndrome→**RTL**→LER closes board-free. See
`docs/perf/qec-q6-cosim.md`.

### [Q6-22] Streaming decoder on silicon + finite-experiment warm-up/drain (measured) — ✅ DONE

**Labels:** `area:fpga`, `area:decoder`, `type:feature`, `priority:medium`
**Milestone:** Phase Q6
**Depends on:** Q6-20

**Description**
Put the Q6-20 sliding-window streaming decoder on **real Arty Z7-20 silicon** over the AXI-DMA path
(`hw/uf_stream_win_core.sv` → `uf_streaming_decoder`, one round/beat in, one word/window out), and close
the Q6-20 warm-up/drain caveat — finite experiments have real time boundaries the steady-state interior
wrapper doesn't model. **Measure before building** the head/tail RTL: compare the on-board streaming
LER (interior windows + zero-drain) to the boundary-aware software `SlidingWindowDecoder` on the same
shots.

**Acceptance Criteria**
- [x] AXI4-Stream front-end + DMA block design + on-board driver; per-frame re-arm so each DMA transfer
  is an independent stream (fresh warm-up). *(PR #412; `stream-axi` frame-independence 6/6.)*
- [x] On silicon: validity drains at every p; sustained window rate under the `C`-µs commit budget.
  *(391k windows/s = 2.55 µs/window, real-time 1.2× @ 50 MHz.)*
- [x] Finite-experiment streaming LER within CI of the boundary-aware software (`qec_q6_stream_ler` +
  `uf_dma_stream_ler.py`). *(d=3: within CI at all p=0.01–0.05; E=40 000 at p=0.01 → |diff| 1.15e-3,
  shrinks with shots ⇒ noise, not a boundary offset.)*
- [x] **Verdict:** interior+drain finite handling is statistically equivalent to boundary-aware
  software at these sizes ⇒ **no head/tail RTL warranted.** See `docs/perf/qec-q6-fpga.md` § Q6-22.

### [Q6-23] Circuit-level (gate-noise) noise through the streaming decoder — ✅ DONE

**Labels:** `area:fpga`, `area:decoder`, `type:feature`, `priority:medium`
**Milestone:** Phase Q6
**Depends on:** Q6-20, Q-surface circuit-level DEM

**Description**
Close the last Q6.5 noise gap: run the circuit-level (gate-noise + hook-error) surface DEM through the
**streaming** window path — realistic noise *and* streaming together, on silicon. The circuit-level DEM
(Stim-verified graphlike) already exists; the streaming decoder is graph-agnostic, so this is a
graph-generation + verification + synth/board task, not an RTL rewrite.

**Acceptance Criteria**
- [x] `qec_surface_uf_graph -- window-circuit` builds the interior window graph from `circuit_level_dem`
  (same detectors + bit-identical streaming metadata; only edges differ, `UF_M` 111→141 hook edges).
- [x] Streaming decoder decodes the circuit-level window graph with **zero RTL change** (`make
  stream-axi-circuit`: validity 40/40, back-pressure 40/40, frame-indep 6/6).
- [x] On silicon (Arty Z7-20, circuit-level bitstream, WNS +0.128 ns, 25.2 % LUT): validity drains at
  every p=0.002–0.010; streaming LER within CI of the boundary-aware software; sustained
  2.87 µs/window (real-time). See `docs/perf/qec-q6-fpga.md` § Q6-23.
- [x] **Verdict:** the complete "real decoder" configuration — circuit-level gate noise, streaming,
  bounded memory, matching software LER within CI — runs on the FPGA at d=3.

-----

# Phase Q7 — ASIC (North Star)

Goal: the endgame from ROADMAP §0. **Gated** on Q6 proving FPGA competitive AND a business
gate (funding + real QPU-company customer). Kept deliberately high-level — detail this only
when Q6 results justify it.

**Exit metric:** an architecture spec + RTL core + tape-out feasibility study with a funding
and customer commitment in place. (Tape-out itself is a separate, funded program.)

> **Do not build a me-too ASIC.** Q6-03 returned a *conditional* GO: an ASIC closes real gaps
> (sub-100 ns, d≥9 real-time, µW/decode, one-decoder-per-qubit, cryo). But it only makes sense once we
> are **at or near the frontier** — i.e. Q6.5 (circuit-level + streaming, Q6-19/20) lands so the chip
> decodes *real* QEC traffic, and measured silicon (Q6-08) confirms the estimates. Taping out a chip
> that merely re-does what Riverlane/Google already ship adds nothing. The trigger is **frontier-parity
> + a customer + funding**, not just funding.

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

-----

### [Q7-04] Multi-round sliding-window streaming relay-BP on FPGA (real-time M9)

**Labels:** `area:fpga`, `type:feature`, `priority:medium`
**Milestone:** Phase Q7
**Estimate:** XL
**Depends on:** Q7-02 (M8 core)

**Description**
The banked relay-BP core (M7/M8: 15.64 µs worst / 0.85 µs median on KV260) decodes one syndrome
batch at rounds=1. Real-time QEC is a continuous stream of measurement rounds (~1 µs/round)
decoded in a sliding window with bounded per-round latency — the backlog problem. Build the
BB-code analog of the surface-code streaming decoder (Q6-20/Q6-22, DONE for UF): multi-round
circuit-level graph from the emitter (rounds>1), a window/commit schedule over the banked core,
and a sustained-throughput measurement on silicon against a target round rate. This is the
product-defining gap between the lab prototype and a deployable decoder, and the main de-risk
input for Q7-01.

**Acceptance Criteria**
- [ ] Emitter generates multi-round (rounds ≥ 3) circuit-level windows; golden model matches.
- [ ] Streaming schedule (window advance + commit) on the banked core, bit-exact to the windowed
      golden in co-sim.
- [ ] On silicon: sustained decode of a round stream with a measured per-round latency
      distribution and the max round rate the decoder keeps up with (no unbounded backlog).

-----

### [Q7-05] KV260 power measurement — W and energy per decode

**Labels:** `area:fpga`, `type:infra`, `priority:medium`
**Milestone:** Phase Q7
**Estimate:** S
**Depends on:** Q7-02

**Description**
No power data exists for any decoder build; ASIC projections (Q7-01) and the FPGA-product story
both need it. The Kria SOM exposes PMBus rails (INA226) readable from PYNQ. Measure idle vs
decode-load power for the shipped M8 overlay (full-schedule and early-exit sweeps), derive
energy per decode.

**Acceptance Criteria**
- [ ] Scripted rail readout on the KV260 alongside the standard 40-vector run.
- [ ] Reported: PL power idle/under-load, energy per decode (both modes), documented in
      `docs/perf/qec-q7-fixed-bp.md`.

-----

### [Q7-06] Silicon-accelerated LER qualification (batched decode interface + MC campaign)

**Labels:** `area:fpga`, `type:feature`, `priority:medium`
**Milestone:** Phase Q7
**Estimate:** L
**Depends on:** Q7-02

**Description**
Freezing a decoder in silicon needs an LER qualification matrix far beyond the 40-vector
bit-exact gate. The board decodes in ~1 µs mean but the AXI-Lite per-word harness dominates
wall time. Add a batched interface (BRAM/DMA batch of syndromes per invocation), then run
Monte-Carlo campaigns (millions of shots across a (p, legs, iters) grid) through the silicon
itself, comparing LER curves against the software golden / Aer references — the FPGA becomes
the accelerator of its own qualification.

**Acceptance Criteria**
- [ ] Batched syndrome-in / correction-out path with ≥100× harness-throughput improvement over
      the AXI-Lite per-word runner.
- [ ] MC campaign ≥10⁶ shots per operating point on ≥3 physical error rates; RTL LER within the
      statistical band of the software golden at every point.

-----

### [Q7-07] Non-convergence fallback policy (valid_flag=0 path)

**Labels:** `area:fpga`, `type:feature`, `priority:low`
**Milestone:** Phase Q7
**Estimate:** M
**Depends on:** Q7-02

**Description**
Relay-BP occasionally fails to converge (valid_flag=0: the best-kept decision may still violate
the syndrome). A deployable decoder needs a defined policy: report-and-flag, cheap
post-processing (e.g. OSD-lite on the residual), or retry with different disorder. Measure the
non-convergence rate across operating points (feeds on Q7-06's campaign), evaluate candidate
fallbacks in software first, and specify (implement only if the rate demands it) the hardware
path.

**Acceptance Criteria**
- [ ] Non-convergence rate quantified per operating point.
- [ ] Fallback policy chosen with data (incl. do-nothing-but-flag if rates are negligible); the
      LER impact of the chosen policy measured in software.
