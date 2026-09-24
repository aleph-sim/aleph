# Q5-04 / Q5-05 — Circuit-level DEM for the gross code + relay-BP+OSD decoder

**Issues:** Q5-04 (circuit-level DEM) + Q5-05 (relay-BP+OSD decoder, per-cycle threshold).
**Phase:** Q5 (qLDPC frontier).
**Depends on:** Q5-01 (BB code construction), Q5-02 (BP+OSD), Q5-03 (relay-BP).
**Status:** done.

## What and why

Q5-01 shipped `BBCode::code_capacity_dem` — a **code-capacity** model: one perfect syndrome round,
ideal measurements, an independent `Z` error per qubit (a 3-detector hyperedge). That is the right
entry point for studying the decoder, but it is *not* the noise a real machine sees, and the Q5-01
brief explicitly called for "a DEM under circuit-level noise." Q5-02/Q5-03 (BP+OSD, relay-BP) then
benchmarked against the code-capacity DEM only. Q5-04 closes that gap.

`BBCode::circuit_level_dem(rounds, noise)` builds the **circuit-level** DEM: it lays down the actual
syndrome-extraction *circuit* for the gross code — with faulty gates, faulty measurements, and idle
errors — and compiles the resulting space-time error mechanisms into a [`DetectorErrorModel`] the
existing decoders consume unchanged.

### The depth-7 syndrome circuit

The hard part of a circuit-level model for a qLDPC code is the syndrome-extraction schedule. Each
gross-code check has weight 6 and each data qubit sits in 6 checks, so a naïve schedule needs depth
12 and risks **hook errors** — a faulty CNOT spreading an error that corrupts a stabiliser
measurement. Bravyi et al. ([arXiv:2308.07915](https://arxiv.org/abs/2308.07915)) give a **depth-7**
schedule that measures all `X`- and `Z`-checks in one cycle without mutual disturbance. We reproduce
it exactly from the authors' reference implementation
([sbravyi/BivariateBicycleCodes](https://github.com/sbravyi/BivariateBicycleCodes)):

- CNOT order `sX = [idle, 1, 4, 3, 5, 0, 2]`, `sZ = [3, 5, 0, 1, 2, 4, idle]` (the monomial-neighbour
  index each check couples to per round).
- Qubit-labelling convention matching the reference: `X`-check `c` couples to `nonzero(A_k[c,:])`
  (a forward shift); `Z`-check `c` to `nonzero(B_k[:,c])` (a backward shift). Getting this convention
  *and* the measurement staggering right is what makes the schedule non-disturbing — see below.
- Measurement staggering: the `Z`-checks are measured at round 6 **before** that round's `X`-check
  CNOTs. Measuring them after would let a round-6 `X`-CNOT spread an error that is pulled back as a
  `Z` hook onto an `X`-ancilla at the `Z`-measurement, corrupting the `X`-stabilisers.

`BBCode::memory_x_experiment(rounds)` runs a `rounds`-cycle **memory-X** experiment: data prepared in
`|+⟩^n` (a `+1` eigenstate of every `X`-stabiliser and logical `X`), `rounds` cycles of the depth-7
circuit, then a transversal `X` readout. Detectors are the `X`-check round differences (plus a final
block reconstructed from the data readout); the observables are the `k = 12` logical-`X` operators.
It is the `Z`-error sector — the circuit-level analogue of `code_capacity_dem`'s `Z`-noise / `X`-check
model.

### The noise model

Bravyi et al.'s circuit-level depolarizing model, projected to the `Z`-error sector
([`CircuitNoise`]): each CNOT contributes `Z(control)`, `Z(target)`, `Z(control)Z(target)` at
`4/15·p` each (the `Z`-shadow of a two-qubit depolarizing channel); each idle data qubit a `Z` at
`2/3·p`; each `X`-basis preparation and measurement a basis flip at `p`. `CircuitNoise::uniform(p)`
sets every rate to `p`.

## Correctness — Stim oracle

The DEM is verified **edge-for-edge against Stim** (`tests/bb_circuit_dem_stim_oracle.rs`): we emit
the identical circuit + noise as a Stim program, let Stim compile its `detector_error_model`, and
compare support → probability. For `[[72,12,6]]` at rounds ∈ {1,2,3} with non-uniform rates, every
edge matches to **< 1e-9**. This is also the determinism gate: Stim refuses to build a DEM if any
detector is non-deterministic in the noiseless circuit, so a clean build certifies the schedule
measures both stabiliser types without disturbance.

(Note: the Pauli-frame sampler `aleph_stab::sample_noisy` used to cross-check the *surface*-code DEM
does **not** apply here — the BB memory-X circuit has genuinely random noiseless measurements
(`Z`-ancillas on `|+⟩^n`, transversal `X` readout), which the frame sampler cannot reference. Only a
full Clifford simulator can validate it, hence the Stim oracle.)

## DEM structure

`cargo run --release -p aleph-qec --example qec_q5_circuit_dem`. Data:
`docs/perf/data/qec-q5-circuit-dem.{csv,log}`.

| code | rounds | detectors | observables | mechanisms |
|------|--------|-----------|-------------|------------|
| [[144,12,12]] gross | 12 | 936 | 12 | 8784 |

Versus the code-capacity DEM (72 detectors, 144 mechanisms), the circuit-level model is ~13× larger
in detectors and ~60× in mechanisms — a genuine space-time hypergraph.

## Results — logical error rate (Q5-04 baseline + Q5-05 decoder)

> **Re-measured 2026-09-24 (#503, #505).** The first version of this section was measured with an OSD
> whose columns were ordered by `|LLR|` instead of posterior LLR, which made OSD far weaker (and on
> some DEMs worse than BP). It reported BP+OSD at 2.7e-2 and relay-BP+OSD at 1.0e-3 at p=0.002, and a
> ~0.3 % threshold. The numbers below are from the fixed decoder; the grid is extended to p=0.008
> (EPYC 8124P, `docs/perf/data/qec-q5-circuit-dem-ext.{csv,log}`) because the crossing moved above the
> original 0.3 % grid edge.

1000 shots/point, normalised min-sum (α=0.875), OSD combination-sweep order 12, `rounds = d`,
uniform noise. **Q5-05** added `RelayBpOsdDecoder` — relay-BP's (Q5-03) disordered-memory soft
output fed into OSD's combination sweep (Q5-02) — the strongest decoder in this crate.

**Gross code (d=12): BP vs BP+OSD vs relay-BP+OSD.** Both OSD decoders clear every shot up to
p=0.002; they tie at 0.003, and from p=0.004 relay-BP+OSD is 1.15–1.6× below BP+OSD:

| p | BP | BP+OSD | **relay-BP+OSD** |
|------|------|--------|------------------|
| 0.0005 | 2.0e-3 | 0 | **0** |
| 0.001  | 1.9e-2 | 0 | **0** |
| 0.0015 | 3.7e-2 | 0 | **0** |
| 0.002  | 5.9e-2 | 0 | **0** |
| 0.003  | 1.56e-1 | 2.0e-3 | **2.0e-3** |
| 0.004  | 2.82e-1 | 1.9e-2 | **1.4e-2** |
| 0.005  | 5.19e-1 | 1.06e-1 | **6.5e-2** |
| 0.006  | 7.48e-1 | 2.91e-1 | **2.23e-1** |
| 0.007  | 9.23e-1 | 5.65e-1 | **4.59e-1** |
| 0.008  | 9.81e-1 | 8.29e-1 | **7.20e-1** |

**Code-size comparison (relay-BP+OSD), [[72,12,6]] vs [[144,12,12]], per-round metric.** A d=12
memory runs 12 rounds vs d=6's 6, so the fair comparison is the logical error rate **per round**,
`ε = 1 − (1 − p_L)^(1/rounds)`. (The first version used the linearisation `p_L / rounds`, which is
fine at small `p_L` but understates the per-round rate as `p_L` grows — here it would still call
d=12 the winner at p=0.008, where the exact per-round rates are equal.)

| p | d=6 `p_L` | d=12 `p_L` | d=6 per-round | d=12 per-round | larger code |
|------|-----------|------------|---------------|----------------|-------------|
| 0.0015 | 1.0e-3 | 0 | 1.7e-4 | 0 | **wins** |
| 0.002  | 2.0e-3 | 0 | 3.3e-4 | 0 | **wins** |
| 0.003  | 1.2e-2 | 2.0e-3 | 2.0e-3 | **1.7e-4** | **wins** |
| 0.004  | 4.5e-2 | 1.4e-2 | 7.6e-3 | **1.2e-3** | **wins** |
| 0.005  | 9.9e-2 | 6.5e-2 | 1.7e-2 | **5.6e-3** | **wins** |
| 0.006  | 2.03e-1 | 2.23e-1 | 3.7e-2 | **2.1e-2** | **wins** |
| 0.007  | 3.03e-1 | 4.59e-1 | 5.8e-2 | **5.0e-2** | **wins** |
| 0.008  | 4.69e-1 | 7.20e-1 | 1.00e-1 | 1.01e-1 | tie |

The crossing sits at **p ≈ 0.007–0.008** → a circuit-level threshold of **~0.7–0.8 %** with
relay-BP+OSD. Statistics are 1000 shots per point, so this is not a precise location: the d=12
advantage is clear through p=0.006, marginal at p=0.007 (the 95 % per-round intervals overlap), and
gone at p=0.008.

### Honest positioning vs the literature

Bravyi et al. report a circuit-level threshold near **~0.7%** for the gross code. With the OSD
ordering fixed, relay-BP+OSD on our Stim-verified DEM lands **at that level** (~0.7–0.8 % by the
two-distance per-round crossing), where the first version of this report found ~0.3 % and attributed
the gap to decoder tuning. The remaining caveats are methodological: two distances, 1000 shots per
point, uniform (not SI1000) noise, and a crossing estimate rather than a finite-size-scaling fit.

## Build cost

`build_dem` is the bottleneck (one symbolic Pauli propagation per mechanism through the full
`rounds`-deep circuit). The mechanism propagations are independent, so the loop is parallelised with
`rayon`: the gross `d=12` DEM build dropped from ~76 s to ~14 s on a 10-core M4 (5.5×). The Stim
oracle and DEM values are unchanged (the merge is order-stable).

## Files

- `crates/aleph-qec/src/bivariate_bicycle.rs` — `memory_x_experiment`, `circuit_level_dem`,
  `CircuitNoise`, `BBMemoryExperiment`, the depth-7 schedule (`SX`/`SZ`), and unit tests.
- `crates/aleph-qec/tests/bb_circuit_dem_stim_oracle.rs` — the Stim edge-for-edge oracle.
- `crates/aleph-qec/src/relay_bp.rs` — `RelayBpOsdDecoder` (Q5-05) + relay-BP `decode_soft`.
- `crates/aleph-qec/src/osd.rs` — `OsdDecoder::correction_from_soft` (consume external soft info).
- `crates/aleph-qec/examples/qec_q5_circuit_dem.rs` — logical-rate curves + per-cycle code-size comparison.
- `crates/aleph-qec/src/builder.rs` — parallelised `build_dem`.
- `docs/perf/data/qec-q5-circuit-dem.{csv,log}` — committed run.
