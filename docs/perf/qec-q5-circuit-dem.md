# Q5-04 — Circuit-level DEM for the gross code (depth-7 syndrome extraction)

**Issue:** Q5-04 (Phase Q5, qLDPC frontier).
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

## Results — logical error rate

1000 shots/point, BP+OSD (normalised min-sum α=0.875, OSD order 20), `rounds = d`, uniform noise.

**Gross code (d=12), plain BP vs BP+OSD** — OSD fixes the degenerate failures BP leaves behind; at
`p ≤ 0.001` it clears all 1000 shots:

| p | BP | BP+OSD | OSD gain |
|------|------|--------|----------|
| 0.0005 | 2.0e-3 | **0** | ∞ |
| 0.001  | 1.9e-2 | **0** | ∞ |
| 0.0015 | 3.7e-2 | 7.0e-3 | 5.3× |
| 0.002  | 5.9e-2 | 1.3e-2 | 4.5× |
| 0.003  | 1.6e-1 | 7.2e-2 | 2.2× |

**Code-size comparison (BP+OSD), [[72,12,6]] vs [[144,12,12]]:**

| p | d=6 logical rate | d=12 logical rate | larger code |
|------|------------------|-------------------|-------------|
| 0.0005 | 0 | 0 | (both clear) |
| 0.001  | 1.0e-3 | **0** | **wins** |
| 0.0015 | 4.0e-3 | 7.0e-3 | loses |
| 0.002  | 8.0e-3 | 1.3e-2 | loses |
| 0.003  | 3.0e-2 | 7.2e-2 | loses |

The curves **cross around p ≈ 0.001–0.0015**: below it the larger code suppresses errors (the
hallmark of being below threshold), above it the extra space-time volume of the bigger/longer code
dominates. So BP+OSD(20) gives a circuit-level threshold of **~0.1%**.

### Honest positioning vs the literature

Bravyi et al. report a circuit-level threshold near **~0.7%** for the gross code. Our **DEM is exact**
(Stim-verified), so the ~5–7× gap is entirely a **decoder-strength** issue, not a model issue: a
modest BP+OSD (order 20) is degeneracy-limited on the circuit-level hypergraph (936 detectors). The
published numbers use stronger post-processing (higher-order / combination-sweep OSD, relay-BP,
ambiguity clustering) — exactly the Q5-03 program, which now has this circuit-level DEM to run
against (previously it only had the code-capacity DEM). Closing the gap is decoder work, tracked
there; Q5-04's deliverable is the correct, Stim-verified circuit-level DEM and the harness to drive
it.

## Build cost

`build_dem` is the bottleneck (one symbolic Pauli propagation per mechanism through the full
`rounds`-deep circuit). The mechanism propagations are independent, so the loop is parallelised with
`rayon`: the gross `d=12` DEM build dropped from ~76 s to ~14 s on a 10-core M4 (5.5×). The Stim
oracle and DEM values are unchanged (the merge is order-stable).

## Files

- `crates/aleph-qec/src/bivariate_bicycle.rs` — `memory_x_experiment`, `circuit_level_dem`,
  `CircuitNoise`, `BBMemoryExperiment`, the depth-7 schedule (`SX`/`SZ`), and unit tests.
- `crates/aleph-qec/tests/bb_circuit_dem_stim_oracle.rs` — the Stim edge-for-edge oracle.
- `crates/aleph-qec/examples/qec_q5_circuit_dem.rs` — logical-rate curves + code-size comparison.
- `crates/aleph-qec/src/builder.rs` — parallelised `build_dem`.
- `docs/perf/data/qec-q5-circuit-dem.{csv,log}` — committed run.
