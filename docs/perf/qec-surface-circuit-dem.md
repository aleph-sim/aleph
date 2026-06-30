# Circuit-level DEM for the surface code (memory-Z)

**Status: done.** The surface-code decoder track had only a *phenomenological* noise model
(`MemoryExperiment::phenomenological_mechanisms`: one data error per round + a measurement flip). The
gross/BB code already had a full **circuit-level** DEM (Q5-04). This adds the matching circuit-level
model for the surface code, mirroring that design, so every part of the syndrome-extraction circuit —
each CNOT, idle interval, preparation, and measurement — is a fault site.

## What it adds

| API (`crates/aleph-qec/src/surface.rs`) | role |
|---|---|
| `MemoryExperiment::circuit_level_mechanisms(CircuitNoise)` | the `X`-sector circuit-level error mechanisms |
| `MemoryExperiment::circuit_level_dem(CircuitNoise)` | convenience: `build_dem` of the above |
| `MemoryExperiment::stim_program_circuit_level(CircuitNoise)` | equivalent Stim program for cross-check |
| `CircuitNoise` (moved to `builder.rs`, shared with `BBCode`) | `{p_cnot, p_init, p_meas, p_idle}` + `uniform(p)` |

The model (memory-Z detects `X` errors, so only each depolarizing channel's `X` sector survives —
`build_dem` drops mechanisms that flip no detector and no observable):

- **CNOT** — a two-qubit depolarizing channel: `X(c)`, `X(t)`, `X(c)X(t)` each at `4/15·p_cnot`. The
  `X(c)X(t)` correlated term and the lone `X(t)` on the ancilla are what create **hook errors** — the
  diagonal space-time edges that make a circuit-level DEM qualitatively harder than the
  phenomenological one and that lower the threshold.
- **Idle / storage** — single-qubit depolarizing `X` at `2/3·p_idle` on every data qubit once per
  round (at the round start).
- **Preparation** — `X` flip at `p_init` on the initial `|0…0⟩` prep of every qubit and after every
  per-round ancilla reset.
- **Measurement** — `X` flip at `p_meas` on every measurement record (ancilla and final data
  readout).

## Correctness

- **Graphlike.** With the surface code's CNOT schedule the `X`-sector circuit-level DEM is graphlike
  (every single-fault mechanism flips ≤ 2 detectors), verified for `d∈{3,5}`, `rounds∈{1,2,3}`
  (`circuit_level_dem_is_graphlike`). So MWPM / Union-Find decode it **directly** — no hyperedge
  decomposition needed.
- **Stim oracle, edge-for-edge.** `circuit_level_dem` matches Stim's `detector_error_model` for the
  same circuit + noise to `< 1e-9` on every edge, `d∈{3,5}`, `rounds∈{1,2,3}`
  (`tests/surface_dem_stim_oracle.rs::circuit_level_dem_matches_stim`, validated against stim 1.16.0).
  The error placed at a CNOT is conjugated *by* that CNOT (a pre-gate channel), matching the emitted
  Stim program, so the two agree exactly.

```text
STIM_PYTHON=/path/to/stimvenv/bin/python \
  cargo test -p aleph-qec --test surface_dem_stim_oracle circuit_level -- --ignored
```

## Threshold: circuit-level vs phenomenological

`qec_threshold` grew a `noise ∈ {phenom, circuit}` argument (the prob grid auto-switches). Union-Find
decoder, `rounds = d`, 100 000 shots/cell, seed 2024 (`p_th` = the per-distance LER crossing):

| noise model | d=3/5/7/9 logical-error rates near the crossing | threshold `p_th` |
|---|---|---|
| **phenomenological** | cross at `p ≈ 0.025` (all ≈ 0.085) | **~2.5 %** |
| **circuit-level (uniform)** | below threshold to `p ≈ 0.008`; above by `p ≈ 0.010` | **~0.8–0.9 %** |
| **circuit-level (SI1000)** | below threshold to `p ≈ 0.004`; above by `p ≈ 0.005` | **~0.4–0.5 %** |

The thresholds form the expected, important hierarchy — **phenomenological > uniform circuit-level >
SI1000** — each step adding realism lowers the noise the code tolerates: once every gate is a fault
site hook errors appear (uniform circuit), and once the long measurement/reset/idle windows cost
2–5× the gate rate (SI1000, [`CircuitNoise::si1000`], Gidney et al. arXiv:2108.10457) the budget
shrinks again. The ~0.4–0.5 % SI1000 figure matches the literature's superconducting-surface-code
threshold. (Absolute values are for the *unweighted* Union-Find decoder and this model's conventions —
idle included, pre-gate depolarizing; a weighted MWPM would sit somewhat higher. The *ratios* between
models are the robust takeaway.)

Reproduce:

```bash
cargo run --release -p aleph-qec --example qec_threshold -- uf analytic 100000 2024 phenom
cargo run --release -p aleph-qec --example qec_threshold -- uf analytic 100000 2024 circuit
cargo run --release -p aleph-qec --example qec_threshold -- uf analytic 100000 2024 circuit-si1000
```

## In the board-free co-sim

Because the DEM is graphlike it drops straight into the RTL matching-graph generator
(`qec_surface_uf_graph -- graph-circuit`) and the board-free Q6-21 co-sim
(`qec_q6_cosim … circuit`): the RTL decoder is driven from circuit-level syndromes and its
logical-error rate matched to the software UF within CI. The circuit-level graph is denser than the
phenomenological one (d=3×3: `M=49` vs 18; d=5×3: `M=165` vs 120 — the extra edges are the hook
errors). See `docs/perf/qec-q6-cosim.md` § circuit-level (`make -C hw cosim-circuit{,-3d}`).

## Scope / follow-ups

- Memory-Z (`X`-sector) only, matching the existing surface track; memory-X (`Z`-sector) is the
  mirror image — a follow-up.
