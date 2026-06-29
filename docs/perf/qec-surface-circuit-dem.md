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
| **circuit-level** | below threshold to `p ≈ 0.008`; above by `p ≈ 0.010` | **~0.8–0.9 %** |

The circuit-level threshold is **~3× lower** than the phenomenological one — the expected, important
qualitative result: once every gate is a fault site (and hook errors appear), the code tolerates far
less physical noise. (Absolute values are for the *unweighted* Union-Find decoder and this model's
conventions — idle included, pre-gate depolarizing; a weighted MWPM would sit somewhat higher. The
circuit-vs-phenomenological *ratio* is the robust takeaway and is consistent with the literature.)

Reproduce:

```bash
cargo run --release -p aleph-qec --example qec_threshold -- uf analytic 100000 2024 circuit
cargo run --release -p aleph-qec --example qec_threshold -- uf analytic 100000 2024 phenom
```

## Scope / follow-ups

- Memory-Z (`X`-sector) only, matching the existing surface track; memory-X is the mirror image.
- The DEM is graphlike, so it drops straight into the RTL matching-graph generator
  (`qec_surface_uf_graph`) and the board-free co-sim (Q6-21) — wiring a **circuit-level** RTL graph +
  co-sim noise mode is the natural next step (it lands once Q6-21 merges; the hook-error edges give
  the FPGA decoder a more realistic graph than the phenomenological one).
