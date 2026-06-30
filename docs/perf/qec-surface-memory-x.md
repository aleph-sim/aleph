# Memory-X mirror for the surface code

**Status: done.** The surface-code track had only the **memory-Z** experiment (measure Z-stabilizers,
detect `X` errors, store logical-`Z`). This adds the **memory-X** mirror — the Hadamard-dual — so both
logical bases are covered as in a real experiment.

## What it adds (`crates/aleph-qec/src/surface.rs`)

`SurfaceCode::memory_x_experiment(rounds)` builds the dual of `memory_z_experiment`:

- data and X-ancillas prepared in `|+⟩` (via `H`);
- each X-check applies `CX(ancilla → data)` (ancilla is the **control**);
- the X-ancilla is read in the X basis (`H`, `Z`-measure, reset, `H` for the next round);
- final data readout in the X basis (`H`, then measure);
- detects `Z` data errors; stores the logical-`X` observable (data column 0).

Both noise models come for free, branching on the detected sector (`X` for memory-Z, `Z` for
memory-X):

| API | memory-Z | memory-X |
|---|---|---|
| `phenomenological_mechanisms` | `X` on data + meas flip | `Z` on data + meas flip |
| `circuit_level_mechanisms` | `X`-sector (`4/15`, `2/3`) | **full depolarizing** (see below) |
| `stim_program` / `stim_program_circuit_level` | emit `X_ERROR` / X-sector | emit `Z_ERROR` / full `DEPOLARIZE` |

**Why full depolarizing for circuit-level memory-X.** Memory-Z has no single-qubit gates, so its
detected (`X`) sector is fixed and the compact sector-grouped enumeration (`4/15` per CNOT, `2/3` per
idle) is exact. Memory-X interleaves `H` gates, which mix the `X`/`Z` sectors — the grouping no longer
holds. So memory-X enumerates the *full* depolarizing channel (all 15 two-qubit / 3 single-qubit
Paulis) at each CNOT, `H`, and idle, and lets `build_dem` drop the components that flip no detector.
The result reduces to a graphlike DEM (the `X`-component of any error flips no X-stabilizer detector).

## Correctness

- **Graphlike + sized**, `d∈{3,5}`, `rounds∈{1,3}` — both phenomenological and circuit-level memory-X
  DEMs have `rounds·nx + nx` detectors, one observable, and ≤2 detectors per error
  (`memory_x_dem_is_graphlike_and_sized`, `memory_x_circuit_level_dem_graphlike_and_sized`).
- **Stim oracle, edge-for-edge `< 1e-9`** (stim 1.16.0, `d∈{3,5}`, `rounds∈{1,2,3}`):
  `memory_x_dem_matches_stim` (phenomenological) and `memory_x_circuit_level_dem_matches_stim`
  (circuit-level) in `tests/surface_dem_stim_oracle.rs`. Because Stim refuses to build a DEM with
  non-deterministic detectors, a clean match also gates the memory-X circuit's determinism (the
  prep / `H` / X-basis-readout wiring).

```text
STIM_PYTHON=/path/to/stimvenv/bin/python \
  cargo test -p aleph-qec --test surface_dem_stim_oracle memory_x -- --ignored
```

## Threshold

`qec_threshold` gained a `basis ∈ {z, x}` argument. By code symmetry the memory-X threshold matches
memory-Z; measured (UF, `rounds=d`) it crosses at `p ≈ 0.023–0.025` phenomenological, the same as
memory-Z within Monte-Carlo noise — the expected validation.

```bash
cargo run --release -p aleph-qec --example qec_threshold -- uf analytic 100000 2024 phenom x
cargo run --release -p aleph-qec --example qec_threshold -- uf analytic 100000 2024 circuit x
```

## Scope

Single logical observable (one X-logical), matching memory-Z. The memory-X DEM is graphlike, so it
also drops into the RTL matching-graph generator / board-free co-sim if a memory-X FPGA decode is ever
wanted (not wired by default — memory-Z is the decoder track's working basis).
