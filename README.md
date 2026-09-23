# aleph

[![CI](https://github.com/aleph-sim/aleph/actions/workflows/ci.yml/badge.svg)](https://github.com/aleph-sim/aleph/actions/workflows/ci.yml)

A high-performance quantum circuit simulator written in Rust. Designed for correctness first, with pluggable backends (state vector, MPS, stabilizer), Python bindings, and a path to CUDA acceleration and distributed multi-GPU execution.

> Status: **v0.2** — Phases 0–4.5 complete: optimized single/multi-threaded CPU state vector, MPS and stabilizer backends, Python bindings on PyPI, and **CPU parity vs the references** — every parity-matrix cell ≤ 1.2× Qiskit Aer (MT statevector + MPS) / Stim, most cells faster ([docs/perf/parity.md](docs/perf/parity.md)). Next: Phase 5 (GPU). See [ROADMAP.md](ROADMAP.md) for phases and [BACKLOG.md](BACKLOG.md) for issues.

## Quick start

Requires Rust **1.89+** (edition 2021).

```bash
# Build everything
cargo build --workspace

# Run all tests
cargo test --workspace

# Lint + format check (CI gate)
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check

# Run benchmarks
cargo bench --workspace
```

For release builds with native CPU optimizations:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release --workspace
```

## Python quickstart

Install from PyPI (the package is `aleph-sim`, the module is `aleph`;
wheels: Linux x86_64 manylinux_2_28 + macOS arm64, Python ≥ 3.10):

```bash
pip install aleph-sim
```

Wheels are also attached to each
[GitHub release](https://github.com/aleph-sim/aleph/releases) if you
prefer a pinned direct download.

```python
import aleph

c = aleph.Circuit(2)
c.h(0)
c.cx(0, 1)

result = aleph.run(c, shots=1024, seed=0)   # backend="auto" → stabilizer (Clifford)
print(result.counts())        # {'00': ~512, '11': ~512}

# statevector() needs the dense backend; "auto" sends this Clifford circuit to
# the stabilizer, which has no amplitudes — ask for "sv" explicitly:
sv = aleph.run(c, shots=1, seed=0, backend="sv").statevector()
print(sv)                     # numpy complex128 array, 4 amplitudes

# Or load OpenQASM 3.0 (from_qasm_file(path) also exists). backend= accepts the
# same names as the CLI: "auto" (default), "statevector"/"sv",
# "stabilizer"/"stab", "mps".
qasm = """OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
cx q[0], q[1];
"""
print(aleph.run(aleph.Circuit.from_qasm(qasm), backend="mps", seed=0).counts())
```

## Backends

| backend | representation | capacity | exactness | use it for |
|---------|----------------|----------|-----------|------------|
| `sv` | dense 2ⁿ complex amplitudes | ≤ 28 qubits (default cap) | exact (FP64) | any circuit that fits in memory — the general-purpose workhorse |
| `mps` | matrix product state (bond dim χ) | 100+ qubits (1024 hard cap) for shallow/local circuits | exact while χ is not binding; controlled truncation otherwise | low-entanglement circuits: shallow brickwork, nearest-neighbour dynamics |
| `stab` | CHP tableau, O(n²) bits | hundreds of qubits (65,536 hard cap) | exact | Clifford-only circuits: error-correction cycles, stabilizer states |
| `metal` | dense 2ⁿ complex amplitudes on the GPU | ≤ 28 qubits | approximate (FP32, ~1e-5 vs FP64) | large dense circuits on Apple Silicon — opt-in GPU build |
| `cuda` / `cuda-f32` | dense 2ⁿ amplitudes on an NVIDIA GPU | ≤ 30 (FP64) / ≤ 31 (FP32) in core, higher out-of-core | exact-ish (FP64) / approximate (FP32, ~1e-5) | large dense circuits on NVIDIA — opt-in GPU build |

Backend names are one shared vocabulary across the CLI (`--backend`) and Python (`backend=`): **`auto`** (default — picks from circuit structure: Clifford → stabilizer, large nearest-neighbour + shallow → MPS, else state vector), **`statevector`** (alias `sv`), **`stabilizer`** (alias `stab`), **`mps`**, **`metal`** (alias `gpu`), and **`cuda`** / **`cuda-f32`**. Under `auto`, a Clifford circuit routes to the stabilizer backend, which has no dense state vector — pass `backend="sv"` (or `--backend sv`) when you need `statevector()`.

The **`metal`** backend is an FP32 Apple Silicon GPU state vector. It is opt-in and macOS-only: build the CLI with `cargo build -p aleph-cli --features metal`, or the Python wheel with `maturin develop --features python,metal`. `auto` never selects it (FP32 accuracy + device availability), so request it explicitly. Both `sv` and `metal` expose `statevector()`.

The **`cuda`** (FP64) / **`cuda-f32`** (FP32) backends are NVIDIA GPU state vectors, opt-in and Linux-only: build the CLI with `cargo build -p aleph-cli --features cuda`, or the Python wheel with `maturin develop --features python,cuda`. The **`--precision`** flag (CLI) / **`precision=`** kwarg (Python) — `auto` (default), `f64`, `f32` — steers an `auto` GPU pick: on a CUDA build with a device, large dense circuits route to `cuda-f32` by default (2× throughput, one extra qubit of in-core reach). Circuits past the in-core cap stream **out-of-core** (the state lives in host memory, device-sized tiles flow through the GPU); the CLI runs these and reports `‖ψ‖²` (paged readout is norm-only — no sampling/expectation/statevector), and the Python binding raises with guidance to use the CLI/Rust API for them.

## Performance

v0.2 closes the CPU-parity matrix: at 16 threads with default settings on both sides, aleph's state-vector backend beats Qiskit Aer on QFT (0.81×), Grover (0.60×) and random brickwall (0.59×) at n=25 (GHZ is an allocation-bound tie at 1.03×); aleph-mps is 4.4–14× faster than Aer MPS on every measured workload; and the stabilizer backend now beats Stim on surface-code cycles at every measured distance (0.79× at d=11, after being 1.64× behind in v0.1). Honest caveats: the Grover cell is the iteration-capped fixture, distances beyond d=11 are unmeasured, and the v0.1 single-thread report ([docs/perf/v0.1.md](docs/perf/v0.1.md)) showed raw unfused random circuits 3–5× behind Aer — fusion in the default pipeline is what closes that gap. Full matrix + protocol: [docs/perf/parity.md](docs/perf/parity.md).

## Using the `aleph` binary

After `cargo build --release --workspace`, the `aleph` binary lives at
`target/release/aleph`.  Four basic invocations:

```bash
# Sample 1024 shots from a Bell-state circuit with a fixed RNG seed.
./target/release/aleph run oracle/circuits/bell_phi_plus.qasm \
  --shots 1024 --seed 0

# Print the full final state vector (capped at 10 qubits;
# use --force-statevector to opt out of the cap).
./target/release/aleph run oracle/circuits/bell_phi_plus.qasm --statevector

# Compute ⟨ψ|ZZ|ψ⟩ and ⟨ψ|XX|ψ⟩ in one run.
./target/release/aleph run oracle/circuits/bell_phi_plus.qasm \
  --expectation ZZ --expectation XX

# Single-iteration timing breakdown (parse / run / sample / total).
./target/release/aleph bench oracle/circuits/bell_phi_plus.qasm
```

Pauli strings for `--expectation` are positional: qubit 0 is the
leftmost character.  `IXZI` means X on q1, Z on q2.  Optional
`coeff*` prefix: `1.5*ZZ`, `-0.5*X`.

See `aleph --help` (and `aleph run --help` / `aleph bench --help`)
for the full flag list.

### Noise simulation

Aer-compatible noise via a `NoiseModel` (Python) or a one-parameter CLI preset.

```python
import aleph

c = aleph.Circuit(2)
c.h(0); c.cx(0, 1)

nm = aleph.NoiseModel()
nm.add_all_qubit_quantum_error(aleph.depolarizing_error(0.01, 1), ["h"])
nm.add_quantum_error(aleph.depolarizing_error(0.02, 2), ["cx"], [0, 1])
nm.add_readout_error([[0.98, 0.02], [0.03, 0.97]], 0)

print(aleph.run(c, shots=100_000, noise=nm, seed=7).counts())
```

Error factories mirror Qiskit Aer names: `depolarizing_error(p, num_qubits)`,
`amplitude_damping_error(gamma)`, `phase_damping_error(lam)`,
`pauli_error([("X", 0.1), ("I", 0.9)])`, plus `bit_flip_error` / `phase_flip_error`.
Noise runs on the state-vector backend as per-shot Monte-Carlo trajectories.

CLI presets — depolarizing on every gate, symmetric readout flip on every qubit
(repeatable; forces the state-vector backend, shots-only):

```bash
aleph run circuit.qasm --shots 4096 --noise depol:0.01 --noise readout:0.02
```

GPU (Phase 5+):

```bash
cargo build --workspace --features cuda
```

Building the Python bindings from source (instead of installing a
release wheel): `cd crates/aleph-py && maturin develop --release`.

## QEC decoding

aleph ships MWPM, Union-Find, BP, BP-OSD and relay-BP decoders for stim
Detector Error Models, usable directly or through sinter:

```bash
pip install "aleph-sim[sinter]"
```

```python
import sinter, stim, aleph.sinter
circ = stim.Circuit.generated("surface_code:rotated_memory_x", distance=5, rounds=5,
                              after_clifford_depolarization=0.003)
stats = sinter.collect(tasks=[sinter.Task(circuit=circ)], num_workers=4,
                       decoders=["aleph-mwpm", "aleph-relay-bp-osd", "pymatching"],
                       custom_decoders=aleph.sinter.decoders(), max_shots=100_000)
```

aleph's batch decode parallelizes across shots on all cores, so under
`sinter.collect(num_workers=N)` set `RAYON_NUM_THREADS=1` (or a small number)
to avoid running `num_workers × cores` threads.

Throughput (`scripts/python/bench_qec.py`, macOS-26.7-arm64-arm-64bit / arm,
Apple Silicon Mac, 10 logical CPUs; `surface d=9` reduced to 5,000 shots —
aleph relay-bp-osd is slow enough that 20,000 would take >60s — every other
row is 20,000). Every aleph decoder is measured on all 10 cores and on one
thread (`RAYON_NUM_THREADS=1`); pymatching's `decode_batch` runs on one
thread:

| DEM | decoder | threads | shots | shots/s |
|---|---|---:|---:|---:|
| surface d=5 | aleph mwpm | 10 | 20,000 | 2,196,645 |
| surface d=5 | aleph mwpm | 1 | 20,000 | 733,746 |
| surface d=5 | aleph union-find-weighted | 10 | 20,000 | 2,476,563 |
| surface d=5 | aleph union-find-weighted | 1 | 20,000 | 576,536 |
| surface d=5 | aleph bp-osd | 10 | 20,000 | 18,555 |
| surface d=5 | aleph bp-osd | 1 | 20,000 | 2,987 |
| surface d=5 | aleph relay-bp-osd | 10 | 20,000 | 4,530 |
| surface d=5 | aleph relay-bp-osd | 1 | 20,000 | 756 |
| surface d=5 | pymatching | 1 | 20,000 | 1,503,434 |
| surface d=9 | aleph mwpm | 10 | 5,000 | 202,386 |
| surface d=9 | aleph mwpm | 1 | 5,000 | 33,684 |
| surface d=9 | aleph union-find-weighted | 10 | 5,000 | 392,584 |
| surface d=9 | aleph union-find-weighted | 1 | 5,000 | 82,226 |
| surface d=9 | aleph relay-bp-osd | 10 | 5,000 | 323 |
| surface d=9 | aleph relay-bp-osd | 1 | 5,000 | 55 |
| surface d=9 | pymatching | 1 | 5,000 | 204,928 |
| color d=5 | aleph bp-osd | 10 | 20,000 | 27,827 |
| color d=5 | aleph bp-osd | 1 | 20,000 | 4,628 |
| color d=5 | aleph relay-bp | 10 | 20,000 | 6,042 |
| color d=5 | aleph relay-bp | 1 | 20,000 | 1,047 |
| color d=5 | aleph relay-bp-osd | 10 | 20,000 | 5,729 |
| color d=5 | aleph relay-bp-osd | 1 | 20,000 | 1,040 |

**Per core, pymatching is faster.** Single-threaded, aleph's dense-blossom
`mwpm` reaches 0.49× pymatching's throughput at d=5 and 0.16× at d=9, and
`union-find-weighted` 0.38× and 0.40×: aleph's per-shot matching algorithm is
asymptotically slower than PyMatching's Sparse Blossom (see
[docs/perf/qec-q1-mwpm.md](docs/perf/qec-q1-mwpm.md); the Sparse Blossom rewrite is tracked in
[#331](https://github.com/aleph-sim/aleph/issues/331) and not yet done).
**On the whole machine, aleph is faster or level.** With all 10 cores
against pymatching's one thread, `mwpm` is 1.46× pymatching at d=5 and
level (0.99×) at d=9, and `union-find-weighted` is 1.65× and 1.92×.
`decode_batch_bit_packed` releases the GIL and splits shots across cores, so
this is what a single-process caller sees; a sinter run with many workers
already uses the cores and gets the per-core ratio instead.

`ldpc`'s `SinterBpOsdDecoder` was installed but has no color-code row here:
its `compile_decoder_for_dem` raises `NotImplementedError` on this ldpc
version for a non-decomposed (hypergraph) DEM, so the benchmark skips it
rather than fail. aleph's iterative BP-family decoders (`bp-osd`, `relay-bp`,
`relay-bp-osd`) are two to three orders of magnitude lower throughput than
the one-shot matching decoders — expected, since each shot runs multiple BP
iterations (and, for `-osd`, a post-processing ordered-statistics step)
rather than a single matching pass.

## Workspace layout

```
aleph/
├── crates/
│   ├── aleph-core/      # Complex, StateVector, Gate, Circuit types
│   ├── aleph-ir/        # Circuit IR + optimization passes
│   ├── aleph-parser/    # OpenQASM 3.0 parser
│   ├── aleph-backend/   # Backend trait + naive impl
│   ├── aleph-sv/        # state vector backends (CPU, later GPU)
│   ├── aleph-mps/       # MPS tensor network backend
│   ├── aleph-stab/      # stabilizer (Aaronson–Gottesman) backend
│   ├── aleph-cli/       # `aleph` binary
│   └── aleph-py/        # PyO3 Python bindings
└── scripts/             # GitHub issue sync, etc.
```

## Project documents

- [`ROADMAP.md`](ROADMAP.md) — strategy and phase plan
- [`BACKLOG.md`](BACKLOG.md) — detailed issue specifications (source of truth)
- [`CREATE ISSUES.md`](CREATE%20ISSUES.md) — how the GitHub backlog is synced
- [`CLAUDE.md`](CLAUDE.md) — instructions for AI assistants working in this repo

## Algorithm optimization playbooks

Per-algorithm guides applying the framework from [`OPTIMIZATION GUIDE.md`](OPTIMIZATION%20GUIDE.md) to specific quantum algorithms.

### Read order

1. [`OPTIMIZATION GUIDE.md`](OPTIMIZATION%20GUIDE.md) — methodology, principles, checklists.
2. [`OPTIMIZATION CYCLE.md`](OPTIMIZATION%20CYCLE.md) — step-by-step iteration playbook.
3. Algorithm-specific playbooks below.

### Playbooks

| Algorithm                       | File                                              | Key win                           | When to consult                           |
|---------------------------------|---------------------------------------------------|-----------------------------------|-------------------------------------------|
| Quantum Fourier Transform       | [QFT.md](QFT.md)                                  | Phase polynomial fusion, AQFT     | Working on diagonal gates, QFT/QPE/Shor   |
| Grover's algorithm              | [GROVER.md](GROVER.md)                            | Specialized MCZ, diffusion fusion | Working on multi-controlled gates, search |
| Variational Quantum Eigensolver | [VQE.md](VQE.md)                                  | Symbolic params, Pauli grouping   | NISQ chemistry; **highest practical ROI** |
| QAOA                            | [QAOA.md](QAOA.md)                                | Diagonal cost-layer fusion, MPS   | Combinatorial optimization, sparse graphs |
| Random Circuits                 | [RANDOM CIRCUIT.md](RANDOM%20CIRCUIT.md)          | Generic kernel quality            | Stress testing, supremacy benchmarks      |
| Stabilizer Circuits             | [STABILIZER CIRCUITS.md](STABILIZER%20CIRCUITS.md) | Bit-packed tableau, batched shots | QEC, surface codes, Clifford-only         |

### Playbook structure

Every playbook follows the same template:

1. **Quick Reference** — algorithm at a glance.
2. **Algorithm Overview** — brief technical recap.
3. **Computational Profile** — where the time goes.
4. **Optimization Ladder** — opportunities in ROI order.
5. **Pitfalls** — algorithm-specific gotchas.
6. **Baseline Comparisons** — what to beat (Qiskit Aer / Stim / cuQuantum).
7. **Phase-by-Phase Sub-goals** — what's expected at each project phase.
8. **Success Metrics** — when an optimization PR is considered successful.
9. **References** — primary literature.

### When to add a new playbook

Add a playbook when:

- The project starts targeting an algorithm not covered (e.g., Shor, Hamiltonian simulation).
- An algorithm reveals optimization opportunities not captured by the global guide.
- Multiple PRs on the same algorithm would benefit from shared context.

Follow the template. Open a PR titled `[playbook] Add {AlgorithmName} playbook`.

### When to update an existing playbook

Update a playbook when:

- A new optimization opportunity is discovered.
- A pitfall is encountered in review or in production.
- Baseline numbers change (new external version, new reference hardware).
- A sub-goal is achieved or refined.

Open a PR titled `[playbook] Update {AlgorithmName}: {reason}`.

## License

MIT — see [`LICENSE`](LICENSE) — **except `hw/`**, which is Apache-2.0
(see [`hw/LICENSE`](hw/LICENSE) and the rationale in [`hw/README.md`](hw/README.md#licence--apache-20-not-mit)).

The split is about patents: MIT carries no patent grant, which blocks organisations that would
otherwise fabricate the decoder RTL. Apache-2.0 grants one explicitly.
