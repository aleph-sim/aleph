# aleph

[![CI](https://github.com/ruslan-splynx/aleph/actions/workflows/ci.yml/badge.svg)](https://github.com/ruslan-splynx/aleph/actions/workflows/ci.yml)

A high-performance quantum circuit simulator written in Rust. Designed for correctness first, with pluggable backends (state vector, MPS, stabilizer), CUDA acceleration, and a path to distributed multi-GPU execution.

> Status: **Phase 0 — Foundation**. Working end-to-end pipeline (parser → IR → naive backend → measurement) is the current milestone. See [`ROADMAP.md`](ROADMAP.md) for phases and [`BACKLOG.md`](BACKLOG.md) for issues.

## Quick start

Requires Rust **1.85+** (edition 2021).

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

### Using the `aleph` binary

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

GPU (Phase 5+):

```bash
cargo build --workspace --features cuda
```

Python bindings (Phase 4+):

```bash
cd crates/aleph-py && maturin develop --release
```

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

MIT — see [`LICENSE`](LICENSE).
