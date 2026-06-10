# P4-08 — v0.1 public release: design

**Date:** 2026-06-10
**Issue:** #46 (`[P4-08] v0.1 public release — benchmark report + Python bindings`)
**Status:** approved by repo owner (this session)

## Decisions (locked by repo owner)

| Decision | Choice |
|---|---|
| PyPI package name | `aleph-sim` (`aleph` is taken on PyPI by the OCCRP data tool). Python module stays `import aleph`. |
| PyPI publish | **Deferred.** v0.1 ships as a GitHub release with wheels + CLI binaries attached. PyPI/TestPyPI is a follow-up. |
| Python API scope | Full `Circuit` builder + `run()` (not just QASM-string entry). |
| Release artifact build | `release.yml` GitHub Actions workflow on tag push `v*`, GitHub-hosted runners only (never the self-hosted EPYC bench runner). |
| Version | Workspace `0.0.0` → `0.1.0`; git tag `v0.1.0`. |

**Honest AC adaptation:** BACKLOG AC "`pip install aleph` works" is satisfied as
"`pip install <wheel downloaded from the GitHub release>` works in a clean venv".
`pip install aleph-sim` (from PyPI) is explicitly deferred; the PR body and the
issue closing comment must say so.

## 1. Python API (`crates/aleph-py`) — the main code deliverable

All new code lives inside the existing feature-gated `mod py` (`--features python`,
pyo3 0.22, abi3-py312). New dependency: `aleph-parser` (path dep, normal section —
it is already in the workspace).

### `Circuit` pyclass

Wraps `aleph_ir::Circuit`. Constructor `Circuit(n_qubits: int)`.

Gate methods, mirroring `aleph_core::Gate` variants the parser/IR support
(each validates qubit indices via the existing IR error paths and raises
`ValueError` with the Rust error message):

- 1q: `h, x, y, z, s, sdg, t, tdg` (`fn name(&mut self, q: u32)`)
- 1q parametric: `rx, ry, rz, p` (θ: f64; `p` = `Gate::Phase`), `u3(theta, phi, lam, q)`
- 2q: `cx, cz, swap, iswap` (`(q0, q1)`; `cx` = `Gate::Cnot` `[control, target]`)
- 2q parametric: `crx, cry, crz` (direct `Gate` variants), `cp(theta, control,
  target)` = `Gate::Phase(θ)` with one external control via
  `GateInstance::controlled` — the exact construction `benches/src/lib.rs`'s
  `qft_circuit` builder uses (the parser has NO `cp` by name; corpus QASM is
  transpiled to `p`+`cx`)
- 3q: `ccx` (= `Toffoli`), `ccz`
- `measure(qubit, clbit)`, `barrier()`
- properties: `num_qubits`, `num_clbits`, `num_gates` (gate-count via existing IR helpers)

Class/static constructors:

- `Circuit.from_qasm(source: str)` → `aleph_parser::parse`
- `Circuit.from_qasm_file(path: str)` → read file + `parse`

Parse errors raise `ValueError` carrying the parser's line/col message.

### `run()` function and `RunResult` pyclass

```python
run(circuit, *, shots=1024, backend="sv", seed=None) -> RunResult
```

- `backend="sv"` → `NaiveSvBackend` via `aleph_backend::run_optimized` (the
  optimized driver — P4-01..06 precedent; the oracle for it already exists).
- `backend="mps"` → `MpsBackend` (existing `with_seed` / default bond cap as in
  the QAOA bindings).
- `backend="stab"` → `StabilizerBackend` (`aleph-stab`, new dep for aleph-py);
  non-Clifford circuit → `ValueError` with the backend's error message (the
  backend already rejects non-Clifford gates; do not pre-screen in Python).
- `seed` controls both backend RNG (measure collapse) and the sampling phase;
  `None` → OS entropy (mirror CLI semantics in `aleph-cli/src/exec.rs`).
- Sampling: final-state `Backend::sample(state, shots)` exactly like the CLI
  (`exec.rs` is the reference implementation to mirror, including its
  measured-circuit handling). No new sampling semantics.

`RunResult`:

- `.counts() -> dict[str, int]` — bitstring keys, qubit 0 leftmost (the existing
  CLI/`format_counts` convention; document it in the docstring).
- `.statevector() -> list[complex]` — SV backend only; `ValueError` on mps/stab.

Existing `PauliSum`, `hea_energy`, `qaoa_energy` stay untouched.

### Tests

- Rust side: existing `crate_loads` smoke stays; binding logic that can be unit
  tested without Python (e.g. counts formatting helper, backend dispatch enum)
  gets plain Rust tests.
- Python side: `scripts/python/test_aleph.py` (stdlib `unittest`,
  `skipUnless(importable aleph)` — same pattern as `scripts/vqe/test_vqe.py`):
  Bell-state counts (two keys, ≈50/50, exact with seed), GHZ on all three
  backends agree, `from_qasm` on a committed oracle fixture matches builder,
  statevector of `h(0)` = [1/√2, 1/√2], non-Clifford on `"stab"` raises,
  deterministic seed reproducibility.
- These Python tests are NOT in CI (CI has no maturin step today and we are not
  adding one in this ticket); they gate the release manually — run locally and
  on the clean-env verification step. Rust `cargo test --workspace` stays the CI
  gate (pyo3 still feature-gated off by default).

## 2. Packaging metadata

`crates/aleph-py/pyproject.toml`:

- `name = "aleph-sim"`, `version = "0.1.0"` (keep in lockstep with workspace
  version manually; no automation this ticket)
- `description`, `readme` (crate-local README.md, new, short — points at repo),
  `license = "MIT"`, `requires-python = ">=3.12"`, classifiers (Rust, Science,
  3.12+), `[project.urls]` → repo + docs/perf/v0.1.md
- `[tool.maturin] module-name = "aleph"` stays.

Workspace `Cargo.toml`: `version = "0.0.0"` → `"0.1.0"` (all crates inherit).

## 3. Benchmark report `docs/perf/v0.1.md`

Consolidation of **already-measured** numbers — no new measurements, no EPYC
run. Hand-written markdown (no generator script; it is a one-off narrative doc,
unlike the per-family generated reports):

- Headline summary table: Tier-1 + Phase-4 algorithm families vs their oracle
  baselines — QFT/Grover/QPE/VQE/QAOA/Sycamore vs Qiskit Aer (single-thread
  EPYC, from `phase4.md`/`qaoa.md`), surface code vs Stim (`surface_code.md`),
  MPS 128q shallow demo (`mps_100q.md`).
- Methodology section: single-thread both sides, EPYC 8124P, corpus-QASM
  shared-input discipline, links to per-phase reports for full detail.
- Honest-caveats section (verbatim spirit from the source reports): Sycamore
  3.1–5.3× slower than Aer (no structure for fusion), surface code 1.64× Stim
  at d=11, VQE ratios overhead-bound at tiny n, MPS crossover >14q for QAOA.
- Backend feature matrix (SV/MPS/stabilizer: qubit ranges, what they're for).

Every number in v0.1.md must be traceable to an existing committed report —
cite the source file per table.

## 4. README rewrite

- Status line: Phases 0–4 complete; link ROADMAP for Phase 5+ (GPU) plans.
- Fix stale "Rust 1.85+" → 1.89.
- Add Python quickstart: install from the GitHub-release wheel URL
  (`pip install https://github.com/ruslan-splynx/aleph/releases/download/v0.1.0/<wheel>`),
  then a builder example (Bell counts) + `from_qasm` one-liner. Note PyPI is
  coming later.
- Keep/refresh the existing CLI quickstart.
- Short backends table + perf-highlights paragraph linking `docs/perf/v0.1.md`.

## 5. Community files

- `CONTRIBUTING.md`: build/test/lint commands, PR workflow (branch naming,
  one-issue-one-PR, CI gates), testing requirements (unit/property/oracle),
  distilled from CLAUDE.md — written for an external contributor, no internal
  jargon (no EPYC/bench-server references).
- `CODE_OF_CONDUCT.md`: Contributor Covenant v2.1, contact = repo owner's
  GitHub profile (no email published).

## 6. `release.yml`

Trigger: `on: push: tags: ['v*']`. Jobs (GitHub-hosted only):

1. `build-linux` (`ubuntu-latest`, x86_64): `cargo build --release -p aleph-cli`
   → `aleph-v{ver}-x86_64-unknown-linux-gnu.tar.gz`; maturin wheel
   (`maturin build --release --features python -m crates/aleph-py/Cargo.toml`)
   → one abi3 manylinux wheel.
2. `build-macos` (`macos-latest`, arm64): same → darwin tar.gz + macOS wheel.
3. `release`: needs both; `softprops/action-gh-release` (or `gh release create`)
   attaching all artifacts, auto-generated notes + hand-written intro pointing
   at v0.1.md. Release created as **draft** — repo owner publishes after the
   clean-env verification passes.

No PyPI step in this ticket. The workflow must NOT request the self-hosted
runner label anywhere.

Pre-merge validation of the workflow itself: `act`-style dry runs are not
available — instead the PR includes a `workflow_dispatch` trigger so the
workflow can be smoke-tested from the branch before tagging (artifacts only,
release step skipped unless a tag ref).

## 7. Release execution order

1. PR `[P4-08] v0.1 public release` (Closes #46) with everything above; CI
   green; merge after review.
2. Local sanity: `maturin build --release --features python` on macOS; clean
   `uv venv` → install wheel → run `scripts/python/test_aleph.py` + README
   quickstart snippets verbatim.
3. Tag `v0.1.0` on the merge commit, push tag → `release.yml` runs → draft
   release with 4 artifacts (2 tar.gz + 2 wheels).
4. Clean-env verification on both platforms (macOS local; Linux on a throwaway
   box or container): download wheel from the draft release, `pip install`,
   tutorial runs.
5. Publish the release; tick ACs on issue #46 with the honest PyPI-deferred
   note.

## Acceptance criteria mapping

| BACKLOG AC | How met |
|---|---|
| `pip install aleph` works | Adapted: `pip install <release wheel>` in clean venv works on Linux + macOS (PyPI deferred — disclosed). |
| Python quickstart works | README quickstart executed verbatim in the clean env (step 4 above). |
| Benchmark report published | `docs/perf/v0.1.md` merged + linked from README + release notes. |
| GitHub release tagged | `v0.1.0` tag + published release with binaries and wheels. |

## Out of scope

- PyPI / TestPyPI publishing (deferred by owner decision).
- Windows binaries/wheels.
- CI maturin/pytest job for the Python tests (release-gated manually).
- New benchmark measurements.
- `Backend` as a Python class hierarchy — backend is a string parameter; a
  richer API can come with v0.2.
