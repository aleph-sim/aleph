# P3-10 — MPS 100+ qubit shallow-circuit demo (design)

**Issue:** #123 · **Branch:** `p3-10-mps-100q-demo` · **Estimate:** S
**Goal:** close the ROADMAP §7 Phase-3 exit metric — *"MPS handles 100+ qubit
shallow circuits"* — with a measured, validated, regression-guarded number.
Shipped MPS tests top out around 50 qubits today; the ≥100-qubit claim is
unverified.

## Decisions (brainstorm outcomes)

- **Deliverable:** integration test + perf note (not a criterion bench).
  The test guards the metric in CI; the note records the measured number.
- **Official measurement:** EPYC (`aleph-bench-server`), idle-verified,
  consistent with every other `docs/perf/` report.
- **Validation strategy:** light-cone exact reference (approach A) —
  non-Clifford circuit, exact SV reference on the causal cone only.

## 1. Circuit — brickwork, n = 128, depth 6

Same style as the `nn_qaoa` bench builder. Brick layers alternate even/odd
bonds; each brick = `CNOT → Rz(θ) → CNOT` (ZZ interaction) followed by an
`Rx` mixer layer. Angles are deterministic and per-qubit distinct
(e.g. `θ_q = 0.3 + 0.05·q`) — generic, non-Clifford.

Any single chain cut is crossed by at most one 2q gate per brick layer, so
the Schmidt rank across any bond is ≤ 2^6 = 64. With χ = 64 the MPS run is
**exact**: `truncation_error() == 0`, which makes a 1e-10 comparison
legitimate rather than truncation-lucky.

## 2. Validation — generic light-cone extractor

A helper local to the test file:

```
fn light_cone_subcircuit(circuit, support: &[u32]) -> (Circuit, mapping)
```

Walk instructions **in reverse** from the observable's support set; keep a
gate iff it intersects the current support, expanding the support with the
kept gate's qubits; finally remap kept qubits to a compact `0..k` range.
For a depth-6 cone of `⟨Z_i Z_{i+1}⟩` this is ≤ 14 qubits — trivial for
`NaiveSvBackend::expectation_value`.

**The extractor validates itself**: a separate test at n = 12 checks
cone-based expectation == full-SV expectation to 1e-12, for several
observables. This answers "who validates the validator".

## 3. Test — `crates/aleph-mps/tests/shallow_100q.rs`

- Run n = 128 / depth 6 / χ = 64 to completion with an explicit wall-time
  ceiling asserted in-test (generous, ~120 s, to avoid CI flakes).
- Assert `truncation_error() == 0.0` and `max_bond_reached() ≤ 64`.
- Assert `⟨Z_i⟩` for i ∈ {0, 1, 63, 64, 127} (edges + middle) and
  `⟨Z_i Z_{i+1}⟩` for bonds {0–1, 63–64, 126–127} against the cone-SV
  reference, tolerance 1e-10.
- If the test exceeds 30 s locally, mark `#[ignore]` per CLAUDE.md (nightly
  CI runs ignored tests); expectation is it stays well under.

Test code may use `unwrap()` per project conventions.

## 4. Measurement + perf note

One EPYC session (idle-checked first per CLAUDE.md): wall time of the full
n = 128 run; optionally the n ∈ {100, 128, 256} curve since the same builder
makes it cheap. Record in `docs/perf/mps_100q.md`: machine, numbers, χ,
truncation, validation summary, and the explicit statement that the
ROADMAP §7 Phase-3 MPS exit metric is now closed with evidence. Update the
metric mention in ROADMAP/BACKLOG checkboxes where present.

## 5. Scope / non-goals

- **No `aleph-mps/src` changes** — tests + docs only.
- No multithreading or lazy-SWAP work (that is P3-09, #125).
- No criterion bench (deliverable decision above).
- PR `[P3-10] MPS 100+ qubit shallow-circuit demo`, body **`Closes #123`**
  (issue number, not PR number).

## Error handling

Library behavior unchanged. Test failures = assertion failures with the
mismatching values printed; no panics on the library path.
