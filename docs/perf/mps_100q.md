# MPS 100+ Qubit Shallow-Circuit Demo (P3-10)

Closes the ROADMAP §7 Phase-3 exit metric — "MPS handles 100+ qubit shallow
circuits" — with a measured, validated number. Previously the claim rested on
the 1024-qubit architectural cap; shipped tests topped out near 50 qubits.

## Setup

- Machine: AMD EPYC 8124P 16-core (aleph-bench-server), idle-verified before
  measurement (load 0.00, no competing CI jobs).
- rustc 1.95.0, `--release` profile, commit `9cc1901`.
- Circuit: non-Clifford NN brickwork, n=128, 6 brick layers
  (CNOT·Rz·CNOT bricks alternating even/odd bonds + Rx mixer walls),
  deterministic per-qubit angles. 2,039 gates total (381 bricks + walls).
- Backend: `MpsBackend`, χ=64, seed 0, single-threaded.

## Result

| n   | depth | χ  | wall time (run) | truncation error | max bond reached |
|-----|-------|----|-----------------|------------------|------------------|
| 128 | 6     | 64 | **10.3 ms**     | 1.07e-13         | 8                |

(Local Apple M-series for reference: 8.6 ms. Debug build: ~0.8 s.)

Two structural notes on why these numbers are what they are:

- **Max bond is 8, not 64.** Brick parity alternates per layer, so any given
  chain cut is crossed by a brick only every other layer — 3 crossings at
  depth 6, hence Schmidt rank ≤ 2³ = 8. χ=64 was never binding; the run is
  exact by a wide margin, not by luck.
- **Truncation error 1.07e-13 is null-space dust, not Schmidt truncation.**
  `truncated_svd` prunes singular values ≤ 1e-7·σ_max as numerical zeros and
  books their σ² as discarded weight (`tensor.rs`). Since the bond never
  reached χ, no genuine Schmidt weight was cut. The test asserts < 1e-12.

## Validation

⟨Z_i⟩ at i ∈ {0, 1, 63, 64, 127} and ⟨Z_i Z_{i+1}⟩ at i ∈ {0, 63, 126}
match an exact state-vector reference computed on the backward light-cone
subcircuit (≤ 14 qubits at depth 6) to 1e-10. The cone extractor is itself
validated against full SV at n=20 (39 observables, 1e-12), and the brickwork
builder against full SV dense amplitudes at n=12 (1e-10). All three tests in
`crates/aleph-mps/tests/shallow_100q.rs`; the whole suite runs in well under
a second in release (~16 s in the debug profile CI uses — comfortably inside
the 30 s `#[ignore]` threshold), so it gates every CI run (no `#[ignore]`).

## Conclusion

The Phase-3 MPS exit metric is closed with evidence: a 128-qubit shallow,
low-entanglement, non-Clifford circuit runs in ~10 ms single-threaded with
validated observables. The guard lives in CI as `mps_128q_shallow_demo`
(budget + invariants + observable validation), so the metric cannot silently
regress. Cost scales as O(n·depth) SVDs of fixed ≤2χ×2χ size at fixed depth
and χ, so headroom above n=128 is linear, not exponential.
