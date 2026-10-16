# Q1-03c: Sparse Blossom allocation reuse (#511)

The matcher now retains region shells, blossom cycles, tree child buffers and
operation scratch across shots. This removes internal steady-state allocations
on the recorded d=11 phenomenological corpus. `decode()` and `decode_errors()`
still allocate their owned result once; `MwpmDecoder::decode_into()` lets callers
reuse the correction and makes the whole warmed call allocation-free.

## Implementation and lifetime

Region and tree arenas retain their slots. `region_count` and `tree_count` identify
the current shot's live prefixes; allocation resets every scalar/link and clears
the reused slot's inner vectors before publishing it. Resolution and leftover-tree
walks visit only the live prefixes. Node reset still follows the touched list.

Tree and region traversal use separate stacks because tree dissolution schedules
region subtrees. Blossom formation reuses two ancestor paths and a moved-child
buffer, builds its cycle directly in retained region storage, and preserves the
previous traversal/insertion order. Shattering returns the emptied cycle and
child buffers to their original slots. Root scheduling no longer collects a
temporary vector. Public syndrome sorting, deduplication and range filtering use
cached scratch rather than two temporary defect arrays.

This trades allocator churn for retained memory: **each buffer retains its own
high-water capacity**, which can exceed the storage needed by the most recent
shot. There is no promise that one warmup shot covers every future syndrome.
Larger/different shots may grow buffers. The existing per-thread FIFO cache still
holds at most eight decoder states; eviction, a new thread or TLS fallback needs
fresh allocation. Precision, matching weights, event ordering and tie-breaking
are unchanged. Dense and localized reference implementations are unchanged.

```rust
use aleph_qec::{Correction, DetectorErrorModel, MwpmDecoder, Syndrome};
let dem = DetectorErrorModel::parse("error(0.1) D0 L0").unwrap();
let decoder = MwpmDecoder::new(&dem).unwrap();
let mut correction = Correction::none(0);
decoder.decode_into(&Syndrome::new(1, vec![0]), &mut correction);
assert_eq!(correction.observable_flips, [true]);
```

## Allocation evidence

A test-only forwarding global allocator counts allocations and reallocations on
the calling thread. Setup and other test threads are excluded. The d=11 corpus
contains the Criterion fixture's 256 shots, 256 further shots, an empty syndrome
and unsorted/duplicate/out-of-range public input. After warming the corpus, three
full `decode_into()` passes allocate zero times. The test also compares every
returned correction/error estimate with the first pass, and exercises output
resizing across models. Debug and release checks pass.

| API | Baseline allocations / 514 calls | Candidate allocations / 514 calls |
| --- | ---: | ---: |
| `decode()` | 281,957 | 514 |
| `decode_errors()` | 281,957 | 514 |
| `decode_into()` | unavailable | 0 |

The baseline probe uses the same corpus and forwarding allocator; its converted
three-pass owned-result loop is labeled separately and is not a baseline
`decode_into()` measurement. The candidate's remaining owned-output allocations
are required by the existing return types, not matcher scratch.

## Correctness

- Existing 100,500-shot phenomenological sparse-vs-dense optimum-weight and
  correction-tie sentinel passes unchanged.
- Existing 100,500-shot error-estimate weight/syndrome invariants pass unchanged.
- Circuit-level sparse/dense differential passes (5,000 shots).
- Both ignored PyMatching oracle tests ran with their default shot counts: five
  distances, 50,000 shots each at p=0.03 and 100,000 each at p=0.006. Logical-error
  rates agree within the existing combined confidence intervals; unambiguous
  nonempty corrections meet the existing 99% agreement requirement.
- Sparse Blossom debug and release suites pass, including three 3,000-case
  property tests. The new reuse property alternates large, empty and smaller
  shots and compares exact output, weight, event count and heap-push count with
  fresh arenas. The existing serial/parallel determinism test also passes locally.
- Native Python API tests and the real Stim/PyMatching Python oracle pass.
- Workspace/all-target Clippy, QEC all-target Clippy, rustfmt and doctests pass.

These are targeted decoder checks; PR CI supplies the full workspace/platform
matrix. Review is manual self-review, not an independent reviewer.

## Timing protocol

Baseline: `5a671ae26b2eb4235d024403952c2f27988bddfb`; validated candidate code:
`56cb01f88703cd8fcfa5e85328db0de1684b4be8`. Both snapshots use the same
Rust compiler and `RUSTFLAGS="-C target-cpu=native"` on the Intel Xeon Silver 4114
server (40 logical CPUs). The recorded host is shared: services and CI were not
stopped. Measurements wait for completed builds/tests, low overall CPU use and
an idle physical core/SMT sibling, then pin both versions to the same CPU.
Per-second overall/sibling activity is retained with the results. This is a
controlled shared-host comparison, not a claim of machine-wide isolation.

Criterion uses the unchanged `benches/benches/mwpm_decode.rs` sparse arm, p=0.03,
256 fixed syndromes per iteration, distances 7, 9, 11, 13, 15 and 17. Two rounds
run in baseline/candidate/candidate/baseline order. Python uses the exact
`scripts/python/bench_qec.py` surface fixtures and timing helper through
`bench_mwpm_allocations.py`: d=5 at 20,000 shots, d=9 at 5,000 shots, p=0.003,
seed 1, one Rayon thread. Each process alternates seven aleph/PyMatching timing
samples; the two versions run in the same ABBA order. Prediction hashes pin
before/after correction equality on those circuit-level samples.

## Results (2026-10-16)

The observed d=11 speedup is **1.22×** (about 18% less latency), below the
issue's **1.5× target**. The allocation objective is met on the recorded warmed
corpus, but #511 remains open for the unachieved throughput target.

| Distance | Baseline µs/shot | Candidate µs/shot | Speedup |
| --- | ---: | ---: | ---: |
| 7 | 24.74 | 20.55 | 1.20× |
| 9 | 65.32 | 52.73 | 1.24× |
| 11 | 128.19 | 104.76 | 1.22× |
| 13 | 225.09 | 186.42 | 1.21× |
| 15 | 382.55 | 328.06 | 1.17× |
| 17 | 581.78 | 509.10 | 1.14× |

Times above are the arithmetic mean of the two runs' Criterion **mean** point
estimates, divided by 256 shots. Raw samples and each run's bootstrap intervals
are retained; no combined confidence interval or cross-hardware speedup is claimed.

| Python case | Baseline shots/s | Candidate shots/s | Speedup | Aleph / PyMatching before → after |
| --- | ---: | ---: | ---: | ---: |
| d=5 | 295,120 | 304,341 | 1.03× | 0.511 → 0.529 |
| d=9 | 35,750 | 39,831 | 1.11× | 0.494 → 0.557 |

Python numbers are medians over 14 timing samples per version/case. The small
observed d=5 gain should not be read as a strong general performance claim.
Candidate PyMatching controls measured about 576k and 71.5k shots/s respectively;
aleph still trails PyMatching on both cases. Before/after prediction hashes agree.
The Python path includes syndrome/output work beyond the matcher, and these are
lower-density circuit-level fixtures, so its gains need not match the Rust
phenomenological benchmark.

All eight timed processes used CPU 3 with SMT sibling 23. Overall CPU activity in
the logged one-second samples stayed below 10%; sibling activity stayed at or
below 1%. Other shared-host tasks were present. Two ABBA rounds reduce ordering
bias but do not eliminate all shared-system noise.

Evidence: [validation metadata](data/q1-511-validation.json),
[validation output](data/q1-511-validation.txt),
[Criterion estimates and raw samples](data/q1-511-criterion.json),
[Python samples and prediction hashes](data/q1-511-python.json),
[host-condition samples](data/q1-511-host-conditions.jsonl),
[source hashes](data/q1-511-source-sha256.txt),
[steady-state perf profiles](data/q1-511-profiles.txt),
[paired-run driver](data/q1-511-run-timings.py), and
[baseline allocation probe](data/q1-511-baseline-allocations.rs).

## Why allocation reuse alone does not close the gap

Linux `perf` was attached only after Criterion entered its steady-state profiling
phase, excluding DEM/graph construction. In the baseline, `next_at` accounts for
about 48% of sampled cycles and `run_to_matching` another 21%; in the candidate
those shares are about 58% and 22%. These are separate sampling profiles, not
elapsed-time comparisons. They show that neighbour/event processing remains the
main cost once allocator churn is removed. The earlier issue's attribution of the
whole PyMatching gap to allocations was too strong. Further work needs to profile
and improve that remaining work rather than assume another allocation tweak will
provide the missing throughput.

## Reproduction

```bash
cargo test -p aleph-qec --test mwpm_allocations -- --nocapture
cargo test --release -p aleph-qec --test mwpm_allocations -- --nocapture
cargo test --release -p aleph-qec --lib sparse_blossom
cargo test --release -p aleph-qec --lib sparse_matches_dense_weight_and_corrections
cargo test --release -p aleph-qec --lib decode_errors_invariants_on_phenomenological_shots
PYMATCHING_PYTHON=/path/to/python cargo test --release -p aleph-qec \
  --test mwpm_pymatching_oracle -- --ignored --nocapture
RAYON_NUM_THREADS=1 RUSTFLAGS="-C target-cpu=native" \
  cargo bench -p aleph-benches --bench mwpm_decode -- sparse/
RAYON_NUM_THREADS=1 python scripts/python/bench_mwpm_allocations.py --repeats 7
```

Build/install the corresponding Python extension separately for each snapshot;
do not compare a new Rust binary with a stale installed wheel.
