# Operation-aware CUDA memory preflight (#554)

CLI and Python now check the requested operations against the actual CUDA device's
free/total memory before constructing an SV backend or allocating its state.
A reference-card qubit cap is a structural hint, not evidence that sampling fits.
The old n=30/n=31 FP32 counts requests need at least 24/48 GiB for state plus CDFs
alone, exceeding the reference card's roughly 20 GiB.

The shared pure planner lives in `aleph-backend::memory`, with no CUDA implementation
types. `aleph_cuda::memory` supplies the current executor/readout layout, circuit
scratch, and runtime capacity. Both frontends use this boundary. A validated state
can exceed the old reference-card cap on a larger device; backend allocation uses
the validated qubit count. Synthetic tests cover 8, 20 and 80 GiB budgets, both
precisions, and changing free capacity. Existing `Reach::for_kind` remains a legacy
reference-card hint for compatibility; it does not decide frontend feasibility.

## Allocation model

For N=2^n:

- State: N times 8 bytes (FP32 complex) or 16 bytes (FP64 complex).
- Reductions: two FP64 values per 256-amplitude block, plus a 16-byte accumulator.
  In-circuit measurement/reset and readout use these. Frontend zero-shot/export-only
  requests conservatively include reduction capacity because sample(0) checks norm.
- Sampling with positive shots: **two N-element FP64 CDF buffers in both precisions**,
  plus FP64 targets and u64 output indices (16 bytes per shot).
- Marginals: 8*2^k bytes for bins and 4*k for positions, with reductions. The pure
  model/audit supports this existing Rust readout; no new Python/CLI marginal API
  is introduced.
- Execution: retained matrix scratch, growth overlap, and maximum phase-polynomial
  angle/condition/offset storage. Both precisions upload FP64 matrix coefficients.
  This describes the current raw frontend executor, not a future fused plan.

The device bound sums readout-stage scratch conservatively: the retaining pool may
keep earlier freed blocks. Existing live/cached pool resources and retained Python
results already reduce runtime free memory; the policy does **not** optimistically
credit cached blocks back to the free budget. New allocator/JIT granularity and
fragmentation use headroom: max(256 MiB, 5% of device total). These are advisory
estimates, not memory reservations. Allocation/launch errors still propagate their
CUDA driver diagnostics if another tenant allocates after the check.

Host output capacity is checked independently, using Linux MemAvailable capped by
visible cgroup-v2 process/ancestor limits. Host headroom is max(64 MiB, 5% of available).
Sampling includes host targets/results and a conservative counts-bookkeeping bound:
min(shots,N)*(512+2*n) + 16*shots. Marginals include their host vector. Execution
transfer scratch is included. Amplitude export reserves up to two complex128 arrays
(32*N bytes), covering FP32 widening/FP64 conversion or cached amplitudes plus NumPy
output. This is a conservative estimate, not an allocator-level byte proof for
Python dictionaries or OS page management.

Python counts keep amplitude download lazy. The export guard runs at the explicit
statevector request, before device download/widening; a second guard covers the
independent NumPy array copy, including reads from a previously materialized cache.
A failed guard retains the device state for retry. CLI amplitude transfers are now
fallible, preserving driver/host-memory errors instead of panicking.

All dimensions/products/sums use checked arithmetic. Infeasible requests report
state, execution, reductions, sampling and marginal bytes, free/total memory and
headroom. No full-state allocation is attempted to discover the limit.

## Output semantics

The old CLI paged branch could print a note and return a norm when counts were
requested. It has been removed from frontend dispatch. A request that does not fit
fails before large allocation. Counts, expectations, marginals and amplitude export
are never replaced by a norm. The Rust paged norm-only APIs are unchanged. Callers
wanting that operation can explicitly use those APIs.

GPU backend discovery/initialization remains separate from state allocation. The
small preflight context stays alive through backend construction to reuse its
primary context; Python performs preflight/execution without the GIL, with the
non-Send pool handle staying on that worker.

## Validation

Host: RTX 4000 SFF Ada, total reported 20,989,804,544 bytes, initially free
19,413,598,208 bytes. Services remained running. No isolation, large allocations,
or intentional OOM were used, and no throughput improvement is claimed.

Each allocation audit ran in a fresh process with n=18 Bell state (qubits 0,17),
using real FP32/FP64 execution, readout and host export. The CUDA pool's live-byte
high-water counter measures actual data allocations, rather than reserved cache or
module/driver allocations. Estimates intentionally include unused gate scratch and
sometimes unused reduction capacity.

| Precision | Operation | Estimated device bytes | Measured live peak |
|---|---|---:|---:|
| FP32 | Execute | 2,115,600 | 2,097,168 |
| FP32 | Sample 64 | 6,310,928 | 6,308,880 |
| FP32 | Expectation | 2,115,600 | 2,113,552 |
| FP32 | Marginal (2 qubits) | 2,115,640 | 2,113,592 |
| FP32 | Export | 2,115,600 | 2,097,168 |
| FP64 | Execute | 4,212,752 | 4,194,320 |
| FP64 | Sample 64 | 8,408,080 | 8,406,032 |
| FP64 | Expectation | 4,212,752 | 4,210,704 |
| FP64 | Marginal (2 qubits) | 4,212,792 | 4,210,744 |
| FP64 | Export | 4,212,752 | 4,194,320 |

The audit asserts that observed peak is covered and that excess is bounded by
explicit gate/reduction slack, so a changed sampler cannot silently retain a stale
full-CDF estimate. #555 must update `CUDA_READOUT_LAYOUT` with the sampler and rerun
this audit. Pool/driver reservations are handled by runtime free memory and headroom,
not confused with these logical allocation counters.

The same audit rejects a 31-qubit FP32 counts request and verifies zero new device
allocation and zero SV backend initialization attempts. Public CLI tests reject
30/31-qubit counts in both precisions, and check small counts, statevector and ZZ
expectation. Public Python tests exercise oversize/overflow rejection, zero shots,
counts and repeated lazy export. Existing concurrent/cache/collapse GPU result
oracles pass in both precisions. Raw measurements: `data/p8-04-*.json*`.

Reproduce on a CUDA host (each case as a fresh process):

```sh
cargo build --release -p aleph-cuda --features cuda --example memory_plan
for precision in f32 f64; do
  for view in execute sample expectation marginal export; do
    target/release/examples/memory_plan "$precision" "$view"
  done
done
ALEPH_TEST_CUDA_MEMORY=1 ALEPH_TEST_GPU_BACKENDS=cuda,cuda-f32 \
  python -m unittest discover -s scripts/python -p 'test_*gpu*.py' -v
python scripts/python/audit_cuda_memory_cli.py --binary target/release/aleph
# Also run the explicit policy tests:
ALEPH_TEST_CUDA_MEMORY=1 python -m unittest discover -s scripts/python \
  -p test_cuda_memory.py -v
```

Pure budget/overflow tests run without a GPU in `cargo test -p aleph-backend --lib`.
The cgroup ancestry test runs under `cargo test -p aleph-cuda --features cuda --lib
memory::tests`; it uses a temporary filesystem fixture and no GPU.
