# #573: manual fused IR on CPU state-vector backends

Manual `UnitaryKq` gates previously bypassed matrix validation in AoS FP64,
SoA FP64 and FP32. Controls were silently ignored. The shared offsets helper
sorted target operands before assigning matrix bits, changing the meaning of
permuted operands. A 4x4 X-on-matrix-MSB gate on targets `[1, 0]` therefore
flipped qubit 0 instead of qubit 1; a false control did not suppress it.

## Contract and implementation

All three CPU backends validate width (2–5), exact square matrix length, finite
coefficients and unitarity before entering fused kernels or narrowing to FP32.
Nonempty controls return `UnsupportedGate`; implementing controlled fused kernels
is outside this change. Bounds, arity and duplicate-qubit checks still run first.
Errors leave state unchanged. Offset construction follows caller operand order;
only the separate fixed-bit list is sorted for the basis expansion walk. The same
helper is used by scalar and AVX512, including FP32.

One private owned matrix snapshot per backend amortizes validation of repeated
immutable matrices. Every hit compares width and **all coefficient values**; it
never trusts pointer identity, a hash, or an optimizer label. Controls are always
checked, even after a hit. Mutating the gate at the same address invalidates the
hit and invokes validation again. The snapshot is independent of caller lifetime.
A changed matrix is validated before replacing the snapshot; invalid inputs do
not replace it. Full validation/copy is a separate cold path. Gram validation
checks one Hermitian triangle and squared deviations, explicitly rejecting
non-finite intermediates before taking a maximum.

At most 16 KiB of matrix data is retained per backend at k=5. Replacement can
briefly own two snapshots (32 KiB); this bound is included in the noisy driver's
per-context plan. Ordinary circuits with no `UnitaryKq` allocate no snapshots.
Noisy preflight shares the validator and preserves its prior error categories;
permuted fused targets are now supported. No production unsafe code was added.

## Correctness and review

The independent dense oracle scatters each input basis amplitude according to
caller operand order and the supplied matrix. It does not use sorted offsets,
fixed-bit expansion, backend fused application or kernel helpers. Tests use dense
complex Fourier matrices with row phases, prepared complex entangled states,
widths 2–5, contiguous/disjoint/permuted targets, scalar-sized states and n=9
states that exercise AVX512 on the Xeon. FP32 k=5 remains the existing scalar
fallback; k=2/3/4 dispatch is exercised on the AVX512 host.

Malformed width/shape, zero/non-unitary matrices, NaN/infinity and finite overflow
are rejected without changing state. Mutation at the same matrix address and
adding controls after a successful cached call are regression cases. Separate
`FuseKq` tests at widths 3/5 compare generated circuits with their original gate
sequences in all three backends. Noisy sampling checks a permuted X-on-MSB gate.
The new permutation oracle fails on the baseline in all three backends and passes
on the candidate. Actual controls remain explicitly unsupported.

Manual review covered every CPU fused dispatcher, shared scalar/SIMD offsets,
validation before FP32 narrowing, snapshot ownership/content comparison, error
state preservation, transient cache allocation, noisy preflight compatibility,
and measurement provenance. This is manual review, not an independent Code Review
service result. Cache validity relies on the immutable borrow during each safe
Rust `apply_gate` call; input mutation between calls is checked again.

## Measurements and provenance

Baseline is main `c283272747877863fa4a2723421c8d710a8550b3`, after #572. The same
`fused_validation.rs` harness is compiled against each source in dedicated target
directories. Host: user-authorized `51.159.0.49`, dual Xeon Silver 4114, CPU-only.
One logical CPU (3) is pinned; its SMT sibling (23) is monitored; monitor runs on
CPU39. `RAYON_NUM_THREADS=1`, standard optimized Cargo profiles. CI and services
remain running. Per-case A/B/B/A, ten Criterion samples, 100 ms warmup and 300 ms
requested measurement; expensive cases extend automatically. A repeat requires
<=2% preflight busy time and <3% sampled external CPU overlap. This is bounded
shared-host evidence, not exclusive CPU isolation or a broad circuit speed claim.

Steady cases repeatedly apply the same sorted, valid phased Fourier gate to a
state initially in |0>, without reallocation/reset between iterations. Backend,
state and matrix construction are outside timing. Matrix is k=2/3/5, state is
n=10/20, each of AoS/SoA/FP32. Cold k=5/n=10 cases use Criterion `iter_batched`
with a fresh backend/state outside timing; the measured operation validates and
applies its first matrix. Returned backend/state are dropped outside the timed
operation. Both cache misses and hits are therefore visible, rather than hiding
first-use cost in warmup. Switching among distinct matrices can miss the single
snapshot repeatedly; there is no claim of amortization in that regime.

The initial full-validation-every-call version produced AoS n=10/k=5 medians
28.817 -> 156.432 us (5.43x slower). A three-second perf profile placed 51.57% of
self samples in `validate_unitary_kq`. This motivated the bounded snapshot and
Hermitian-triangle validation, not removal of checks. The first cache version
still showed a SoA n=20/k=3 regression. Profiling places about 90% of samples in
the unchanged AVX512 block closure; its cause is not proved by these samples.
A sorted-target fast-path experiment did not eliminate it and was removed.
The final candidate extracts full validation/copy into a cold method, keeping
matrix-reuse checking small. All intermediate valid data is retained separately.

Two measurement stages are invalidated in full: a wrong working directory built
candidate code into the baseline target, and correcting the manifest reused the
contaminated target's stale library. Both nominal baseline ELFs contained the new
`UnitaryKqCache::validate` symbol. Their source-file hashes alone did not prove
build provenance. No rows from these stages are used as before/after evidence.
The baseline was rebuilt in a brand-new target, its source hash checked against
main, absence of the cache symbol verified, and a distinct candidate ELF used.
Later candidate revisions also get new target directories. The reusable guard
now rejects identical binary hashes; manifest/source/symbol provenance checks
remain necessary because distinct hashes alone do not prove distinct code.

Raw Criterion estimates/samples, all accepted/rejected overlap attempts, valid
intermediate stages and the two explicitly invalid stages are retained under
`docs/perf/data/p8-573-*`. Final results and validation are recorded below.

## Reproduction

```sh
cargo test --locked -p aleph-sv --lib --tests
cargo test --locked -p aleph-backend --test run_optimized_oracle
cargo clippy --locked --workspace --all-targets -- -D warnings
# Use a fresh target for EACH source revision; never switch sources in it.
CARGO_TARGET_DIR=/tmp/fresh-target-for-this-revision \
  cargo bench --locked --manifest-path /absolute/source/Cargo.toml \
  -p aleph-sv --bench fused_validation --no-run
# Copy verified executables to an artifact directory as bench-base/candidate.
# Write affinity.txt and core-policy.json matching the isolated core/SMT pair.
python3 scripts/python/bench_noisy_trajectories_guard.py /tmp/artifacts --suite fused
cargo run --locked --release -p aleph-sv --example trajectory_budget
```

## Final results and limits

Final outlining also prevents snapshot-checking from being inlined into backend
dispatch. Correctness is complete, but this is a safety fix with costs, not a
universal speedup. The final shared-host matrix still records **32.6% slower SoA
n=20/k=2 and 24.7% slower SoA n=20/k=3**. Cache checking itself is not established
as their cause; the profiler samples the fused block closure. These regressions
remain actionable performance findings and must not be hidden by selecting a
better intermediate build. Other binary layouts shifted the affected backend,
including a 41.7% FP32 n=20/k=2 regression in the cold-only experiment. This
supports further codegen/layout/allocator investigation, not a proved attribution.

First-use k=5/n=10 medians are 29.019 -> 75.787 us (AoS), 28.218 -> 77.135 us
(SoA), and 58.808 -> 86.760 us (FP32). Full validation and a snapshot are new work
that the old unsafe path skipped. Repeated small-state k=5 adds roughly 15.6%
(AoS), 14.4% (SoA), and 5.6% (FP32) in this build. No throughput gain is claimed
from the apparent faster AoS cells or sub-5% differences on this shared host.
Switching among distinct matrices can repeatedly pay validation/copy cost.

| Public apply_gate case | Baseline us | Candidate us | Candidate / baseline |
|---|---:|---:|---:|
| aos/n10_k2 | 17.107 | 14.273 | 0.834x |
| aos/n10_k3 | 12.196 | 11.852 | 0.972x |
| aos/n10_k5 | 28.198 | 32.594 | 1.156x |
| aos/n20_k2 | 24856.217 | 15389.747 | 0.619x |
| aos/n20_k3 | 17818.755 | 14718.413 | 0.826x |
| aos/n20_k5 | 22542.804 | 21649.013 | 0.960x |
| cold_aos/n10_k5 | 29.019 | 75.787 | 2.612x |
| cold_fp32/n10_k5 | 58.808 | 86.760 | 1.475x |
| cold_soa/n10_k5 | 28.218 | 77.135 | 2.734x |
| fp32/n10_k2 | 7.182 | 7.389 | 1.029x |
| fp32/n10_k3 | 9.393 | 9.470 | 1.008x |
| fp32/n10_k5 | 58.801 | 62.113 | 1.056x |
| fp32/n20_k2 | 11374.340 | 11314.765 | 0.995x |
| fp32/n20_k3 | 12044.646 | 11784.749 | 0.978x |
| fp32/n20_k5 | 60116.736 | 59296.016 | 0.986x |
| soa/n10_k2 | 13.761 | 15.082 | 1.096x |
| soa/n10_k3 | 11.384 | 12.436 | 1.092x |
| soa/n10_k5 | 28.495 | 32.592 | 1.144x |
| soa/n20_k2 | 20952.643 | 27783.033 | 1.326x |
| soa/n20_k3 | 14277.480 | 17805.042 | 1.247x |
| soa/n20_k5 | 21716.551 | 22791.267 | 1.049x |

The fused-cache allocation evaluator alternates widths 5/2 through eight noisy
shots, one context in a four-thread pool. Requested additional live heap is
160,152 bytes, below the 192,968-byte conservative plan including replacement
snapshots. The ordinary n=20 Pauli/Kraus budget cases keep their previous owned
heap scale; the new cache allocates nothing for those non-fused circuits.

Local suites pass 440 SV unit/property/integration checks; the real AVX512
release suites pass 495 (including architecture-specific tests). Both include
nine manual-IR/dense-oracle/generated-fusion checks.
Four optimized-driver oracle checks also passed. Workspace/all-target clippy,
fmt/diff, and identical-binary guard negative verification pass. The existing
ignored GHZ-25 resource smoke test remains ignored; no fused oracle was skipped.
Manual correctness review has no unresolved blocker. Performance regressions
above remain a merge consideration; this PR is kept draft for that reason.
