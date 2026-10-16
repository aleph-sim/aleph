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

Cache hits compare coefficient bits exactly; signed-zero changes conservatively
miss and validate again. At most 16 KiB of matrix data is retained per backend at k=5. Replacement can
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

## Initial results and limits (before the follow-up)

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
above caused the original draft hold. The follow-up below records the subsequent changes and remeasurement.


## Follow-up: integrate main and reduce fused-kernel overhead

Main `64b729f6d7f000d0640439bc4c80ae23cf9a1405` is merged into the branch;
the Cargo conflict retains both `fused_validation` and `expectation_query` benches.
The public validation contract, bounded owned snapshot, invalid-input rejection,
operand ordering and unsupported-control behavior remain unchanged.

A new profile of the integrated previous implementation on the same Xeon places
most samples in the SoA AVX512 block closure, with the general fixed-bit index
walk its largest named inlined component. This motivates two bounded changes:

- Per-block input vectors become initialized fixed-size stack scratch: 512 bytes
  for scalar FP64, 4 KiB for AVX512 FP64 and 2 KiB for AVX512 FP32. Only the active
  prefix is used; k remains bounded to 2–5 (FP32 SIMD to 2–4). Every input is read
  before writes, preserving the existing gather/scatter and parallel disjointness
  proof. No new unsafe code is needed.
- Kq's target bits are always zero in the block base. Insert one zero bit at each
  sorted target position using a mask/shift instead of walking each free bit. This
  is O(k), independent of state width, and does not change matrix operand order.
  Exhaustive tests compare against the general walk for all supported target
  subsets/counters in nine bits and separately test high target positions.

The public cache still compares every coefficient and width on every hit. There
is no unchecked optimizer flag, pointer/hash trust or route for manually forged
IR to bypass validation. First-use matrix validation is intentional new work.

The main integration also brings #545's immutable expectation query. Uncontrolled
Kq gates now enter the SIMD-capable kernel directly from that private validated
plan, avoiding a redundant public-backend snapshot without trusting mutable public
IR. Controlled Kq stays on its bounded query-local path. Preflight includes the
largest uncontrolled Kq SIMD broadcast buffer (`128 * 4^k` bytes) in addition to
`40 * 2^active + 4096`; it conservatively applies without AVX512 too. Runtime and
worker stacks remain outside the dense heap-buffer cap. A new maximum-width
analytic H-tensor-five query checks execution/reuse at the exact expanded budget
and rejection one byte below it. The #545 local18/global16 measurements contain no
Kq blocks, so their bounds and reported timings retain their meaning.

`trajectory_budget --fused-kernels` checks warmed public apply allocation growth
between n=9 and n=18 for all three layouts in a one-thread pool. Bounded setup
allocations are allowed, but fewer than 64 calls are required; the old per-block
scratch allocation behavior fails that bound. The default noisy-budget evaluator
is also retained and rerun. Matrix snapshots and broadcast buffers remain bounded
per apply; these are not claims of zero allocation or process RSS.


After index/scratch changes, the remaining small-state k=5 profile includes
coefficient equality in `UnitaryKqCache::validate`. Its hit check now OR-reduces
exact XORs of every real/imaginary coefficient's `to_bits()` value. This allows
integer SIMD instead of a floating-equality branch per coefficient, without
unsafe casts, a hash or pointer trust. Width/length/controls are still checked;
any bit change revalidates. In particular +0/-0 changes conservatively miss the
cache. No NaN/Inf can be admitted by a hit because only successfully validated
finite matrices enter the owned snapshot.

The FP32 scalar fallback dispatches its supported widths to fixed-size input
arrays, exposing each matrix dimension/index bound to the compiler. This keeps
k=5 heap-free while avoiding the runtime-width scratch experiment's measured
~8% n=20 slowdown. All intermediate accepted campaigns remain alongside final
receipts; the complete final matrix, not the favorable probe, determines the
reported result.


A complete bit-comparison campaign still exposed FP32/k=5 sensitivity, so it is
not used as the final result. The final FP32 specialization additionally converts
the already shape-checked matrix slice to a reference to its exact-size array
once per apply. Matrix shape remains checked; the typed bound lets LLVM remove
coefficient-index checks from every block matvec. No arithmetic order, precision,
unsafe access or public input-validation bypass is introduced. Targeted probes
and the subsequent full matrix are retained separately.


### Final follow-up measurements (2026-10-16)

Same Xeon Silver 4114, CPU 3 / SMT sibling 23, monitor CPU39, Rust 1.96.0;
`RAYON_NUM_THREADS=1`, default bench profile/features, no extra RUSTFLAGS.
Baseline is current main `64b729f`, with the identical harness copied in.
Each source used its own fresh target; the baseline lacks the validation-cache
symbol, the candidate contains it, and ELF/source fingerprints are retained.
All seven measured source fingerprints match the final checkout, including the
matrix-bound specialization and bitwise cache check.

The **complete final 21-case campaign** has 84 accepted per-case A/B/B/A runs.
The shared-host guard rejects >2% preflight busy time or >=3% sampled external
CPU overlap on the selected physical core; all attempts and accepted samples are
retained. No CI/services were stopped. This does not isolate shared caches,
frequency or memory bandwidth, so differences below 5% get no speedup claim.
The numbers below average each arm's two Criterion median estimates; they are
public `apply_gate` timings, not complete CLI/circuit throughput.

| Case | Main us | Final us | Main / final |
| --- | ---: | ---: | ---: |
| aos/n10_k2 | 15.741 | 5.430 | 2.899x |
| aos/n10_k3 | 12.309 | 8.986 | 1.370x |
| aos/n10_k5 | 28.388 | 27.678 | 1.026x |
| aos/n20_k2 | 27503.668 | 6023.609 | 4.566x |
| aos/n20_k3 | 17705.783 | 8225.127 | 2.153x |
| aos/n20_k5 | 22794.262 | 19200.474 | 1.187x |
| cold_aos/n10_k5 | 28.536 | 79.397 | 0.359x |
| cold_fp32/n10_k5 | 60.427 | 86.558 | 0.698x |
| cold_soa/n10_k5 | 28.112 | 78.618 | 0.358x |
| fp32/n10_k2 | 7.149 | 3.037 | 2.354x |
| fp32/n10_k3 | 9.347 | 5.674 | 1.647x |
| fp32/n10_k5 | 58.734 | 58.940 | 0.996x |
| fp32/n20_k2 | 11460.675 | 2928.564 | 3.913x |
| fp32/n20_k3 | 12007.613 | 5113.775 | 2.348x |
| fp32/n20_k5 | 60140.993 | 58964.704 | 1.020x |
| soa/n10_k2 | 13.755 | 5.519 | 2.492x |
| soa/n10_k3 | 11.223 | 8.318 | 1.349x |
| soa/n10_k5 | 28.511 | 29.106 | 0.980x |
| soa/n20_k2 | 20841.530 | 5670.882 | 3.675x |
| soa/n20_k3 | 14333.812 | 9123.227 | 1.571x |
| soa/n20_k5 | 21416.938 | 20704.064 | 1.034x |

The original large-state SoA regressions are removed: n=20/k=2 is
20.842 -> 5.671 ms (3.68x), k=3 is 14.334 -> 9.123 ms (1.57x).
AoS n=20/k=2,3 improves 4.57x / 2.15x; FP32 improves 3.91x / 2.35x.
Warmed k=5 cases stay near baseline (small SoA is 2.1% slower, small FP32 0.4%
slower), with a larger AoS n=20 improvement; no universal speedup is claimed.
First-use k=5 remains materially more expensive: AoS 28.536 -> 79.397 us,
SoA 28.112 -> 78.618 us, FP32 60.427 -> 86.558 us. That new validation/copy
cost, including repeated misses when switching matrices, remains a real tradeoff.

The final warmed allocation evaluator reports 5 heap calls for AoS/SoA and 6
for FP32 at **both n=9 and n=18**, width two, one thread. The same evaluator on
main fails its bounded-allocation assertion with 16,389 calls for AoS n=18.
SIMD broadcast/setup and FP32 conversion allocations remain per apply; no zero
allocation or RSS reduction is claimed. The final noisy fused-cache budget
probe measures 156,056 additional requested heap bytes, below its 192,968-byte
plan. Stack scratch is accounted separately from requested heap.

Final validation: **518 AVX512 release SV tests pass**, including nine manual
matrix/dense-oracle/generated-fusion tests, five FP32 scalar/SIMD comparisons,
and 14 expectation-query tests; only the pre-existing GHZ-25 resource smoke
remains ignored. The final focused 28-test subset also passes. Local SV regression
checks, four optimized-driver oracles, final manual/query tests, workspace/all-target
Clippy, fmt and diff checks pass. Manual review covers cache mutation/bit equality,
fixed-width bounds, scatter disjointness, query ownership and broadcast budgeting;
there is no remaining blocking correctness finding. PR CI remains the merge gate.

Raw evidence and reproduction patches: `data/p8-573-followup/README.md`.
Intermediate complete matrices (including the unfavorable FP32 experiments),
probes, profiler/assembly output, all guard attempts and final samples are retained.
The PR can return to review with the first-use cost explicitly visible.
