# P7-03: exact expectations on compact causal cones

`aleph_ir::expectation_query::ExpectationQueryPlan` plans a Hermitian Pauli sum
on a unitary circuit starting in |0…0>. `aleph_sv::expectation_query::ExactExpectationQuery`
adds a dense workspace cap and exact FP64 execution. The API returns expectation
values and diagnostics; compact states are not exposed as full output states.
Existing full-state, sampling and optimizer APIs are unchanged.

```rust
use aleph_core::{Pauli, PauliString, PauliSum};
use aleph_ir::Circuit;
use aleph_sv::expectation_query::ExactExpectationQuery;

let mut circuit = Circuit::new(128, 0);
circuit.ry(0.73, 124)?;
circuit.cnot(124, 125)?;
let observable = PauliSum {
    terms: vec![PauliString::new(1.0, vec![(125, Pauli::Z)])?],
};
let query = ExactExpectationQuery::compile(&circuit, &observable, 1 << 20)?;
let result = query.execute()?;
// result.value = cos(0.73); one compact 2-qubit state.
// query.plan().cones() exposes circuits, both logical maps and input term IDs.
```

Run the standalone Rust tool:

```sh
# Default: P3-10-style 128-qubit depth-three brickwork, local X64.
cargo run --release -p aleph-sv --example expectation_query
# QASM plus a Pauli sum text file (one '<real coefficient> <IXYZ string>' per line).
# Character q of the string acts on logical qubit q, matching PauliSum::parse.
cargo run --release -p aleph-sv --example expectation_query -- circuit.qasm observable.txt 67108864
```

The JSON reports original/active qubits, retained operations/source indices,
bidirectional original-logical/compact maps, per-term values, planning/execution
times and the peak dense workspace estimate. Parsing/filesystem/startup are
outside these API stage timers. The 128-qubit demo retains six qubits and 31
operations, needs a conservative 6,656-byte dense workspace bound and matches an
independently constructed 13-qubit spatial-window reference. Analytic pair
fixtures additionally check nonzero Z/XX/YY values on a 128-qubit circuit.

## Planning and supported semantics

The backward-support proof promotes the gate-only P3-10 reference in
`crates/aleph-mps/tests/shallow_100q.rs`: a unitary outside the live observable
support cancels in U† O U. A retained gate grows the cone by every target and
external control. Barriers retain relevant ordering without growing physical
support. Tiles expand into their constituent gates in original order; diagonal
polynomials conservatively retain their complete touched support. This can
produce a wider cone than a representation-specific decomposition, but never
silently discards an unknown operation.

All input operations are validated before pruning, including disjoint gates
and analytic identity/zero terms. Measure/reset are rejected with a boundary-state
error. Noise, classical feedback and custom/entangled initial states are not
inputs to this API; they need a different boundary treatment. Non-finite gate
parameters, malformed fused matrices/masks, nonunitary matrices and invalid Pauli
fields are rejected. Manual `UnitaryKq` is bounded to its documented width 2–5.

Relabel metadata maps logical observables into physical IR coordinates before
support propagation. Compact bit order follows sorted physical positions, so gate
matrix operand order is preserved; exposed maps always refer back to original
logical qubits. Uncontrolled Kq matrices use the existing SIMD-capable kernel directly from the
private immutable validated query plan, preserving arbitrary operand order and
avoiding a redundant public-backend validation snapshot. Controlled Kq matrices
use the query-local bounded matvec. The resource preflight adds the largest
uncontrolled Kq SIMD broadcast buffer bound (`128 * 4^k` bytes); it conservatively
applies on CPUs without AVX512 too. Public full-state CPU Kq controls remain
explicitly unsupported after #573.

Each nonzero/nonidentity term gets its own cone. Identical active-support and
retained-operation plans share one simulation, even for different Pauli axes or
coefficients. Disjoint cones stay separate instead of forming a global union.
They execute sequentially, so dense peak memory is the largest cone's workspace,
not the sum over terms. Identity terms are analytic; zero coefficients require
no state. Finite-result checks reject overflowing sums.

## Memory and complexity

For each active width k, the preflight bound is `40 * 2^k + 4096` bytes: state
(16N), backend norm-validation probabilities (8N), the X/Y expectation clone
(16N), and 4 KiB fixed gate scratch, plus `128 * 4^w` bytes where w is the largest
uncontrolled Kq matrix width (2–5) (zero when absent). Z-only and gate-free cases are conservatively
overestimated. All cones must fit the caller's explicit cap before any state is
allocated; dimension/size overflow produces a resource error. Identity-only
queries accept a zero-byte dense budget.

The cap applies per execution call; concurrent callers budget their aggregate
usage separately. This is a dense-buffer/scratch cap, not process RSS: the input circuit, owned
compact plans/matrices/maps, result vectors, allocator metadata, worker/runtime
stacks (including per-worker SIMD stack scratch) and library overhead are excluded. A large declared circuit still has
polynomial planning bookkeeping. No full `2^original_qubits` state is allocated.

Planning costs the unitary validation pass plus roughly T backward traversals of
G operations and Q-bit bookkeeping. Matrix validation is bounded by width five
(O(32³) per maximum fused matrix). Execution is exponential in each cone width,
with retained gates and Pauli terms determining work. Deep/global queries may
retain everything and add planning overhead; the planner is not a universal
speedup. Reusing the immutable query avoids replanning but does not retain an
exponentially sized cached state.

## Validation and measurements

The focused suite covers 128 random small circuits against full FP64 SV before
and after relabel/fusion/tiling, independent contractions of committed Qiskit Aer
amplitudes, nontrivial control/target order, Y signs, maps, identity, disjoint and
shared cones, barriers, diagonal bit 63, malformed/nonunitary inputs, finite sums,
and resource rejection before allocation. The 128-qubit fixtures include both
an analytic pair oracle and the independent P3-10 spatial window.

```sh
cargo test -p aleph-sv --test expectation_query
cargo bench -p aleph-sv --bench expectation_query -- --noplot
# Five alternating pairs, comparing the same public Rust API request:
RAYON_NUM_THREADS=1 cargo run --release -p aleph-sv --example expectation_query -- --audit
```

Criterion compares the same exact FP64 expectation request on local18 and
global16 circuits: full SV execution plus readout versus compile+compact-query
execution plus readout/release. Fixture construction and QASM parsing are outside
both timed paths. Separate planning-only and reused-plan tiers are labelled and
must not be substituted for end-to-end query results. Global16 has no cone-width
reduction and serves as an overhead/no-gain control.

The authorized CPU campaign ran on openwebgui (i5-13500, Rust 1.96.0,
Linux x86_64), pinned to performance-core logical CPU 2 with
`RAYON_NUM_THREADS=1`. The host was idle before building (load 0.00, no active
runner/benchmark); services stayed running. During timing, the benchmark used one
CPU and the process samples showed no competing CPU job. The load average still
included the preceding compilation; it is not a direct CPU-utilization measure.

| Exact FP64 request | Full SV | Compile + query | Planning only | Reused plan |
| --- | ---: | ---: | ---: | ---: |
| local18, depth 3, X9 (18 -> 6 qubits) | 48.461 ms | 24.162 us | 19.293 us | 4.103 us |
| global16, GHZ X15 (16 -> 16 qubits) | 779.44 us | 784.36 us | 3.786 us | 782.71 us |

These are Criterion slope estimates (10 samples, 0.5 s warmup, 2 s target;
local18/full automatically collected for 2.665 s). The local end-to-end benefit
is about **2,006x** for this deliberately small-cone fixture. Its dense workspace
bound falls from 10,489,856 to 6,656 bytes (about 1,576x); this is a buffer bound,
not measured RSS. Global16 keeps all 16 gates and the same 2,625,536-byte bound:
**no gain**, about 0.6% extra caller time. This tiny overhead should not be
generalized to fused matrices or many-term queries, whose validation/planning
cost can be larger. The global fixture is a support/no-reduction control, not
a representative hard global quantum workload.

Five alternating AB/BA API pairs independently check whole-call behavior and
correctness. Median caller times are local18 48.815 ms -> 27.786 us (median
paired speedup 1,767x), global16 795.439 us -> 800.839 us (0.6% slower).
The local query spends median 21.697 us planning and 5.391 us executing;
global16 spends 4.921 us planning and 795.340 us executing. The small difference
between Criterion and paired query timings is retained, not averaged away.
Reused-plan figures are a separate tier, not the end-to-end headline.

All 13 focused correctness/oracle tests passed again on this Linux host; the
128-qubit demo returns the same value and six-qubit plan. Source commit
`addb154286ecb84f3011d4f5b0570a5a5658ddf0` was transferred with `git archive`
to `/tmp/aleph-545-JgzjYC`; no uncommitted patch was needed. Raw Criterion samples,
estimates/log, alternating audit rows, compiler/hardware metadata, resource
samples, snapshot SHA256 and the demo are in `data/p7-03/server/`.

Reproduce after building (default features, no extra RUSTFLAGS):

```sh
cargo test -p aleph-sv --test expectation_query
cargo bench -p aleph-sv --bench expectation_query --no-run
# Use the executable printed above; pin both arms identically.
RAYON_NUM_THREADS=1 taskset -c 2 target/release/deps/expectation_query-<hash> --bench --noplot
cargo build --release -p aleph-sv --example expectation_query
RAYON_NUM_THREADS=1 taskset -c 2 target/release/examples/expectation_query --audit
```


Local validation: 13 focused tests pass (128 random property cases and five Aer
fixtures), along with 751 IR/SV regression tests and bench/example smoke cases.
The existing ignored `ghz_25_runs` remains explicitly excluded. Workspace Clippy,
changed-crate all-target Clippy and rustfmt pass. The default demo returns
X64 = -0.7190633875120727; exact source fingerprints and its structural output
are recorded in `data/p7-03/`. Its busy-Mac stage timings are correctness evidence
only. The paired API audit completes 20 rows, showing widths 18 -> 6 for the local
case and 16 -> 16 for the global case; no speed claim is made from that host.
