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
logical qubits. Generic external controls and arbitrary target order on manual
Kq matrices are executed by a query-local bounded matvec when controls or target
order need it. Validated canonical fused blocks retain the existing SIMD kernel. This avoids the existing
full-state CPU Kq limitations tracked by #573 without modifying that path.

Each nonzero/nonidentity term gets its own cone. Identical active-support and
retained-operation plans share one simulation, even for different Pauli axes or
coefficients. Disjoint cones stay separate instead of forming a global union.
They execute sequentially, so dense peak memory is the largest cone's workspace,
not the sum over terms. Identity terms are analytic; zero coefficients require
no state. Finite-result checks reject overflowing sums.

## Memory and complexity

For each active width k, the preflight bound is `40 * 2^k + 4096` bytes: state
(16N), backend norm-validation probabilities (8N), the X/Y expectation clone
(16N), and 4 KiB fixed gate scratch. Z-only and gate-free cases are conservatively
overestimated. All cones must fit the caller's explicit cap before any state is
allocated; dimension/size overflow produces a resource error. Identity-only
queries accept a zero-byte dense budget.

The cap applies per execution call; concurrent callers budget their aggregate
usage separately. This is a dense-buffer/scratch cap, not process RSS: the input circuit, owned
compact plans/matrices/maps, result vectors, allocator metadata, worker/runtime
stacks and library overhead are excluded. A large declared circuit still has
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

Clean timing evidence remains pending the requested CPU server campaign. The
busy Mac's correctness/demo execution is not speedup evidence. Keep the PR in
draft until the Criterion numbers, stage/memory changes and validation provenance
are recorded here.


Local validation: 13 focused tests pass (128 random property cases and five Aer
fixtures), along with 751 IR/SV regression tests and bench/example smoke cases.
The existing ignored `ghz_25_runs` remains explicitly excluded. Workspace Clippy,
changed-crate all-target Clippy and rustfmt pass. The default demo returns
X64 = -0.7190633875120727; exact source fingerprints and its structural output
are recorded in `data/p7-03/`. Its busy-Mac stage timings are correctness evidence
only. The paired API audit completes 20 rows, showing widths 18 -> 6 for the local
case and 16 -> 16 for the global case; no speed claim is made from that host.
