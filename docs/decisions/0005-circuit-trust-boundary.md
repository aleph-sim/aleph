# ADR 0005 — `Circuit::new` vs `Circuit::try_new` trust boundary

**Status**: Accepted (2026-05-25, retroactive)
**Issues**: [P0-07](../../BACKLOG.md) §12.2, [P0-08](../../BACKLOG.md)

## Context

`aleph_ir::Circuit` is constructed in two very different
contexts:

1. **Trusted in-process code** (test fixtures, the bench
   harness, the IR builder methods themselves) — qubit/clbit
   counts are compile-time-ish constants known to be within
   limits; an out-of-range value is a programming error, not a
   user error.
2. **Untrusted parsed input** (the OpenQASM parser P0-08, any
   future deserialiser) — a malformed `qubit[99999] q;`
   declaration must surface as a structured error, not a panic.

Pre-P0-08, `Circuit::new(num_qubits, num_clbits)` was the only
constructor — and it returned `Self`.  Hard limits
(`MAX_QUBITS = 65_535`, `MAX_CLBITS = 65_535`) were checked via
`debug_assert!`, which is compiled out in release.  P0-08's
parser then needed a fallible variant; P0-07's spec § 12.2
captured the requirement and § 12.4 closed the loop after the
parser landed in P0-08.

## Decision

Two constructors with different contracts:

* **`Circuit::new(num_qubits: u32, num_clbits: u32) -> Self`**
  — trusted-input fast path.  Panics with a clear message if
  `num_qubits > MAX_QUBITS` or `num_clbits > MAX_CLBITS`.
  Intended for fixtures, builders, and any code where the
  counts are known safe.

* **`Circuit::try_new(num_qubits: u32, num_clbits: u32) ->
  Result<Self, CircuitError>`** — untrusted-input safe path.
  Returns `CircuitError::TooManyQubits { requested, limit }` or
  `TooManyClbits { requested, limit }` instead of panicking.
  Required for parsers and any future deserialisers.

The parser uses `try_new`; the IR's own builder helpers
(`Circuit::new` in tests, `bell_circuit()` / `ghz_circuit()` in
benches) use the panicking variant.

## Consequences

* The trust boundary is **explicit at the call site**.  A code
  reviewer seeing `Circuit::new(...)` knows the inputs are
  trusted; `try_new(...)` signals "external input, must handle
  rejection".
* `MAX_QUBITS = MAX_CLBITS = 65_535` is the IR-wide hard cap.
  Backends layer their own caps on top — `aleph-sv` enforces
  `MAX_NAIVE_QUBITS = 28` (memory) via
  `BackendError::TooManyQubits`.  See ADR 0006 (forthcoming)
  for the cap derivation.
* Future fallible constructors (e.g. `Circuit::from_layers`)
  should follow the same `try_*` convention.

## Alternatives considered

* **One `try_new` only, no panicking variant.**  Rejected
  because it forces every test and every internal builder to
  `.unwrap()` an obviously-safe construction — line noise that
  obscures intent.

* **Type-state encoding** (`Circuit<Validated>` vs
  `Circuit<Raw>`).  Rejected as over-engineering for a value
  that's already cheap to construct and validate.

## References

* `crates/aleph-ir/src/circuit.rs::Circuit::new`,
  `Circuit::try_new`.
* `crates/aleph-parser/src/lib.rs` — parser uses `try_new`.
* `benches/src/lib.rs` — bench helpers use `Circuit::new`.
* `crates/aleph-core/src/error.rs` — `BackendError::TooManyQubits`
  for the per-backend layer above.
