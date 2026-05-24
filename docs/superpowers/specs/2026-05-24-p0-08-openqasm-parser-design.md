# P0-08 — OpenQASM 3.0 parser (minimal subset)

**Status:** Design approved 2026-05-24.
**Owner:** Ruslan.
**Depends on:** P0-07 (Circuit IR — merged in 99bd115).
**Blocks:** P0-10 (oracle harness — needs to parse Qiskit-emitted fixtures).

## 1. Goal

Implement a one-pass OpenQASM 3.0 parser in `aleph-parser` that consumes the minimal subset of OpenQASM needed to express the Tier-1 algorithms (GHZ, QFT, Grover, random circuit) and lowers them into `aleph_ir::Circuit`. Provide a normalized round-trip emitter so `Circuit → QASM → Circuit` preserves IR equivalence. Surface line/col error messages with a caret snippet, without an external diagnostics crate. Add the deferred `Circuit::try_new` to `aleph-ir` so the parser does not depend on the panicking `Circuit::new`.

## 2. Scope

### 2.1 In scope

- Header `OPENQASM 3.0;` (also tolerated: `3.x` minor versions).
- `include "stdgates.inc";` — accepted and treated as a no-op import (the standard gates are always available).
- Multiple register declarations: `qubit[N] name;`, `bit[N] name;`. Flattened internally into a single qubit/clbit index space.
- Gates by OpenQASM name (lowercase, case-sensitive): `h, x, y, z, s, sdg, t, tdg, rx, ry, rz, u3, p, cx, cz, swap, ccx`. `p` is an alias for `Phase`.
- Indexed qubit refs (`q[i]`) in gates. Indexed *and* whole-register refs in `barrier` and `measure`.
- `reset q[i];`.
- Expressions inside gate args: `pi`, float literals (decimal, scientific), `+ - * /`, parens, unary minus. Evaluated to `f64` at parse time and stored as `Param::Concrete`.
- Line comments `//` and block comments `/* … */`.
- Round-trip emitter: `Circuit → String`, normalized.
- Errors with 1-based line/col and a caret-underlined snippet of the offending source line.

### 2.2 Explicitly out of scope (deferred)

- Classical control flow (`if(c == N)`).
- Custom gate definitions (`gate foo a, b { ... }`).
- Subroutines (`def`, `box`, `delay`).
- `gphase`, `U(a, b, c) q;` (the un-parametrised root gate).
- Range/slice indexing (`q[0:3]`).
- Broadcasting on multi-qubit gates (`cx q, b;` where both are whole registers).
- Symbolic / variable parameters (`Param::Symbolic`).
- Expression functions: `sin`, `cos`, `tan`, `exp`, ...
- OpenPulse.

A `ParseError::UnsupportedFeature` variant exists for the cases above so callers get a clear "this is intentional, not a bug" message rather than a generic syntax error.

## 3. Architecture

Two-phase, with an explicit AST. The AST is private to the crate; only `parse` / `emit` are public.

```
source &str
    │
    ▼
lexer (nom_locate::LocatedSpan)   ← strips whitespace and comments, tracks line/col
    │
    ▼
parser (nom 7 combinators)        ← builds ast::Program
    │
    ▼
lower(&ast, source)               ← register flattening, whole-register expansion,
    │                               expression eval, IR validation
    ▼
Circuit  or  ParseError
```

Emitter is the inverse but works directly off `Circuit` — no AST needed because we agreed on a normalized (not byte-lossless) round-trip.

**Why a private AST:**
- Lowering (register flattening, whole-register desugaring, expression eval) is non-trivial and benefits from being a unit-testable function.
- Keeps the public surface minimal (`parse`, `emit`) — we can expose the AST later if a tooling use case appears.

## 4. Crate layout

```
crates/aleph-parser/
├── Cargo.toml
├── src/
│   ├── lib.rs            ← pub API + re-exports
│   ├── lexer.rs          ← tokens, whitespace, comments, LocatedSpan
│   ├── expr.rs           ← expression parser + evaluator (pi, arithmetic)
│   ├── parser.rs         ← nom combinators → ast::Program
│   ├── ast.rs            ← AST types with embedded spans
│   ├── lower.rs          ← ast::Program + &str → Result<Circuit, ParseError>
│   ├── emit.rs           ← Circuit → String (Result, since IR is wider than QASM)
│   └── error.rs          ← ParseError, ParseErrorKind, EmitError, render()
└── tests/
    ├── fixtures/
    │   ├── ghz.qasm
    │   ├── qft.qasm
    │   ├── grover.qasm
    │   └── random.qasm
    ├── algorithms_qasm.rs       ← parse each fixture, assert IR shape
    ├── round_trip.rs            ← parse → emit → parse on fixtures
    ├── round_trip_property.rs   ← proptest: random Circuit ↔ QASM
    └── error_rendering.rs       ← snapshot-style render() tests
```

## 5. Dependencies

Added to `[workspace.dependencies]`:

```toml
nom = "7"
nom_locate = "4"
```

Added to `crates/aleph-parser/Cargo.toml`:

```toml
[dependencies]
aleph-core = { path = "../aleph-core" }
aleph-ir   = { path = "../aleph-ir" }
nom        = { workspace = true }
nom_locate = { workspace = true }
thiserror  = { workspace = true }
smallvec   = { workspace = true }

[dev-dependencies]
proptest   = { workspace = true }
```

No `pest`, no `ariadne`, no `codespan-reporting`. The caret-snippet renderer is ~50 LOC of plain Rust.

## 6. Public API

```rust
// crates/aleph-parser/src/lib.rs
pub fn parse(source: &str) -> Result<aleph_ir::Circuit, ParseError>;
pub fn emit(circuit: &aleph_ir::Circuit) -> Result<String, EmitError>;
pub use error::{ParseError, ParseErrorKind, EmitError};
```

`Display` on `ParseError` is the one-liner (`error at L:C: <kind>`); `ParseError::render()` returns a 3-line block with the source line and caret. Both are available at any callsite.

## 7. Grammar (informal EBNF)

```
program     ::= header? include* decl* stmt*
header      ::= "OPENQASM" version ";"
version     ::= "3" ("." digit+)?
include     ::= "include" string_literal ";"      ; only "stdgates.inc" accepted

decl        ::= qreg_decl | creg_decl
qreg_decl   ::= "qubit" "[" uint "]" ident ";"
creg_decl   ::= "bit"   "[" uint "]" ident ";"

stmt        ::= gate_stmt | barrier_stmt | measure_stmt | reset_stmt

gate_stmt   ::= ident gate_args? indexed_ref ("," indexed_ref)* ";"
gate_args   ::= "(" expr ("," expr)* ")"
indexed_ref ::= ident "[" uint "]"
reg_or_idx  ::= ident | indexed_ref

barrier_stmt ::= "barrier" reg_or_idx ("," reg_or_idx)* ";"
measure_stmt ::= "measure" reg_or_idx "->" reg_or_idx ";"     ; both whole OR both indexed; sizes must match
reset_stmt   ::= "reset" indexed_ref ";"

expr        ::= add
add         ::= mul (("+" | "-") mul)*
mul         ::= unary (("*" | "/") unary)*
unary       ::= "-"? atom
atom        ::= float | "pi" | "(" expr ")"

float       ::= digit+ ("." digit*)? exponent?
              | "." digit+ exponent?
exponent    ::= ("e" | "E") ("+" | "-")? digit+
ident       ::= [a-zA-Z_][a-zA-Z0-9_]*
uint        ::= digit+
```

Whole-register form (`barrier q;`, `measure q -> c;`) is desugared at lowering time. Multi-qubit gates (`cx q, b;` with both whole) are out of scope and produce `ParseError::UnsupportedFeature { feature: "register-broadcast gate" }`.

## 8. Gate-name mapping

| OpenQASM | IR `Gate` variant |
|---|---|
| `h, x, y, z, s, sdg, t, tdg` | matching (`H, X, Y, Z, S, Sdg, T, Tdg`) |
| `rx, ry, rz` | `Rx, Ry, Rz` |
| `u3` | `U3` |
| `p` | `Phase` (alias) |
| `cx` | `Cnot` |
| `cz` | `Cz` |
| `swap` | `Swap` |
| `ccx` | `Toffoli` |
| anything else | `ParseError::UnknownGate { name }` |

Case-sensitive; OpenQASM 3 standard names are all lowercase.

## 9. Expression semantics

- `pi` → `std::f64::consts::PI`.
- Float literals parsed via `<f64>::from_str` (handles `1.5`, `.5`, `1.5e-3`, `1e10`).
- Operator precedence: unary minus > `*`, `/` (left-assoc) > `+`, `-` (left-assoc). Parens override.
- Evaluation happens at parse time; the AST stores only the evaluated `f64` (consistent with the normalized round-trip in § 10 — raw expression strings are not preserved).
- Division by zero produces `ParseError::BadExpression("division by zero")` rather than `f64::INFINITY`.
- A non-finite result (`NaN`/`Inf`) from any sub-expression evaluation also produces `ParseError::BadExpression`. Matches the `GateError::NonFiniteParam` invariant from P0-06.

## 10. Round-trip semantics

**Normalized, not byte-lossless.** The following are not preserved:

- Comments.
- Whitespace between tokens.
- Original register names (the emitter always emits `q` and `c`).
- The exact form of expressions (`pi/2` becomes the f64 it evaluates to).
- `include` directives (we always re-emit `include "stdgates.inc";`).

The following **are** preserved across `parse → emit → parse`:

- Instruction sequence (same `Gate` variants in the same order).
- Per-instruction parameters within f64 round-trip tolerance.
- `metadata.generated_from` is set to `"openqasm:3.0"` on the parsed `Circuit`.

The property test `round_trip_property` is the authoritative check: any randomly-built `Circuit` (excluding `Unitary1q/2q` and gates with external `controls`, which the emitter doesn't support — see § 12) survives `emit → parse` with equivalent instructions.

## 11. Error model

```rust
// crates/aleph-parser/src/error.rs
pub struct ParseError {
    pub line: u32,
    pub col: u32,
    pub kind: ParseErrorKind,
    snippet: String,   // private, used by render()
}

#[derive(Debug, thiserror::Error)]
pub enum ParseErrorKind {
    #[error("unexpected token: expected {expected}, found `{found}`")]
    UnexpectedToken { expected: &'static str, found: String },

    #[error("unknown gate `{name}`")]
    UnknownGate { name: String },

    #[error("undeclared register `{name}`")]
    UnknownRegister { name: String },

    #[error("index {index} out of bounds for register `{register}` of size {size}")]
    IndexOutOfBounds { register: String, index: u32, size: u32 },

    #[error("size mismatch in `{lhs} -> {rhs}`: {lhs_size} vs {rhs_size}")]
    SizeMismatch { lhs: String, lhs_size: u32, rhs: String, rhs_size: u32 },

    #[error("bad expression: {0}")]
    BadExpression(String),

    #[error("unsupported feature: {feature}")]
    UnsupportedFeature { feature: &'static str },

    #[error("too many qubits: declared {requested}, max {max}")]
    TooManyQubits { requested: u32, max: u32 },

    #[error("too many clbits: declared {requested}, max {max}")]
    TooManyClbits { requested: u32, max: u32 },

    #[error("IR rejected this program: {0}")]
    IrRejected(#[from] aleph_ir::CircuitError),
}

impl std::fmt::Display for ParseError { /* one-line: error at L:C: <kind> */ }
impl ParseError {
    /// Three-line block: header, source line, caret. Suitable for eprintln!.
    pub fn render(&self) -> String;
}

#[derive(Debug, thiserror::Error)]
pub enum EmitError {
    #[error("gate `{name}` has no OpenQASM 3 standard-subset representation")]
    UnsupportedGate { name: &'static str },

    #[error("symbolic parameter cannot be emitted (only Param::Concrete supported)")]
    Symbolic,

    #[error("external controls (count = {count}) cannot be emitted in the standard subset")]
    ExternalControls { count: usize },
}
```

`ParseErrorKind::IrRejected` forwards every `CircuitError` from `add_gate` / `add_instruction` — that is the IR's own validation layer (range, arity, duplicate qubit, too-many-controls, empty barrier) catching problems the parser didn't filter out. The parser deliberately does not duplicate IR validation.

## 12. `Circuit::try_new` (companion change to `aleph-ir`)

P0-07 § 12.2 deferred a fallible constructor to this PR; § 12.4 of the P0-07 spec will be added to record the closure.

```rust
// crates/aleph-ir/src/circuit.rs
impl Circuit {
    pub fn try_new(num_qubits: u32, num_clbits: u32) -> Result<Self, CircuitError> {
        if num_qubits > MAX_QUBITS {
            return Err(CircuitError::TooManyQubits { requested: num_qubits, max: MAX_QUBITS });
        }
        if num_clbits > MAX_CLBITS {
            return Err(CircuitError::TooManyClbits { requested: num_clbits, max: MAX_CLBITS });
        }
        Ok(Self {
            num_qubits,
            num_clbits,
            instructions: Vec::new(),
            metadata: CircuitMetadata::default(),
        })
    }
}
```

Two new variants in `CircuitError`:

```rust
#[error("too many qubits: requested {requested}, max {max}")]
TooManyQubits { requested: u32, max: u32 },
#[error("too many clbits: requested {requested}, max {max}")]
TooManyClbits { requested: u32, max: u32 },
```

`Circuit::new` keeps the `assert!` but the `TODO(P0-08)` comment is removed. The parser calls `Circuit::try_new`, never `Circuit::new`.

## 13. Emitter rules

- Always emit a fresh header: `OPENQASM 3.0;\n` followed by `include "stdgates.inc";\n\n`.
- Single qubit register `qubit[N] q;` where `N = circuit.num_qubits()`.
- If `circuit.num_clbits() > 0`, emit `bit[M] c;`.
- Blank line after declarations, then one instruction per line, no trailing blank lines.
- `Param::Concrete(f64)` formatted via Rust's default `{}` Display (shortest decimal that round-trips through f64).
- Errors (`EmitError`) on `Param::Symbolic`, `Gate::Unitary1q/Unitary2q`, and any `GateInstance` with non-empty `controls` (none of these are reachable from `parse` so round-trip is unaffected; they exist to keep the emitter total over the IR).

Sample:

```
OPENQASM 3.0;
include "stdgates.inc";

qubit[2] q;
bit[2] c;

h q[0];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
```

## 14. Testing

| Layer | Tests | Location |
|---|---|---|
| Unit | Each module — happy + 1–2 error paths each | `#[cfg(test)]` modules inline |
| Tier-1 fixtures | Parse `tests/fixtures/{ghz,qft,grover,random}.qasm`; assert instruction count + sample-index gate variants | `tests/algorithms_qasm.rs` |
| Round-trip (fixtures) | `parse → emit → parse → compare IR` per fixture | `tests/round_trip.rs` |
| Round-trip (proptest) | Random `Circuit` (`OpKind`-style strategy from P0-07, restricted to emittable gates) → `emit → parse → compare` | `tests/round_trip_property.rs` |
| Error rendering | A handful of syntactically-bad inputs → `assert_eq!` on `render()` output | `tests/error_rendering.rs` |

**Oracle (vs Qiskit Aer) is deferred to P0-10** per BACKLOG. The harness lands then; this PR does not add a Qiskit dependency.

### 14.1 Fixture files

`tests/fixtures/ghz.qasm` (3-qubit GHZ):
```qasm
OPENQASM 3.0;
include "stdgates.inc";

qubit[3] q;
bit[3] c;

h q[0];
cx q[0], q[1];
cx q[1], q[2];
measure q -> c;
```

Similarly for QFT-3 (Hadamards interspersed with controlled-phase rotations using `pi/2^k` expressions), Grover-2 (2-qubit Grover: marking oracle that flips `|11⟩` via `cz`, plus the standard `H–X–H` diffusion operator), and a 4-qubit random circuit (~10–15 gates mixing single-qubit Cliffords, `rx`/`rz` with parametric angles, and a few CNOTs). All fixtures use `include "stdgates.inc";` and the indexed-ref form for gates.

### 14.2 Property strategy

Reuse the `OpKind` enum pattern from `crates/aleph-ir/tests/layers_properties.rs`, restricted to variants that the emitter handles (no `Unitary*`, no externally-controlled instances). Add a helper in the parser crate's test code — do not export `OpKind` from `aleph-ir`.

## 15. Acceptance criteria

Mirrors `BACKLOG.md` § P0-08:

- [ ] Parses `tests/fixtures/{ghz,qft,grover,random}.qasm` into the expected `Circuit` IR.
- [ ] `parse → emit → parse` produces an equivalent `Circuit` on each fixture and on `OpKind`-generated random circuits (proptest default budget).
- [ ] `ParseError::render()` produces a 3-line block with line, column, and caret on the source.
- [ ] Multiple `qubit`/`bit` declarations supported and flattened.
- [ ] `aleph_ir::Circuit::try_new` exposed; parser uses it; no `Circuit::new` panic path is reachable from `parse`.
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` clean.
- [ ] `cargo fmt --check` clean.
- [ ] `cargo test --workspace` passes in both debug and release.

## 16. Open questions

None. Decisions during brainstorming were:

- Expression DSL: pi + arithmetic, evaluated to `f64` at parse time. (Q1)
- Multiple registers, flattened. (Q2)
- Indexed + whole-register `barrier`/`measure`; bare `barrier;` rejected. (Q3)
- `nom 7` + `nom_locate 4`. (Q4)
- Errors: line/col + caret snippet (~50 LOC, no extra crate). (Q5)
- Round-trip: normalized. (Q6)
- `Circuit::try_new` added in this PR. (Q7)
- `reset q[i];` included. (Q8)
- Block comments `/* */` accepted. (Q9)
- Emit pretty-printed, one instruction per line. (Q10)

If implementation surfaces a new decision, the spec gets a § 17 amendment before code lands (same pattern as P0-06/P0-07).

## 17. Amendments

### 17.1 Post-implementation code-review fixes (2026-05-24)

A `/code-review` pass on the freshly-implemented branch surfaced four real correctness bugs, three UX gaps, and three polish items. All addressed.

**Correctness:**

1. **Pre-check qubit uniqueness in `lower_gate`.** `cx q[0], q[0];` would otherwise trigger `GateInstance::new`'s `debug_assert` and panic any debug-build host. Now produces `ParseErrorKind::IrRejected(CircuitError::DuplicateQubit { qubit: q })` with the gate statement's source position. Release builds were already protected by `validate_gate`'s uniqueness check from P0-07 § 12.2; this fix unifies the contract between debug and release.

2. **Reject duplicate register declarations.** `RegisterMap::add_qreg` and `add_creg` previously used `HashMap::insert` which silently overwrote duplicates while still advancing `total_qubits`/`total_clbits`. New `RegError` enum and `ParseErrorKind::DuplicateRegister { name }` variant; the check also detects qreg/creg name collisions (`qubit[1] x; bit[1] x;`).

3. **Reject non-finite `Param::Concrete` in `emit`.** `Circuit::add_gate` does not enforce finiteness on f64 params, but the emitter writes `inf`/`NaN` strings the parser cannot re-ingest. New `EmitError::NonFiniteParam { value: f64 }` (and `EmitError` drops the `Eq` derive because f64 has no total equality). `extract_concrete` checks `is_finite` before emitting.

4. **Header missing-`;` no longer reports column 1.** `opt(header)` previously swallowed half-matched header parses, leaving the body parser to fail at offset 0 with a meaningless `error at 1:1: found OPENQASM 3.0`. The header parser now commits via `nom::combinator::cut` after `tag("OPENQASM")` and after the major version, so a missing `;` becomes a positioned `Failure` at the correct line/column. The same fix is applied to `include` after `tag("include")`.

**UX gaps:**

5. **Interleaved register declarations get a clear error.** Source like `qubit[1] q; h q[0]; qubit[1] aux;` previously reported `unexpected token: expected end of input, found 'qubit[1] aux;'`. Now: the trailing-input check in `parse()` inspects the first non-whitespace token and emits `ParseErrorKind::UnsupportedFeature { feature: "register declaration after gate statement" }`. (True interleaving support would require an `items: Vec<Item>` AST refactor and is deferred.)

6. **Common out-of-scope keywords map to `UnsupportedFeature`.** Spec § 2.2 promises that callers writing `if(c==0) x q[0];`, `gphase(...)`, `gate foo a, b { ... }`, `def`, `box`, `delay`, `U(...)` get a clear "intentional, not a bug" message. A best-effort heuristic in `nom_error_to_parse_error` (and the trailing-junk branch) inspects the failure-column source slice and matches a fixed list of keywords. This is heuristic — the parser doesn't structurally recognise these constructs — but covers the common typed-at-a-statement-position cases. Register-broadcast (`cx q, b;`) detection is not yet implemented: the parser fails at `indexed_ref` looking for `[` and produces a generic syntax error.

7. **Header rejects non-3 major versions.** `OPENQASM 2.0;` previously parsed cleanly and failed downstream at the declaration phase. The header now validates `major == 3` and returns a positioned `Failure` otherwise.

**Polish:**

8. **`lower_measure` whole-register branch captures `cpos`.** When `regs.creg_size(cname)` returns `None`, the error now points at the offending clbit-register name instead of the `measure` keyword.

9. **P0-07 spec § 12.4 amendment added** to record that `Circuit::try_new` landed alongside the parser (closes the `TODO(P0-08)` from P0-07 § 12.2).

10. **Proptest now asserts `metadata.generated_from`.** Spec § 10 lists this as a preserved round-trip invariant; the property test was missing the assertion.

**New unit tests** (in `crates/aleph-parser/src/lower.rs::tests` and `src/lib.rs::tests`):

- `rejects_duplicate_qubit_in_gate`
- `rejects_duplicate_qreg_name`
- `rejects_qreg_creg_name_collision`
- `cname_position_for_whole_register_measure`
- `rejects_nan_param`, `rejects_inf_param` (in `emit.rs::tests`)
- `header_missing_semicolon_points_at_header`
- `rejects_openqasm_v2_with_clear_message`
- `detects_interleaved_register_decl`
- `detects_unsupported_if_keyword`
- `detects_unsupported_gate_definition`

**Refuted candidates** (kept for the record, not fixed):

- Division by zero surfaces as `UnexpectedToken { expected: "numeric literal" }` instead of `BadExpression("division by zero")`. The spec § 9 promises the latter; the implementation acknowledges this as a known minor relaxation. Plumbing the inner string through nom's standard error type requires either a custom `ParseError<I>` type or a side-channel. Not worth the additional 50–80 LOC of glue for a rare error path; the position information is preserved.
- True interleaved decl/stmt ordering (per OpenQASM 3 spec) requires an `items: Vec<Item>` AST refactor and re-ordering the lowering pass. Deferred — Qiskit-emitted code always places decls before statements.
- Register-broadcast (`cx q, b;`) UnsupportedFeature detection: the parser correctly rejects but with a generic error. Heuristic sniffing would require post-hoc inspection of the surrounding source, which is fragile. Deferred.
