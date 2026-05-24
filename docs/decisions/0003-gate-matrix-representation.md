# 0003 — Stack-allocated `GateMatrix` enum for `Gate::matrix()`

**Status:** Accepted
**Date:** 2026-05-24
**Decision driver:** P0-06 implementation.

## Context

`Gate::matrix()` is on the hot path for any backend that doesn't
specialize on the gate kind — the naive state-vector kernel reads
the matrix once per gate application. The return type sits at the
boundary between `aleph-core` and every backend; changing it later
is expensive.

Representations considered:

1. `ndarray::Array2<Complex>` — universal, runtime shape, but
   allocates on the heap for every call.
2. Const-generic per-arity methods: `matrix_1q() -> [[Complex; 2]; 2]`,
   `matrix_2q() -> [[Complex; 4]; 4]`, `matrix_3q() -> [[Complex; 8]; 8]`.
   No heap, but no uniform return type — callers must dispatch on
   gate kind before calling.
3. An enum wrapping fixed-size nested arrays:
   `enum GateMatrix { M2x2, M4x4, M8x8 }`.

## Decision

Adopt option 3: `enum GateMatrix { M2x2([[Complex; 2]; 2]),
M4x4([[Complex; 4]; 4]), M8x8([[Complex; 8]; 8]) }`.

The enum carries `#[allow(clippy::large_enum_variant)]` with an
inline justification — the 1024 B `M8x8` variant is the whole
point. Boxing would defeat the design.

## Consequence

- **No heap allocation per call.** Sizes: 64 B (M2x2), 256 B (M4x4),
  1024 B (M8x8) — all stack-friendly.
- Backends pattern-match on the variant once and forward to the
  arity-appropriate kernel.
- An n-qubit `Dense(Box<Array2<Complex>>)` variant can be added
  non-breakingly when an MPS or large-unitary use case arrives in
  Phase 1+.

## Alternatives considered

- **`ndarray::Array2`.** Rejected for hot-path heap allocations and
  for pulling `ndarray` into `aleph-core` purely for shape
  uniformity. Backends are free to convert internally if they want
  it.
- **Const-generic methods only.** Rejected because callers (IR
  passes, generic backends) lose a uniform API and must duplicate
  arity dispatch they would do anyway.
