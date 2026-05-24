# 0002 — Always-false `is_clifford` for parametric gates in Phase 0

**Status:** Accepted
**Date:** 2026-05-24
**Decision driver:** P0-06 implementation.

## Context

`Gate::is_clifford()` reports whether a gate belongs to the Clifford
group. Standard gates (H, X, Y, Z, S, Sdg, Cnot, Cz, Swap, Iswap) are
unambiguously Clifford. Parametric rotations (`Rx(θ)`, `Ry(θ)`,
`Rz(θ)`, `Phase(θ)`, controlled variants, `U3`) are Clifford **only
for specific angles** (`θ = k · π/2` for the single-axis rotations;
`U3` has a narrower set still).

A correct angle-aware implementation would compare `θ` against `π/2`
multiples using a floating-point tolerance. The tolerance is its own
design problem:

- A loose tolerance produces false positives — non-Clifford gates
  silently slip into a stabilizer-only code path and corrupt results
  without raising any signal.
- A tight tolerance produces false negatives — Clifford-equivalent
  rotations get treated as generic non-Clifford and fall back to a
  slower path, but results stay correct.

## Decision

In Phase 0, `is_clifford()` returns `false` for every parametric
variant (`Rx`, `Ry`, `Rz`, `Phase`, `CRx`, `CRy`, `CRz`, `U3`) and
for both `Unitary1q` / `Unitary2q`.

## Consequence

- **Safe by default.** No silent corruption is possible from a
  parametric gate being mis-classified as Clifford.
- Phase 0 has no stabilizer backend, so the missed optimization
  costs nothing today.
- Phase 2 (stabilizer backend) will revisit this with a dedicated
  detection routine — likely a separate `clifford_decompose()` API
  that returns an optional sequence of Clifford generators, rather
  than a boolean predicate. That keeps the safe-default contract on
  `is_clifford()` intact.

## Alternatives considered

- **Tolerance-based detection now.** Rejected: silent-corruption
  failure mode is worse than the lost optimization, and Phase 0 has
  no consumer that would benefit.
- **Drop the method entirely.** Rejected: standard-gate Clifford
  classification is already useful for the IR (`Cnot.is_clifford()`
  lets a pass identify a Clifford prefix without re-deriving it
  from the matrix).
