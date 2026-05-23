# ADR 0001 — Complex number primitive

**Status**: Accepted (2026-05-24)
**Issue**: [P0-03](../../BACKLOG.md)

## Context

Every quantum state amplitude, gate matrix entry, and kernel intermediate is a complex number. The choice of complex-number type touches every crate in the workspace and is hard to change later because:

- Public API (`StateVector`, `Gate`, backend traits) returns complex values; swapping the type breaks every consumer.
- Performance-critical kernels (`aleph-sv` 1q/2q gate application) are written against this type — its memory layout and arithmetic codegen matter.
- FFI boundaries (CUDA `cuDoubleComplex` in P5, PyO3 in P4) must accept the same representation without copying.

The decision was scoped to FP64 (the project's correctness target, per CLAUDE.md tolerance 1e-10). FP32 and FP16 may matter later for GPU memory bandwidth but are out of scope for Phase 0.

## Decision

Use **`num_complex::Complex<f64>`** as the canonical element type, exposed in `aleph-core` as a generic alias:

```rust
pub type Complex<T = f64> = num_complex::Complex<T>;
```

Every workspace crate consumes `aleph_core::Complex` rather than reaching for `num_complex` directly. The `T = f64` default keeps current call sites unchanged while allowing future `Complex<f32>` / `Complex<f16>` specialisations for GPU paths without churning the type name.

State-vector storage layout (AoS vs SoA) is a **separate** concern handled by `aleph_core::StateVector` (lands in P0-09) and any future SoA variant in `aleph-sv`. The element-type choice does not foreclose either.

## Alternatives considered

### Custom `struct Complex { re: f64, im: f64 }`

Pros considered:
- Full control over layout, repr, and impls.
- Tailored API surface (e.g., expose only the operations we use).
- No transitive dependency on `num-traits`.

Why rejected:
- `num_complex::Complex<T>` is already `#[repr(C)]`, which is the only layout guarantee we actually need for FFI to `cuDoubleComplex` (`struct { double x, y; }`) and for PyO3 buffer protocol. Custom would not add anything FFI-relevant.
- Reimplementing `Add`/`Sub`/`Mul`/`Div`/`Neg`/`AddAssign`/.../`abs`/`arg`/`conj`/`Display`/`Debug`/`Hash`/`PartialEq` plus the `num-traits` integrations is meaningful code we'd have to test, document, and maintain — for zero observable gain over `num-complex`.
- `num-complex` is depended on by the wider Rust scientific ecosystem (`ndarray`, `nalgebra`, `faer`, etc.). If we ever pull one in, type interop is free.

### SoA `StateVector { re: Vec<f64>, im: Vec<f64> }` instead of a `Complex` element type

Pros considered:
- Best case for AVX-512 / NEON: separate real/imag arrays let the compiler vectorise tight loops without packing/unpacking.
- Matches what Qiskit Aer and QuEST do in their hot kernels.

Why rejected at this stage:
- P0-03's acceptance criteria explicitly require `aleph_core::Complex` to exist as an alias and "All current usage routed through this alias". A pure-SoA design has no element type to alias.
- SoA is a *storage* decision that belongs in `StateVector` (P0-09) and gate-application kernels (P1-01: "Struct-of-Arrays (SoA) memory layout for amplitudes"), not in the element type. We can introduce `StateVector` with an internal SoA representation while still exposing `Complex` for matrix entries, scalars, IR constants, and any future generic kernels.
- Going SoA-only in Phase 0 would commit us before we have benchmarks (P0-04) showing it's the right call against AoS.

## Consequences

Positive:
- Every consumer gets a battle-tested `Complex<T>` with the full `num-traits` ecosystem (`Float`, `Num`, `Zero`, `One`) — useful for generic gate kernels in P0-06 and tolerance-based proptest invariants in P0-05.
- FFI to CUDA / PyO3 should be straightforward: `#[repr(C)]` + `(f64, f64)` matches the field layout of `cuDoubleComplex` and Python's `complex` (verified for `cuDoubleComplex`; PyO3 interop needs `pyo3` built with its `num-complex` feature — to be confirmed when P4 lands).
- `Complex<f32>` available for free if/when GPU memory bandwidth becomes the bottleneck (lock-in test exists in `aleph-core::tests::f32_precision_works`).

Negative:
- One extra workspace dependency (`num-complex` + transitive `num-traits`). Compile-time cost is small (~1 s) and both crates are dependency-free leaves.
- AoS-style element doesn't help SIMD on its own — we depend on writing SoA `StateVector` in P0-09 / P1-01 to get vectorisation, which was always the plan.

Neutral:
- If we ever need to swap the underlying type (e.g., `f16` via a different crate), the `aleph_core::Complex` alias contains the blast radius: change the alias, recompile.

## Constraints this decision imposes on downstream code

1. **`Backend` trait (P0-09) must not expose `&[Complex]` / `&mut [Complex]` in its public API.** Every method either operates on an opaque `Self::State` or returns owned data. The reason: an SoA backend (P1-01) stores amplitudes as `re: Vec<f64>, im: Vec<f64>` and cannot cheaply hand out an AoS `&[Complex]` slice without a copy. Exposing AoS slices would force perpetual SoA↔AoS conversions in the hot path. The trait can still publish `Self::Amplitude = Complex` and let callers reach individual entries via methods like `fn amplitude(&self, idx: usize) -> Complex` — point access is cheap; bulk slice access is not.
2. **No workspace crate may name `num_complex::Complex` directly.** Enforced by a `disallowed-types` rule in `clippy.toml`; the single allowlisted site is the alias declaration in `aleph-core/src/lib.rs`. This keeps the alias's "change one line to swap the type" contract from drifting as new crates come online.
3. **The generic parameter `T` is conventionally restricted to `num_traits::Float` types.** Type aliases in Rust can't carry trait bounds, so `Complex<i32>` typechecks at the alias site; the compile error only surfaces deep in a downstream function. Public APIs that take `Complex<T>` should bound `T: Float` explicitly so the error happens at the boundary.
4. **`pyo3` and `cudarc` integrations must verify their assumed layout** when they land:
   - PyO3 needs to be built with its `num-complex` cargo feature for `IntoPy<Complex>` / `FromPyObject` to work — to be wired in P4.
   - cuDoubleComplex / cuComplex are `#[repr(C)] { double|float x, y }`. Field order matches `(re, im)` in `num_complex::Complex`, but cuBLAS / cuFFT may impose stricter alignment than the default `align_of::<Complex<f64>> = 8`. To be verified against the target NVIDIA library in P5; if cuBLAS wants 16-byte alignment, we'll wrap with `#[repr(align(16))]` at the FFI boundary, not at the canonical type.

## References

- `num-complex` crate: <https://docs.rs/num-complex/>
- ADR format: <https://adr.github.io/>
- `cuDoubleComplex` ABI (motivating `#[repr(C)]`): <https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__CAST.html>
