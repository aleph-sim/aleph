# Q5-02 — BP+OSD decoder

**Issue:** Q5-02 (Phase Q5, qLDPC frontier).
**Depends on:** Q5-01 (gross code + DEM), Q3-02 (BP).
**Status:** done.

## What and why

Belief propagation alone is **degeneracy-limited** on quantum LDPC codes: many distinct errors share
a syndrome, so BP either oscillates or — worse — *converges to a valid-but-wrong-coset* solution.
The standard fix is **BP + ordered-statistics decoding** (Panteleev–Kalachev; Fossorier–Lin OSD;
Roffe's `ldpc`, concepts only): run BP, and on the shots where it fails, turn BP's *soft* output into
a guaranteed syndrome-consistent low-weight error by a reliability-ordered GF(2) solve.

`OsdDecoder` (`crates/aleph-qec/src/osd.rs`) wraps the Q3-02 `BpDecoder` and adds the OSD tail:

1. Order variables by BP reliability `|posterior LLR|`, **descending**.
2. Gauss–Jordan-reduce `H` over GF(2), taking pivots greedily in that order — the **most-reliable
   basis**.
3. **OSD-0**: keep BP's hard decision `ê` on the non-pivot columns and solve the pivots so `H e = s`.
4. **OSD combination sweep (order `w`)**: try every flip pattern on the `w` least-reliable non-pivot
   columns, re-solving pivots, keep the least-soft-weight candidate.

Two design points proved decisive (both found by instrumenting BP convergence):

- **Normalised min-sum (`α = 0.875`), not plain `α = 1`.** Plain min-sum over-converges to
  wrong-coset solutions (BP reports `H ê = s` and OSD never runs); at `p = 0.04` it failed **29%** of
  shots. Normalised min-sum drops that to **2.3%** and makes BP *oscillate* on the genuinely
  degenerate shots, handing them to OSD. This is now the `OsdDecoder::new` default.
- **Keep `ê` on non-pivot columns, don't zero them.** OSD-0 with the BP hard decision on non-pivots
  *refines* BP; zeroing them lands in the wrong logical coset far more often.

## Acceptance criteria

- [x] **Decodes the gross code; logical-error rate vs physical-error rate curve produced.**
- [x] **Threshold within range of published BP+OSD results for the same code/noise.**

## Results

Box: M4 Mac Mini. Independent-`Z` code-capacity noise, normalised min-sum `α = 0.875`, OSD
combination-sweep order 10, 30 000 shots. `cargo run --release -p aleph-qec --example qec_q5_bposd`.
Data: `docs/perf/data/qec-q5-bposd.{csv,log}`.

### Gross `[[144,12,12]]` — BP vs BP+OSD

| p | plain BP | BP+OSD | improvement |
|------|----------|--------|-------------|
| 0.01 | 1.67e-4 | **1.33e-4** | 1.25× |
| 0.02 | 1.13e-3 | **9.00e-4** | 1.26× |
| 0.03 | 4.37e-3 | **4.27e-3** | 1.02× |
| 0.04 | 2.09e-2 | **2.01e-2** | 1.04× |
| 0.05 | 6.22e-2 | **6.12e-2** | 1.02× |
| 0.06 | 1.44e-1 | **1.42e-1** | 1.01× |

The curve is monotonic and BP+OSD ≥ BP everywhere, with the gain largest at low `p` (≈1.25×) where the
residual failures are exactly the degenerate cases OSD targets. With `α = 0.875`, BP already converges
on 98–100% of low-`p` shots, so the OSD tail acts on few shots here; its value grows at higher `p`,
larger codes, and (future) circuit-level noise. A higher combination-sweep order widens the gap
further (e.g. order 16 roughly halves the residual at `p ≤ 0.02`).

### Threshold — BB family `[[72,12,6]]` (d=6) and `[[144,12,12]]` (d=12)

A constant-`k=12` BB family with **known growing distance** (same polynomials, `m = 6`, `ℓ = 6, 12`;
`d` from Bravyi et al.):

| p | d=6 (n=72) | d=12 (n=144) |
|-------|-----------|--------------|
| 0.06  | 0.257 | 0.142 |
| 0.08  | 0.473 | 0.412 |
| 0.085 | 0.527 | 0.492 |
| 0.09  | 0.577 | 0.568 |
| 0.095 | 0.625 | 0.639 |
| 0.10  | 0.671 | 0.702 |

Below the crossing the larger (d=12) code suppresses errors more; above it, it amplifies them. The
curves cross at

> **p_th ≈ 0.092** (9.2%).

### Is that "within range of published"?

This is the **single-Pauli code-capacity** channel (independent `Z`, perfect measurements). For
reference, the surface/toric code under the *same* channel has a code-capacity threshold of **≈10.9%**.
A qLDPC code at rate `k/n = 1/12` reaching **≈9.2%** — close to, and just below, the surface code's
code-capacity threshold while encoding 12× as many logical qubits per physical qubit — is squarely in
the range published for qLDPC BP+OSD code-capacity decoding (high-single-digit to ~10%). The result
is a genuine threshold (the d=6/d=12 curves cross cleanly), reproduced by BP+OSD on aleph's own DEM.

## Honest boundary

- The threshold is a **two-distance crossing estimate**, not a full finite-size-scaling fit; the
  same-polynomial family has paper-confirmed distance only at `ℓ = 6, 12` (larger `ℓ` distances are
  not certified here).
- This is **code capacity**, not the **circuit-level** ~0.7% the gross code is famous for. A
  circuit-level DEM (syndrome-extraction schedule + measurement noise) is the natural follow-up and
  is where BP+OSD's advantage over BP is largest; Q5-01 shipped only the code-capacity DEM.
- OSD currently runs only when BP fails to converge; an always-on OSD-CS variant is a Q5-03 lever.

## Files

- `crates/aleph-qec/src/osd.rs` — `OsdDecoder` + GF(2) reliability-ordered solve + unit tests.
- `crates/aleph-qec/src/bp.rs` — `BpDecoder::decode_bp_soft` / `BpSoft` (posterior LLRs for OSD).
- `crates/aleph-qec/examples/qec_q5_bposd.rs` — BP-vs-BP+OSD curve + threshold crossing.
- `crates/aleph-qec/tests/bposd.rs` — decodes-at-low-p, distance-suppression, sweep-no-regress.
- `docs/perf/data/qec-q5-bposd.{csv,log}` — committed run.
