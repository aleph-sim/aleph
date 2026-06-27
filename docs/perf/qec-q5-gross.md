# Q5-01 — Bivariate-bicycle (gross) code construction + DEM

**Issue:** Q5-01 (Phase Q5, qLDPC frontier).
**Depends on:** Q0-03.
**Status:** done.

## What and why

Phase Q5 moves past the surface code to **qLDPC** codes, where decoders are *not* a solved problem
(ROADMAP §2.3). The entry point is the IBM **bivariate-bicycle (BB)** family and its flagship
**gross code `[[144,12,12]]`** (Bravyi et al., [arXiv:2308.07915](https://arxiv.org/abs/2308.07915)):
12 logical qubits in 144 physical qubits with distance 12 and weight-6 checks — a 12× better
encoding rate than the surface code at comparable distance.

A BB code is a CSS code over the group algebra of `Z_ℓ × Z_m`. With `x, y` the cyclic shifts
(`xᵉ = yᵐ = I`, `xy = yx`) acting on `ℓm` cells, pick two three-term polynomials `A`, `B`; on
`n = 2ℓm` qubits split into a left/right block,

```text
H_X = [ A | B ]        H_Z = [ Bᵀ | Aᵀ ].
```

The CSS condition `H_X H_Zᵀ = AB + BA = 0 (mod 2)` is automatic because `A` and `B` commute. The
**gross code** is `ℓ = 12, m = 6, A = x³ + y + y², B = y³ + x + x²`.

The key structural fact for the decoder track: each check has weight 6, and **each qubit's error
lights 3 checks** — the syndrome graph is a *hypergraph*, not a matching graph. MWPM/Union-Find do
not apply; this is precisely why qLDPC needs belief propagation (Q3-02) and **BP+OSD** (Q5-02).

## What shipped

`crates/aleph-qec/src/bivariate_bicycle.rs` — `BBCode`:

- `BBCode::new(ℓ, m, A_monomials, B_monomials)` builds `H_X`, `H_Z` from the shift polynomials,
  **asserts the CSS condition**, and extracts a logical basis via GF(2) linear algebra (a small
  bit-vector kernel/echelon/inverse toolkit, no new dependencies).
- `BBCode::gross()` — the `[[144,12,12]]` code.
- Logical operators: `lz` = basis of `ker(H_X) / rowspace(H_Z)`, `lx` its **symplectic dual**
  (`lx[i]·lz[j] = δ_ij`) so the DEM's observables correctly report logical flips.
- `BBCode::code_capacity_dem(p)` — a [`DetectorErrorModel`] for independent `Z` noise: one mechanism
  per qubit (a 3-detector hyperedge over the `X`-checks), observables = the dual logical-`X`
  operators it anticommutes with. The Tanner graph for BP comes straight from this DEM via
  `BpDecoder::new(&dem).tanner()`.

## Acceptance criteria

- [x] **`[[144,12,12]]` gross code constructed; parameters verified.** `n = 144` and `k = 12` are
  computed from the GF(2) ranks (`k = n − rank H_X − rank H_Z`), not hard-coded; checks are weight 6,
  qubit degree 3; the CSS condition and the dual-logical relations are asserted in unit tests.
  `d = 12` is cited from Bravyi et al. (exact minimum distance of a `[[144,12]]` code is intractable
  to recompute here).
- [x] **Tanner graph + DEM emitted for decoding.** `code_capacity_dem` → 72 detectors, 12
  observables, 144 mechanisms; the BP Tanner graph has 432 edges (144 × 3). A single-qubit error
  produces its 3-check hyperedge syndrome and BP decodes it without panic/NaN; the DEM runs
  end-to-end through the Monte-Carlo harness.

## Results

`cargo run --release -p aleph-qec --example qec_q5_gross`. Data: `docs/perf/data/qec-q5-gross.{csv,log}`.

| property | value |
|----------|-------|
| n | 144 |
| k | 12 |
| d | 12 (cited) |
| X-checks / Z-checks | 72 / 72 |
| check weight | 6 |
| qubit degree | 3 |
| DEM detectors / observables / mechanisms | 72 / 12 / 144 |
| Tanner edges | 432 |

Sanity end-to-end (plain min-sum BP, code-capacity Z noise, 20k shots — **not** a threshold):

| p | logical rate |
|------|--------------|
| 0.002 | 0.0 |
| 0.005 | 5e-5 |
| 0.01  | 2e-4 |

Plain BP already suppresses low-`p` code-capacity errors, but standalone BP is **degeneracy-limited**
on qLDPC (the Q3-02 caveat): it stalls on symmetric error configurations. The proper decoder — BP
followed by ordered-statistics post-processing (**BP+OSD**) — is Q5-02, and it consumes exactly this
DEM + Tanner graph. The threshold curve and the comparison to published BB-code numbers are Q5-02 /
Q5-03.

## Files

- `crates/aleph-qec/src/bivariate_bicycle.rs` — `BBCode` + GF(2) toolkit + unit tests.
- `crates/aleph-qec/examples/qec_q5_gross.rs` — construction / Tanner / DEM / BP-sanity.
- `docs/perf/data/qec-q5-gross.{csv,log}` — committed run.
