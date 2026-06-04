# P3-02 — Stabilizer measurement with collapse — Design

**Issue:** #33 (P3-02)
**Milestone:** Phase 3 — Alternative Backends
**Depends on:** P3-01 (merged `2dbbb8e`)
**Status:** Design approved, pending spec review
**Date:** 2026-06-04

---

## 1. Goal & scope

Add projective Z-basis measurement with state collapse to the
`aleph-stab` `Tableau`, following Aaronson-Gottesman (2004) §3. This is
the operation the P3-01 scratch row was reserved for.

**In scope:**

- `g(x1,z1,x2,z2)` — the AG §2 phase-exponent helper.
- `rowsum(h, i)` — sign-tracking generator left-multiply.
- `measure<R: rand::Rng>(&mut self, qubit, rng) -> Result<bool, StabError>`
  — deterministic + random cases with collapse.
- `copy_row` / `zero_row` private helpers.
- Tests: unit, statistical (GHZ 50/50), Stim oracle (postselect), proptest.

**Out of scope:**

- `Backend` trait impl + CLI `--backend stabilizer` → **P3-03**.
- Multi-qubit `sample`, mid-circuit reset, X/Y-basis measurement.

---

## 2. Algorithm (Aaronson-Gottesman §3)

Tableau layout (from P3-01): rows `0..n` destabilizers, `n..2n`
stabilizers, row `2n` scratch; each row has `n` x-bits, `n` z-bits, sign
`r` (`true` = `-`).

### 2.1 Phase-exponent `g(x1, z1, x2, z2) -> i32`

The power of `i` produced when the Pauli `(x1,z1)` is left-multiplied
onto `(x2,z2)` on one qubit (AG §2):

| (x1, z1) | g |
|----------|---|
| (0,0)    | `0` |
| (1,0)    | `z2 * (2*x2 - 1)` |
| (0,1)    | `x2 * (1 - 2*z2)` |
| (1,1)    | `z2 - x2` |

(bits are 0/1 `i32`s.)

### 2.2 `rowsum(h, i)`

Sets generator row `h ← (row i) · (row h)`, tracking the sign:

```
acc = 2*r_h + 2*r_i + Σ_j g(x_{i,j}, z_{i,j}, x_{h,j}, z_{h,j})
acc mod 4 == 0  → r_h = false
acc mod 4 == 2  → r_h = true
(1 and 3 are unreachable for valid Pauli products; debug_assert)
for all j: x_{h,j} ^= x_{i,j};  z_{h,j} ^= z_{i,j}
```

Per-column loop over `0..n`. Measurement is not the P3-01 perf-AC hot
path, so correctness-first plain code — no branchless micro-optimization.
Use `i32`/`i64` accumulation; the sum is bounded (`≤ ~3n`) so no overflow.

### 2.3 `measure(a, rng)`

1. Bounds: `a < n` else `StabError::QubitOutOfRange`.
2. Find the first stabilizer row `p ∈ [n, 2n)` with `x(p, a) == 1`.
3. **Random outcome** (`p` found — `Z_a` anticommutes with a stabilizer):
   - For every row `i ∈ [0, 2n)` with `i != p` and `x(i, a) == 1`:
     `rowsum(i, p)`.
   - Copy row `p` into row `p - n` (the paired destabilizer).
   - Zero row `p`; set `z(p, a) = 1`; set `r_p = rng.gen::<bool>()`.
   - Outcome = `r_p`.
4. **Deterministic outcome** (no such `p`):
   - Zero the scratch row `2n`.
   - For every destabilizer `i ∈ [0, n)` with `x(i, a) == 1`:
     `rowsum(2n, i + n)`.
   - Outcome = `r_{2n}`.
5. Return `Ok(outcome)`.

`copy_row(dst, src)` copies all x/z bits + sign; `zero_row(r)` clears
all x/z bits + sign of row `r`. Both implemented over the existing
`BitGrid` accessors plus a column loop (or word loop — implementer's
choice; correctness-first).

---

## 3. Dependency

Add `rand` (workspace dep, already used by `aleph-sv`) to
`crates/aleph-stab/Cargo.toml`. Justified: the random-outcome branch
needs a uniform random bit. The generic `R: rand::Rng` signature mirrors
`aleph-sv`'s `measure_impl(&mut rng, …)`, so P3-03 wiring (the
`Backend::measure` method already takes `&mut self.rng: StdRng`) is a
direct call.

---

## 4. Testing

### 4.1 Unit — `g` and `rowsum`
- `g` exhaustive 4×4-input table vs the §2.1 table.
- `rowsum` hand-checked small cases: combining two stabilizer rows of a
  known state yields the expected Pauli string + sign; `rowsum(h, i)`
  followed by `rowsum(h, i)` restores row `h`'s bits (XOR involution),
  with the sign tracked.

### 4.2 Unit — measurement
- **Bell forcing (AC):** `H(0); CNOT(0,1)`; `measure(0, rng)` → `b`;
  `measure(1, rng)` returns the **same** `b`; and re-measuring q0 returns
  `b` again (post-collapse determinism).
- **Deterministic:** fresh `|0…0⟩`, `measure(a)` → `false` for all `a`,
  with no RNG advance dependence (outcome independent of seed).
- **Random single qubit:** `H(0)`, `measure(0)` is random across seeds.

### 4.3 Statistical (AC)
- GHZ-`n` (`H(0)` then `CNOT(i,i+1)`): over `K` seeded trials, the q0
  outcome is ~50/50 (binomial tolerance), and within every trial all `n`
  qubits measure equal (all-0 or all-1).

### 4.4 Stim oracle (`#[ignore]`, EPYC) — `tests/stim_measure_oracle.rs`
Reuse the P3-01 stim harness pattern + `/root/stim312` venv (stim 1.16).
Per random Clifford circuit and target qubit `a`:
- Build the tableau (our `apply_gate`), `measure(a, rng)` → outcome `b`.
- In Stim: run the same circuit on a `TableauSimulator`, then
  `postselect_z(a, desired_value = b)`.
- Compare post-measurement **canonical stabilizer groups**
  (`Tableau.from_stabilizers(...).to_stabilizers(canonicalize=True)`,
  sorted-set equality), exactly as P3-01.
- Determinism cross-check: query Stim `peek_z(a)`; if it reports a
  determined sign, assert our `measure(a)` returns that value for every
  seed (no randomness consumed).

### 4.5 Property (proptest)
After applying a random Clifford circuit and one `measure(a)`, the
tableau still satisfies the symplectic invariant
(`rows_anticommute(i, n+i)` etc., reusing the P3-01 helper) — i.e.
collapse leaves a well-formed tableau.

---

## 5. Error handling

Only `StabError::QubitOutOfRange { qubit, num_qubits }` (existing
variant). A measurement on a valid qubit cannot fail. No new variants.

---

## 6. Acceptance-criteria mapping

| AC | Covered by |
|----|-----------|
| Measurement implemented | §2.3, all of §4 |
| Deterministic + random cases correct | §4.2, §4.4 determinism cross-check |
| Bell pair: measuring q0 forces q1 | §4.2 Bell forcing |
| Equivalence vs Stim | §4.4 |
| GHZ measurements 50/50 | §4.3 |

---

## 7. Risks & notes

- **`rowsum` phase math** is the subtle part — the `g` table and the
  `mod 4 ∈ {0,2}` reduction. The Stim oracle (post-collapse group
  equality) plus the Bell/GHZ sign-correlation tests are the guards; a
  sign bug surfaces as a flipped stabilizer sign or a wrong correlation.
- **Stim `postselect_z` zero-probability guard:** we always postselect
  to *our own* measured outcome `b`, which by construction has nonzero
  probability in the same state, so postselect never rejects. (Driving
  Stim from our outcome — not an independent draw — is deliberate.)
- **RNG reproducibility:** tests use a seeded `StdRng`; the statistical
  test fixes a seed and asserts a binomial-tolerance band, not an exact
  count, to stay deterministic yet meaningful.
- The scratch row `2n` is written/read only inside the deterministic
  branch and left in a scratch state afterwards (callers never read it).
