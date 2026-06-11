# P3-09: MPS Lazy SWAP Permutation + Multithreading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the swap-back half of the MPS non-adjacent-2q SWAP network via lazy permutation tracking, and parallelize the 2q hot path (theta gemm, SVD, QR, bond absorption) via faer's rayon backend.

**Architecture:** `MpsState` gains a site↔qubit permutation maintained by `swap_adjacent`; all reads route through it, so the reverse SWAP ladder disappears. The 2q hot path is recast onto zero-copy faer views (`Site.data` row-major layout *is* both the grouped-left and grouped-right matrix), with theta-build and bond absorption as parallel gemms and SVD/QR on faer with the global rayon pool.

**Tech Stack:** Rust, faer 0.24 (features `linalg`, `std`, + new `rayon`), nalgebra (kept only for transfer contractions in `overlap`/`probabilities`), criterion, proptest.

**Spec:** `docs/superpowers/specs/2026-06-11-p3-09-mps-lazy-swap-multithreading-design.md`
**Branch:** `p3-09-mps-lazy-swap-multithreading` (already exists, contains the spec commit). NO worktrees.

**Key API facts (verified against vendored faer 0.24.0 sources):**
- `aleph_core::Complex` = `num_complex::Complex<f64>` = `faer::c64` — the *same* type alias. faer matrices can be `faer::Mat<Complex>`; no conversion, no casts.
- `Site.data` row-major layout `(l*2+p)*right + r` viewed as `(left·2) × right` row-major = grouped-left matrix; viewed as `left × (2·right)` row-major = grouped-right matrix. Both are zero-copy `MatRef::from_row_major_slice` views.
- `view.qr()` → `Qr` with `.compute_thin_Q() -> Mat`, `.thin_R() -> MatRef` (impl is on generic `mat::generic::Mat<Inner>`, so it works on `MatRef` views).
- `view.thin_svd() -> Result<Svd, SvdError>` with `.U()`, `.S()`, `.V()` (V columns = right singular vectors; Vᴴ row t = conj of V col t).
- `faer::linalg::matmul::matmul(dst, Accum::Replace, lhs, rhs, alpha, par)`.
- `faer::get_global_parallelism()` / `faer::set_global_parallelism(Par)`; with the `rayon` feature the global default is already `Par::rayon` (shared global rayon pool → `RAYON_NUM_THREADS` controls it).
- `Mat`/`MatRef` have `.adjoint()` views.

**File map:**

| File | Change |
|---|---|
| `crates/aleph-mps/src/mps.rs` | permutation fields + routing, lazy `apply_2q`, faer hot path, faer center moves |
| `crates/aleph-mps/src/tensor.rs` | faer view/from helpers on `Site`; `truncated_svd` faer-native; `thin_qr` deleted; nalgebra group helpers deleted |
| `crates/aleph-mps/src/lib.rs` | crate docs: lazy strategy, parallelism note |
| `crates/aleph-mps/Cargo.toml` | faer `rayon` feature; `[[bench]] wide_bond` |
| `crates/aleph-mps/tests/sv_equivalence.rs` | lazy-perm read oracle + thread-invariance tests |
| `crates/aleph-mps/benches/long_range.rs` | header comment update (lazy now live) |
| `crates/aleph-mps/benches/wide_bond.rs` | NEW: χ=256 brickwall thread-sweep bench |
| `docs/perf/mps_parallel.md` | NEW: EPYC + Ryzen numbers |
| `BACKLOG.md` | flip P3-09 AC checkboxes |

Note: the long-range proptest oracle already exists (`random_long_range_matches_sv` in `tests/sv_equivalence.rs`) — no generator extension needed.

---

## Stage 1 — lazy permutation

### Task 1: Permutation bookkeeping fields

**Files:**
- Modify: `crates/aleph-mps/src/mps.rs` (struct ~line 16, `with_policy` ~line 31, `swap_adjacent` ~line 217)

- [ ] **Step 1: Write the failing test** (in `mod tests` of `mps.rs`)

```rust
#[test]
fn swap_adjacent_updates_permutation_maps() {
    let mut s = MpsState::new(3, 64);
    assert_eq!(s.qubit_of_site, vec![0, 1, 2]);
    assert_eq!(s.site_of_qubit, vec![0, 1, 2]);
    assert_eq!(s.swaps_applied(), 0);
    s.swap_adjacent(1).unwrap();
    assert_eq!(s.qubit_of_site, vec![0, 2, 1]);
    assert_eq!(s.site_of_qubit, vec![0, 2, 1]);
    assert_eq!(s.swaps_applied(), 1);
    s.swap_adjacent(1).unwrap();
    assert_eq!(s.qubit_of_site, vec![0, 1, 2]);
    assert_eq!(s.swaps_applied(), 2);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p aleph-mps swap_adjacent_updates_permutation_maps`
Expected: COMPILE ERROR (`qubit_of_site` does not exist)

- [ ] **Step 3: Implement**

Add fields to `MpsState`:

```rust
#[derive(Debug, Clone)]
pub struct MpsState {
    pub(crate) sites: Vec<Site>,
    pub(crate) center: usize,
    pub(crate) policy: TruncationPolicy,
    pub(crate) trunc_error: f64,
    pub(crate) max_bond_seen: usize,
    /// qubit_of_site[s] = the logical qubit currently stored at site s (P3-09).
    pub(crate) qubit_of_site: Vec<u32>,
    /// site_of_qubit[q] = the site currently holding logical qubit q (P3-09).
    pub(crate) site_of_qubit: Vec<usize>,
    /// Physical nearest-neighbor SWAPs applied so far (lazy-router evidence).
    pub(crate) swaps_applied: u64,
}
```

In `with_policy`, initialise to identity:

```rust
MpsState {
    sites,
    center: 0,
    policy,
    trunc_error: 0.0,
    max_bond_seen: 1,
    qubit_of_site: (0..n as u32).collect(),
    site_of_qubit: (0..n).collect(),
    swaps_applied: 0,
}
```

Add the getter next to `max_bond_reached`:

```rust
/// Number of physical nearest-neighbor SWAP gates applied by the lazy
/// permutation router so far (P3-09).
pub fn swaps_applied(&self) -> u64 {
    self.swaps_applied
}
```

Extend `swap_adjacent` to maintain the maps:

```rust
/// Swap the qubit states on adjacent sites `(k, k+1)` via a SWAP gate and
/// update the site↔qubit permutation accordingly.
fn swap_adjacent(&mut self, k: usize) -> Result<(), MpsError> {
    let g = GateInstance::new(Gate::Swap, vec![k as u32, (k + 1) as u32]);
    let u = crate::gate::matrix_4x4(&g)?;
    self.apply_2q_adjacent(k as u32, (k + 1) as u32, &u)?;
    let qa = self.qubit_of_site[k];
    let qb = self.qubit_of_site[k + 1];
    self.qubit_of_site[k] = qb;
    self.qubit_of_site[k + 1] = qa;
    self.site_of_qubit[qb as usize] = k;
    self.site_of_qubit[qa as usize] = k + 1;
    self.swaps_applied += 1;
    Ok(())
}
```

(`apply_2q` still does forward + reverse ladders at this point, so the maps return to identity after every gate — nothing else breaks.)

- [ ] **Step 4: Run tests**

Run: `cargo test -p aleph-mps`
Expected: ALL PASS (including the new test)

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-mps/src/mps.rs
git commit -m "[P3-09] Add site<->qubit permutation bookkeeping to MpsState

Maps stay identity for now (apply_2q still swaps back); the lazy router
lands after reads are routed through the permutation."
```

### Task 2: Route all reads through the permutation

**Files:**
- Modify: `crates/aleph-mps/src/mps.rs` (`apply_1q` ~line 58, `measure` ~line 577, `sample` ~line 430, `probabilities` ~line 482, `dense_statevector` ~line 382)

- [ ] **Step 1: Write the failing test** (in `mod tests` of `mps.rs`)

```rust
#[test]
fn reads_route_through_permutation() {
    // X(0), then a raw physical swap of sites 0,1: qubit 0 (|1>) now lives
    // at site 1. Every read must still report in logical-qubit order.
    let mut s = MpsState::new(2, 64);
    let x = crate::gate::matrix_2x2(&GateInstance::new(Gate::X, smallvec![0u32])).unwrap();
    s.apply_1q(0, &x);
    s.swap_adjacent(0).unwrap();
    // dense: qubit 0 occupies bit 0 → index 0b01.
    let v = s.dense_statevector();
    assert!((v[0b01].re - 1.0).abs() < 1e-10, "dense not routed");
    // probabilities over qubit 0: [0, 1].
    let p = s.probabilities(&[0]).unwrap();
    assert!((p[1] - 1.0).abs() < 1e-10, "probabilities not routed");
    // sample: qubit 0 packs into bit 0.
    let mut rng = StdRng::seed_from_u64(1);
    assert_eq!(s.sample(3, &mut rng), vec![0b01, 0b01, 0b01], "sample not routed");
    // apply_1q routes: a second X on qubit 0 returns it to |0>.
    s.apply_1q(0, &x);
    let v = s.dense_statevector();
    assert!((v[0b00].re - 1.0).abs() < 1e-10, "apply_1q not routed");
    // measure(0) must read site 1's data: re-flip then measure.
    s.apply_1q(0, &x);
    assert!(s.measure(0, &mut rng).unwrap(), "measure not routed");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p aleph-mps reads_route_through_permutation`
Expected: FAIL (dense reports bit 1 set instead of bit 0, because reads still assume site == qubit)

- [ ] **Step 3: Implement routing**

`apply_1q` — parameter is the *logical qubit*; look up the site:

```rust
/// Apply a 1q unitary to logical qubit `q` (routed to its current site).
/// Preserves canonical form, so neither the center nor any SVD is touched.
pub(crate) fn apply_1q(&mut self, q: usize, u: &[[Complex; 2]; 2]) {
    let site = &mut self.sites[self.site_of_qubit[q]];
    // ... body unchanged ...
}
```

`measure` — collapse at the qubit's current site (replace both `self.move_center_to(q)` and the two `&self.sites[q]` / `&mut self.sites[q]` borrows):

```rust
let s = self.site_of_qubit[q];
self.move_center_to(s);
let site = &self.sites[s];
// ... probability scan unchanged ...
let site = &mut self.sites[s];
// ... collapse unchanged ...
```

`sample` — pack site `i`'s outcome into the logical qubit's bit (replace `bits |= 1u64 << i;`):

```rust
if outcome {
    bits |= 1u64 << work.qubit_of_site[i];
}
```

`probabilities` — measured-site lookup goes through the permutation (replace `out_bit_for_site[q as usize] = Some(pos);`):

```rust
out_bit_for_site[self.site_of_qubit[q as usize]] = Some(pos);
```

`dense_statevector` — site `s` contributes the bit of the qubit it currently holds (rename the loop variable `q` → `s` and replace `prefix | (p << q)`):

```rust
for (s, site) in self.sites.iter().enumerate() {
    // ...
    let new_prefix = prefix | (p << self.qubit_of_site[s] as usize);
    // ...
}
```

`expectation` needs no change: it routes via `apply_1q`, and `overlap` is site-wise between `self` and a clone sharing the same permutation.

- [ ] **Step 4: Run tests**

Run: `cargo test -p aleph-mps`
Expected: ALL PASS (identity permutation makes routing a no-op for every existing test; the new test passes because each read is routed)

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-mps/src/mps.rs
git commit -m "[P3-09] Route measure/sample/probabilities/dense/apply_1q through the permutation

No-op while the permutation is identity; prerequisite for dropping the
swap-back ladder."
```

### Task 3: Site-based `apply_2q_adjacent`

**Files:**
- Modify: `crates/aleph-mps/src/mps.rs` (`apply_2q_adjacent` ~line 107, `apply_2q` ~line 76, `swap_adjacent`)

- [ ] **Step 1: Refactor the signature** — parameters become *site* indices (`usize`), with `s_msb` the site currently holding `g.qubits[0]` (the matrix MSB, ADR-0004):

```rust
/// Apply a 2q unitary `u` to the adjacent sites `(s_msb, s_lsb)`, where
/// `s_msb` currently holds the gate's first qubit (`g.qubits[0]`, the
/// matrix MSB per ADR-0004) and `s_lsb` the second.
///
/// Caller must ensure `s_msb.abs_diff(s_lsb) == 1`.
fn apply_2q_adjacent(
    &mut self,
    s_msb: usize,
    s_lsb: usize,
    u: &[[Complex; 4]; 4],
) -> Result<(), MpsError> {
    let i = s_msb.min(s_lsb);
    let j = i + 1;
    // ... body unchanged except the `out` closure:
    let out = |phys_i: usize, phys_j: usize| -> usize {
        let bit_msb = if s_msb == i { phys_i } else { phys_j };
        let bit_lsb = if s_lsb == i { phys_i } else { phys_j };
        (bit_msb << 1) | bit_lsb
    };
    // ...
}
```

Update callers:
- `swap_adjacent`: `self.apply_2q_adjacent(k, k + 1, &u)?;`
- `apply_2q` adjacent branch: `return self.apply_2q_adjacent(qa as usize, qb as usize, u);`
- `apply_2q` post-ladder call: `self.apply_2q_adjacent(s0, s1, u)?;` where `let (s0, s1) = if qa < qb { (lo as usize, lo as usize + 1) } else { (lo as usize + 1, lo as usize) };` (drop the `u32` casts).

- [ ] **Step 2: Run tests**

Run: `cargo test -p aleph-mps`
Expected: ALL PASS (pure refactor; sites == qubits at the call sites today)

- [ ] **Step 3: Commit**

```bash
git add crates/aleph-mps/src/mps.rs
git commit -m "[P3-09] apply_2q_adjacent takes site indices, not qubit names

MSB orientation is now expressed in sites, which stays correct once the
permutation diverges from identity."
```

### Task 4: Lazy `apply_2q` — drop the swap-back ladder

**Files:**
- Modify: `crates/aleph-mps/src/mps.rs` (`apply_2q`)

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn lazy_swap_counts_amortize() {
    // CNOT(0,4) on n=5: the ladder is 3 SWAPs (always-swap-back paid 6).
    let mut s = MpsState::new(5, 64);
    let h = crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
    s.apply_1q(0, &h);
    let gi = GateInstance::new(Gate::Cnot, smallvec![0u32, 4u32]);
    let u = crate::gate::matrix_4x4(&gi).unwrap();
    s.apply_2q(&gi, &u).unwrap();
    assert_eq!(s.swaps_applied(), 3);
    // Qubit 4 stayed next to qubit 0 → repeating the gate costs 0 SWAPs.
    s.apply_2q(&gi, &u).unwrap();
    assert_eq!(s.swaps_applied(), 3);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p aleph-mps lazy_swap_counts_amortize`
Expected: FAIL — `swaps_applied() == 6` after the first gate (forward + reverse ladders), 12 after the second

- [ ] **Step 3: Implement the lazy router**

Replace `apply_2q` entirely:

```rust
/// Apply a 2q gate (4×4 matrix `u`) on the qubits named by `g`
/// (`g.qubits[0]`=MSB, ADR-0004). The qubits' current sites come from the
/// lazy permutation: non-adjacent sites are brought together by moving the
/// qubit at the higher site down with nearest-neighbor SWAPs, and the
/// permutation is left in place afterwards — no swap-back (P3-09). Reads
/// route through the permutation, so `site == qubit` is no longer an
/// invariant. `MpsError::NonNearestNeighbor` is retained as a defensive
/// variant but is no longer reached on the normal 2q path.
pub(crate) fn apply_2q(
    &mut self,
    g: &GateInstance,
    u: &[[Complex; 4]; 4],
) -> Result<(), MpsError> {
    let qa = g.qubits[0] as usize;
    let qb = g.qubits[1] as usize;
    let sa = self.site_of_qubit[qa];
    let sb = self.site_of_qubit[qb];
    if sa.abs_diff(sb) != 1 {
        // Ladder: walk the occupant of the higher site down to lo+1.
        let lo = sa.min(sb);
        let hi = sa.max(sb);
        for k in (lo + 1..hi).rev() {
            self.swap_adjacent(k)?;
        }
    }
    // Re-resolve sites: the ladder moved one of the qubits.
    self.apply_2q_adjacent(self.site_of_qubit[qa], self.site_of_qubit[qb], u)
}
```

- [ ] **Step 4: Run the full crate suite (existing non-adjacent tests are the oracle)**

Run: `cargo test -p aleph-mps`
Expected: ALL PASS — `ghz_via_nonadjacent_cnots`, `swap_via_nonadjacent`, `nonadjacent_matches_sv`, `random_long_range_matches_sv` (256 proptest cases) all read through the routed paths

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-mps/src/mps.rs
git commit -m "[P3-09] Lazy permutation routing: drop the swap-back ladder

A long-range 2q gate now costs (d-1) SWAPs instead of 2(d-1), and
consecutive long-range gates on nearby qubits amortize to zero."
```

### Task 5: Permutation oracle tests

**Files:**
- Modify: `crates/aleph-mps/src/mps.rs` (`mod tests`)
- Modify: `crates/aleph-mps/tests/sv_equivalence.rs`

- [ ] **Step 1: Add the map-inverse proptest** (in `mps.rs` `mod tests`; add `use proptest::prelude::*;` inside the test module)

```rust
proptest::proptest! {
    /// After any random long-range circuit the two maps stay mutually inverse.
    #[test]
    fn permutation_maps_stay_inverse(seq in proptest::collection::vec((0u8..5, 0u8..5, 0u8..5), 0..20)) {
        let n = 5u32;
        let mut s = MpsState::new(n as usize, 64);
        let h = crate::gate::matrix_2x2(&GateInstance::new(Gate::H, smallvec![0u32])).unwrap();
        for (op, x, y) in seq {
            let a = (x as u32) % n;
            match op {
                0 | 1 => s.apply_1q(a as usize, &h),
                _ => {
                    let b = (y as u32) % n;
                    if a != b {
                        let gi = GateInstance::new(Gate::Cnot, smallvec![a, b]);
                        let u = crate::gate::matrix_4x4(&gi).unwrap();
                        s.apply_2q(&gi, &u).unwrap();
                    }
                }
            }
        }
        for q in 0..n as usize {
            proptest::prop_assert_eq!(s.qubit_of_site[s.site_of_qubit[q]] as usize, q);
        }
        for site in 0..n as usize {
            proptest::prop_assert_eq!(s.site_of_qubit[s.qubit_of_site[site] as usize], site);
        }
    }
}
```

- [ ] **Step 2: Add the routed-reads oracle vs SV** (in `tests/sv_equivalence.rs`)

```rust
#[test]
fn lazy_perm_reads_match_sv() {
    // Long-range gates leave a non-identity permutation; every read API
    // must still report in logical-qubit order (P3-09).
    let n = 5u32;
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(g(Gate::H, &[q])).unwrap();
    }
    c.add_gate(g(Gate::Cnot, &[0, 4])).unwrap(); // distance 4
    c.add_gate(g(Gate::Rz(Param::Concrete(0.4)), &[2])).unwrap();
    c.add_gate(g(Gate::Cnot, &[3, 1])).unwrap(); // reversed, distance 2
    c.add_gate(g(Gate::Cz, &[4, 2])).unwrap(); // distance 2 after permutation drift

    let a = mps_dense(&c, 64);
    let b = sv_dense(&c);
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).norm() < 1e-10, "dense mismatch under permutation");
    }

    let mut mps = MpsBackend::with_seed(0).with_max_bond(64);
    let ms = run(&mut mps, &c).unwrap();
    assert!(ms.swaps_applied() > 0, "circuit must exercise the lazy router");
    let mut sv = NaiveSvBackend::with_seed(0);
    let svs = run(&mut sv, &c).unwrap();

    for subset in [vec![0u32], vec![4, 0], vec![1, 3, 2]] {
        let pm = mps.probabilities(&ms, &subset).unwrap();
        let ps = sv.probabilities(&svs, &subset).unwrap();
        for (x, y) in pm.iter().zip(ps.iter()) {
            assert!((x - y).abs() < 1e-10, "probabilities mismatch under permutation");
        }
    }
    for terms in [
        vec![(0u32, Pauli::Z), (4, Pauli::Z)],
        vec![(2, Pauli::X)],
        vec![(1, Pauli::Z), (3, Pauli::Z)],
    ] {
        let p = PauliString::new(1.0, terms).unwrap();
        let em = mps.expectation_value(&ms, &p).unwrap();
        let es = sv.expectation_value(&svs, &p).unwrap();
        assert!((em - es).abs() < 1e-10, "expectation mismatch: {em} vs {es}");
    }
}

#[test]
fn lazy_perm_sample_matches_probabilities() {
    // Sampling under a non-identity permutation: empirical distribution over
    // all qubits must match the exact marginals.
    let n = 4u32;
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(g(Gate::H, &[q])).unwrap();
    }
    c.add_gate(g(Gate::Cnot, &[0, 3])).unwrap();
    c.add_gate(g(Gate::Cnot, &[2, 0])).unwrap();
    let mut be = MpsBackend::with_seed(7).with_max_bond(64);
    let st = run(&mut be, &c).unwrap();
    assert!(st.swaps_applied() > 0);
    let shots = be.sample(&st, 20000).unwrap();
    let mut counts = [0u32; 16];
    for sh in &shots {
        counts[*sh as usize] += 1;
    }
    let probs = be.probabilities(&st, &[0, 1, 2, 3]).unwrap();
    for idx in 0..16 {
        let emp = counts[idx] as f64 / 20000.0;
        assert!((emp - probs[idx]).abs() < 0.02, "idx {idx}: {emp} vs {}", probs[idx]);
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p aleph-mps`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-mps/src/mps.rs crates/aleph-mps/tests/sv_equivalence.rs
git commit -m "[P3-09] Oracle tests for routed reads under non-identity permutation"
```

### Task 6: Stage-1 docs

**Files:**
- Modify: `crates/aleph-mps/src/lib.rs` (lines 43–47, the always-swap-back paragraph)
- Modify: `crates/aleph-mps/src/mps.rs` (struct doc comment, `MpsState` header)
- Modify: `crates/aleph-mps/benches/long_range.rs` (lines 1–5, header comment)

- [ ] **Step 1: Update the prose**

`lib.rs` — replace the final paragraph (lines 43–47) with:

```rust
//! 2q gates between non-adjacent qubits are handled by a lazy SWAP network
//! (P3-09): the qubits are brought together with nearest-neighbor SWAPs and
//! the resulting site↔qubit permutation is tracked rather than undone —
//! `(d-1)` SWAPs per long-range gate instead of `2(d-1)`, amortizing to zero
//! for repeated gates on nearby qubits. All reads (measure, sample,
//! probabilities, expectation, dense reconstruction) route through the
//! permutation, so results are always reported in logical-qubit order.
```

`mps.rs` struct doc — append to the `MpsState` doc comment:

```rust
/// Sites hold logical qubits per the `qubit_of_site`/`site_of_qubit`
/// permutation (lazy SWAP routing, P3-09); `site == qubit` only until the
/// first long-range 2q gate.
```

`long_range.rs` header — replace lines 1–5 with:

```rust
//! Wall-clock cost of a single non-adjacent 2q gate as a function of qubit
//! distance. Since P3-09 the lazy permutation router applies `(distance-1)`
//! nearest-neighbor SWAPs (no swap-back); compare against the `main`
//! baseline to see the always-swap-back → lazy improvement.
```

- [ ] **Step 2: Verify docs build and tests still pass**

Run: `cargo test -p aleph-mps --doc && cargo test -p aleph-mps`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git add crates/aleph-mps/src/lib.rs crates/aleph-mps/src/mps.rs crates/aleph-mps/benches/long_range.rs
git commit -m "[P3-09] Document the lazy SWAP permutation strategy"
```

---

## Stage 2 — multithreading

### Task 7: Enable faer's rayon backend

**Files:**
- Modify: `crates/aleph-mps/Cargo.toml`

- [ ] **Step 1: Add the feature**

```toml
faer = { version = "0.24.0", default-features = false, features = ["linalg", "std", "rayon"] }
```

- [ ] **Step 2: Verify**

Run: `cargo test -p aleph-mps && cargo clippy -p aleph-mps --all-targets -- -D warnings`
Expected: ALL PASS. With the `rayon` feature, faer's global parallelism defaults to the shared rayon pool — `thin_svd` parallelizes immediately; tolerance-based tests (1e-9/1e-10) are unaffected by the rounding differences.

- [ ] **Step 3: Commit**

```bash
git add crates/aleph-mps/Cargo.toml Cargo.lock
git commit -m "[P3-09] Enable faer rayon parallelism (shared global rayon pool)"
```

### Task 8: Zero-copy faer views on `Site`

**Files:**
- Modify: `crates/aleph-mps/src/tensor.rs`

- [ ] **Step 1: Write the failing test** (in `mod tests` of `tensor.rs`)

```rust
#[test]
fn faer_views_match_element_access() {
    let mut s = Site::zeros(2, 3);
    for l in 0..2 {
        for p in 0..2 {
            for r in 0..3 {
                *s.get_mut(l, p, r) = Complex::new((l * 100 + p * 10 + r) as f64, 0.5);
            }
        }
    }
    let gl = s.group_left_view(); // (left*2) x right
    assert_eq!((gl.nrows(), gl.ncols()), (4, 3));
    for l in 0..2 {
        for p in 0..2 {
            for r in 0..3 {
                assert_eq!(gl[(l * 2 + p, r)], s.get(l, p, r));
            }
        }
    }
    let gr = s.group_right_view(); // left x (2*right)
    assert_eq!((gr.nrows(), gr.ncols()), (2, 6));
    for l in 0..2 {
        for p in 0..2 {
            for r in 0..3 {
                assert_eq!(gr[(l, p * 3 + r)], s.get(l, p, r));
            }
        }
    }
}

#[test]
fn faer_from_group_roundtrip() {
    let mut s = Site::zeros(2, 3);
    for (k, v) in s.data.iter_mut().enumerate() {
        *v = Complex::new(k as f64, -(k as f64));
    }
    let back_l = Site::from_group_left_faer(s.group_left_view(), 2, 3);
    assert_eq!(back_l, s);
    let back_r = Site::from_group_right_faer(s.group_right_view(), 2, 3);
    assert_eq!(back_r, s);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p aleph-mps faer_views`
Expected: COMPILE ERROR (`group_left_view` does not exist)

- [ ] **Step 3: Implement** (in `impl Site`; `aleph_core::Complex` IS `faer::c64`, so the views need no conversion)

```rust
/// Zero-copy faer view of the grouped-left matrix `(left·2) × right`
/// (row `l·2+p`, col `r`) — identical to the row-major layout of `data`.
pub fn group_left_view(&self) -> faer::MatRef<'_, Complex> {
    faer::MatRef::from_row_major_slice(&self.data, self.left * 2, self.right)
}

/// Zero-copy faer view of the grouped-right matrix `left × (2·right)`
/// (row `l`, col `p·right + r`) — the same bytes, regrouped.
pub fn group_right_view(&self) -> faer::MatRef<'_, Complex> {
    faer::MatRef::from_row_major_slice(&self.data, self.left, 2 * self.right)
}

/// Build a `Site` from a faer `(left·2) × right` grouped-left matrix.
pub fn from_group_left_faer(m: faer::MatRef<'_, Complex>, left: usize, right: usize) -> Site {
    let mut s = Site::zeros(left, right);
    for row in 0..left * 2 {
        for r in 0..right {
            s.data[row * right + r] = m[(row, r)];
        }
    }
    s
}

/// Build a `Site` from a faer `left × (2·right)` grouped-right matrix.
pub fn from_group_right_faer(m: faer::MatRef<'_, Complex>, left: usize, right: usize) -> Site {
    let mut s = Site::zeros(left, right);
    for l in 0..left {
        for col in 0..2 * right {
            s.data[l * 2 * right + col] = m[(l, col)];
        }
    }
    s
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p aleph-mps`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-mps/src/tensor.rs
git commit -m "[P3-09] Zero-copy faer views over Site data

Row-major (l*2+p)*right+r is simultaneously the grouped-left and
grouped-right matrix; no reshape copies needed for the faer hot path."
```

### Task 9: faer-native `truncated_svd` + gemm theta-build in `apply_2q_adjacent`

**Files:**
- Modify: `crates/aleph-mps/src/tensor.rs` (`truncated_svd`, its tests)
- Modify: `crates/aleph-mps/src/mps.rs` (`apply_2q_adjacent`)

- [ ] **Step 1: Make `truncated_svd` faer-native**

New signature and the diff from the current body — the input is already a faer matrix, so the element-by-element copy into `faer::Mat` disappears, and the outputs become faer matrices:

```rust
/// `(u_kept, s_kept, vt_kept, discarded_weight)` returned by [`truncated_svd`].
pub type TruncatedSvd = (faer::Mat<Complex>, Vec<f64>, faer::Mat<Complex>, f64);

pub fn truncated_svd(
    m: faer::MatRef<'_, Complex>,
    policy: &TruncationPolicy,
) -> Result<TruncatedSvd, MpsError> {
    let rows = m.nrows();
    let cols = m.ncols();
    let svd = m.thin_svd().map_err(|_| MpsError::SvdFailed)?;
    let fu = svd.U();
    let fv = svd.V();
    let fs = svd.S();
    let k = fs.column_vector().nrows(); // = min(rows, cols)
    let sigmas: Vec<f64> = (0..k).map(|t| fs[t].re).collect();
    // ... `significant` / suffix_sq / chi / discarded / scale logic UNCHANGED ...
    let u_kept = faer::Mat::from_fn(rows, chi, |r, t| fu[(r, t)]);
    // vt row t = (t-th right singular vector)^H = conjugate of V's column t.
    let vt_kept = faer::Mat::from_fn(chi, cols, |t, c| fv[(c, t)].conj());
    let s_kept: Vec<f64> = (0..chi).map(|t| sigmas[t] * scale).collect();
    Ok((u_kept, s_kept, vt_kept, discarded))
}
```

Keep the `# Why faer` doc comment. Drop `use nalgebra::DMatrix;` from `tensor.rs` only if nothing else in the file still uses it (the old group helpers do, until Task 10).

- [ ] **Step 2: Migrate `truncated_svd` tests in `tensor.rs`** — replace every `DMatrix::from_fn(...)` test-input with `faer::Mat::from_fn(...)` and pass `m.as_ref()`; indexing syntax `m[(i, j)]` is the same. Example for `truncated_svd_reconstructs_complex_full_rank`:

```rust
let m = faer::Mat::from_fn(4, 4, |i, j| {
    Complex::new(
        (i as f64 - j as f64) * 0.3 + 1.0,
        (i * 2 + j) as f64 * 0.17 - 0.5,
    )
});
let fro: f64 = (0..4)
    .flat_map(|i| (0..4).map(move |j| (i, j)))
    .map(|(i, j)| m[(i, j)].norm_sqr())
    .sum::<f64>()
    .sqrt();
let (u, s, vt, _disc) = truncated_svd(m.as_ref(), &TruncationPolicy::FixedBond(64)).unwrap();
// reconstruction loop unchanged (faer indexing is also m[(r, c)])
```

Apply the same mechanical change to `truncated_svd_rank1_complex_collapses_to_chi1`, `diag_sigma()` (return `faer::Mat<Complex>`), and the four policy tests.

- [ ] **Step 3: Rewrite `apply_2q_adjacent` on the faer hot path**

Add imports at the top of `mps.rs`:

```rust
use faer::linalg::matmul::matmul;
use faer::Accum;
```

Replace the body from the theta-build through the site write-back (center move and dims stay):

```rust
let li = self.sites[i].left;
let ri = self.sites[j].right;

// Θ as a (li·2) × (2·ri) matrix (row l·2+a, col b·ri+r): exactly the
// grouped-left × grouped-right product — one parallel gemm (P3-09).
let mut theta = faer::Mat::<Complex>::zeros(li * 2, 2 * ri);
matmul(
    theta.as_mut(),
    Accum::Replace,
    self.sites[i].group_left_view(),
    self.sites[j].group_right_view(),
    Complex::new(1.0, 0.0),
    faer::get_global_parallelism(),
);

let out = |phys_i: usize, phys_j: usize| -> usize {
    let bit_msb = if s_msb == i { phys_i } else { phys_j };
    let bit_lsb = if s_lsb == i { phys_i } else { phys_j };
    (bit_msb << 1) | bit_lsb
};

// Θ' = U·Θ over the joint physical index — O(16·li·ri), a factor χ
// cheaper than the gemm above, so plain loops are fine here.
let mut theta2 = faer::Mat::<Complex>::zeros(li * 2, 2 * ri);
for ap in 0..2usize {
    for bp in 0..2usize {
        let row_u = out(ap, bp);
        for a in 0..2usize {
            for b in 0..2usize {
                let u_entry = u[row_u][out(a, b)];
                if u_entry == Complex::new(0.0, 0.0) {
                    continue;
                }
                for l in 0..li {
                    for r in 0..ri {
                        theta2[(l * 2 + ap, bp * ri + r)] +=
                            u_entry * theta[(l * 2 + a, b * ri + r)];
                    }
                }
            }
        }
    }
}

// Truncated SVD of Θ' (already in (li·2) × (2·ri) grouped form).
let (u_s, s_kept, vt_s, discarded) = truncated_svd(theta2.as_ref(), &self.policy)?;
self.trunc_error += discarded;
let chi = s_kept.len();
self.max_bond_seen = self.max_bond_seen.max(chi);

// New site i: left-canonical from the U factor, shape (li, chi).
self.sites[i] = Site::from_group_left_faer(u_s.as_ref(), li, chi);

// New site j: singular values folded into Vᴴ rows, shape (chi, ri).
let mut sv = vt_s;
for t in 0..chi {
    for c in 0..2 * ri {
        sv[(t, c)] *= Complex::new(s_kept[t], 0.0);
    }
}
self.sites[j] = Site::from_group_right_faer(sv.as_ref(), chi, ri);
self.center = j;

Ok(())
```

Note the `mi` variable and the old `theta`/`theta2` flat vecs and the `DMatrix::from_fn` reshape all disappear.

- [ ] **Step 4: Run the full suite**

Run: `cargo test -p aleph-mps`
Expected: ALL PASS — the SWAP-network proptest oracle (`random_long_range_matches_sv`, `regression_svd_norm_loss_seq`) gates this rewrite

- [ ] **Step 5: Commit**

```bash
git add crates/aleph-mps/src/tensor.rs crates/aleph-mps/src/mps.rs
git commit -m "[P3-09] faer-native truncated SVD + theta-build as one parallel gemm

The two-site contraction is grouped-left x grouped-right on zero-copy
views; the intermediate flat vecs and nalgebra reshape are gone."
```

### Task 10: Center moves and bond absorption on faer

**Files:**
- Modify: `crates/aleph-mps/src/mps.rs` (`move_center_right` ~line 270, `move_center_left` ~line 284; DELETE `absorb_into_left`, `absorb_into_right`)
- Modify: `crates/aleph-mps/src/tensor.rs` (DELETE `thin_qr`, `to_group_left`, `from_group_left`, `to_group_right`, `from_group_right` and their roundtrip tests)

- [ ] **Step 1: Rewrite the center moves**

```rust
/// Shift center right from `i` to `i+1` using thin QR on the grouped-left
/// view. Site `i` becomes left-canonical; the R factor is absorbed into
/// site `i+1`'s left bond via a parallel gemm.
fn move_center_right(&mut self) {
    let i = self.center;
    let left = self.sites[i].left;
    let qr = self.sites[i].group_left_view().qr();
    let q = qr.compute_thin_Q(); // (left·2) × k
    let r = qr.thin_R(); // k × right
    let k = q.ncols();
    let next_right = self.sites[i + 1].right;
    // A'[l',p,r2] = Σ_l R[l',l] · A[l,p,r2]  ==  R · group_right(A).
    let mut absorbed = faer::Mat::<Complex>::zeros(k, 2 * next_right);
    matmul(
        absorbed.as_mut(),
        Accum::Replace,
        r,
        self.sites[i + 1].group_right_view(),
        Complex::new(1.0, 0.0),
        faer::get_global_parallelism(),
    );
    self.sites[i + 1] = Site::from_group_right_faer(absorbed.as_ref(), k, next_right);
    self.sites[i] = Site::from_group_left_faer(q.as_ref(), left, k);
    self.center += 1;
}

/// Shift center left from `i` to `i-1` using thin QR on the adjoint of the
/// grouped-right view (LQ decomposition). Site `i` becomes right-canonical;
/// the Rᴴ factor is absorbed into site `i-1`'s right bond via a gemm.
fn move_center_left(&mut self) {
    let i = self.center;
    let right = self.sites[i].right;
    let qr = self.sites[i].group_right_view().adjoint().qr();
    let q = qr.compute_thin_Q(); // (2·right) × k
    let r = qr.thin_R(); // k × left
    let k = q.ncols();
    let prev_left = self.sites[i - 1].left;
    // A'[l2,p,r'] = Σ_r A[l2,p,r] · Rᴴ[r,r']  ==  group_left(A) · Rᴴ.
    let mut absorbed = faer::Mat::<Complex>::zeros(prev_left * 2, k);
    matmul(
        absorbed.as_mut(),
        Accum::Replace,
        self.sites[i - 1].group_left_view(),
        r.adjoint(),
        Complex::new(1.0, 0.0),
        faer::get_global_parallelism(),
    );
    self.sites[i - 1] = Site::from_group_left_faer(absorbed.as_ref(), prev_left, k);
    self.sites[i] = Site::from_group_right_faer(q.as_ref().adjoint(), k, right);
    self.center -= 1;
}
```

- [ ] **Step 2: Delete dead code**

- `mps.rs`: delete `absorb_into_left` and `absorb_into_right`.
- `tensor.rs`: delete `thin_qr`, `to_group_left`, `from_group_left`, `to_group_right`, `from_group_right`, the `group_left_roundtrip`/`group_right_roundtrip` tests, and the now-unused `use nalgebra::DMatrix;` import. (`mps.rs` keeps its own `nalgebra::DMatrix` import for the transfer contractions in `overlap`/`probabilities` — out of P3-09 scope.)

- [ ] **Step 3: Run the full suite + clippy**

Run: `cargo test -p aleph-mps && cargo clippy -p aleph-mps --all-targets -- -D warnings`
Expected: ALL PASS — `move_center_*` canonical-form tests and the full oracle suite gate the rewrite

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-mps/src/mps.rs crates/aleph-mps/src/tensor.rs
git commit -m "[P3-09] Center moves on faer thin QR; bond absorption as parallel gemm

nalgebra remains only in the overlap/probabilities transfer contractions
(read path, out of P3-09 scope)."
```

### Task 11: Thread-invariance test

**Files:**
- Modify: `crates/aleph-mps/tests/sv_equivalence.rs`

- [ ] **Step 1: Write the test**

```rust
#[test]
fn results_invariant_across_parallelism() {
    // Same circuit under sequential and rayon-parallel faer must agree to
    // 1e-10 (not bit-exact: parallel SVD may round differently).
    let n = 8u32;
    let mut c = aleph_ir::Circuit::new(n, 0);
    for q in 0..n {
        c.add_gate(g(Gate::H, &[q])).unwrap();
    }
    for layer in 0..4u32 {
        let start = layer % 2;
        let mut q = start;
        while q + 1 < n {
            c.add_gate(g(Gate::Ry(Param::Concrete(0.3 + q as f64 * 0.11)), &[q]))
                .unwrap();
            c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap();
            q += 2;
        }
    }
    c.add_gate(g(Gate::Cnot, &[0, 7])).unwrap(); // exercise the lazy router too

    let prev = faer::get_global_parallelism();
    faer::set_global_parallelism(faer::Par::Seq);
    let a = mps_dense(&c, 128);
    faer::set_global_parallelism(faer::Par::rayon(0));
    let b = mps_dense(&c, 128);
    faer::set_global_parallelism(prev);
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).norm() < 1e-10, "parallelism changed the state");
    }
}
```

Add `faer` to `[dev-dependencies]` in `crates/aleph-mps/Cargo.toml`:

```toml
faer = { version = "0.24.0", default-features = false, features = ["linalg", "std", "rayon"] }
```

(Other tests only assert tolerances, so the global-parallelism toggling cannot make a concurrently running test flaky.)

- [ ] **Step 2: Run**

Run: `cargo test -p aleph-mps results_invariant_across_parallelism`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add crates/aleph-mps/tests/sv_equivalence.rs crates/aleph-mps/Cargo.toml
git commit -m "[P3-09] Thread-invariance oracle: Par::Seq vs Par::rayon to 1e-10"
```

### Task 12: Wide-bond benchmark

**Files:**
- Create: `crates/aleph-mps/benches/wide_bond.rs`
- Modify: `crates/aleph-mps/Cargo.toml`

- [ ] **Step 1: Register the bench**

```toml
[[bench]]
name = "wide_bond"
harness = false
```

- [ ] **Step 2: Write the bench**

```rust
//! Wide-bond MPS benchmark: a random brickwall whose central bond saturates
//! the χ cap, so the per-gate cost is dominated by the (2χ)×(2χ) SVD and the
//! theta gemm — the surfaces parallelized in P3-09 (AC-2).
//!
//! Thread sweep: `RAYON_NUM_THREADS=1|2|4|8|16 cargo bench -p aleph-mps --bench wide_bond`
//! (faer shares the global rayon pool).

use aleph_backend::run;
use aleph_core::{Gate, GateInstance, Param};
use aleph_mps::MpsBackend;
use criterion::{criterion_group, criterion_main, Criterion};

fn g(gate: Gate, qubits: &[u32]) -> GateInstance {
    GateInstance::new(gate, qubits.to_vec())
}

/// Brickwall of parameterized 2q blocks; depth `layers` doubles the exact
/// central Schmidt rank per layer, so χ saturates the cap after ~log2(χ)+1
/// layers and the remaining layers run at full bond dimension.
fn brickwall(n: u32, layers: u32) -> aleph_ir::Circuit {
    let mut c = aleph_ir::Circuit::new(n, 0);
    let mut t = 0.1f64;
    for q in 0..n {
        c.add_gate(g(Gate::H, &[q])).unwrap();
    }
    for layer in 0..layers {
        let mut q = layer % 2;
        while q + 1 < n {
            c.add_gate(g(Gate::Ry(Param::Concrete(t)), &[q])).unwrap();
            c.add_gate(g(Gate::Ry(Param::Concrete(t * 1.3 + 0.2)), &[q + 1]))
                .unwrap();
            c.add_gate(g(Gate::Cnot, &[q, q + 1])).unwrap();
            c.add_gate(g(Gate::Rz(Param::Concrete(t * 0.7 + 0.1)), &[q + 1]))
                .unwrap();
            t += 0.37;
            q += 2;
        }
    }
    c
}

fn bench(cr: &mut Criterion) {
    let mut grp = cr.benchmark_group("wide_bond_brickwall");
    grp.sample_size(10);
    for (n, chi, layers) in [(20u32, 128usize, 10u32), (24, 256, 12)] {
        let c = brickwall(n, layers);
        grp.bench_function(format!("n{n}_chi{chi}_d{layers}"), |b| {
            b.iter(|| {
                let mut be = MpsBackend::with_seed(0).with_max_bond(chi);
                run(&mut be, &c).unwrap()
            })
        });
    }
    grp.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
```

- [ ] **Step 3: Smoke-run locally (short)**

Run: `cargo bench -p aleph-mps --bench wide_bond -- --test`
Expected: both cells execute without error (`--test` runs one iteration, no statistics)

- [ ] **Step 4: Commit**

```bash
git add crates/aleph-mps/benches/wide_bond.rs crates/aleph-mps/Cargo.toml
git commit -m "[P3-09] wide_bond bench: brickwall saturating chi for the thread sweep"
```

### Task 13: Full local validation

- [ ] **Step 1: Workspace gates (what CI runs)**

Run:
```bash
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
```
Expected: ALL GREEN

- [ ] **Step 2: x86_64 cross-check (catches target-specific issues without EPYC)**

Run: `cargo check -p aleph-mps --target x86_64-unknown-linux-gnu`
Expected: clean (P2-04 lesson: this catches x86-only breakage on the aarch64 dev box)

- [ ] **Step 3: Push the branch**

```bash
git push -u origin p3-09-mps-lazy-swap-multithreading
```
Expected: CI (build/test/clippy/fmt) green on GitHub before the perf trip

### Task 14: EPYC measurement (AC-1 + AC-2)

Box: `ssh root@195.154.249.85` (EPYC 8124P, 16c, AVX-512). Cargo lives at `~/.rustup/toolchains/*/bin/cargo`, not on PATH.

- [ ] **Step 1: Verify the box is idle** (CLAUDE.md rule — CI shares this machine)

```bash
ssh root@195.154.249.85 'uptime; pgrep -af "cargo bench|bencher run|Runner.Worker" || echo IDLE'
```
Expected: load ≈ 0 and `IDLE`. If not idle, wait — do not measure.

- [ ] **Step 2: Baseline on `main`** (always-swap-back + single-thread)

```bash
ssh root@195.154.249.85 'cd ~/aleph && git fetch origin && git checkout origin/main && \
  ~/.rustup/toolchains/*/bin/cargo bench -p aleph-mps --bench long_range -- --save-baseline main && \
  ~/.rustup/toolchains/*/bin/cargo bench -p aleph-mps --bench nn_qaoa -- --save-baseline main'
```
(`wide_bond` doesn't exist on main; its baseline is the branch's own `RAYON_NUM_THREADS=1` run.)

- [ ] **Step 3: Branch numbers**

```bash
ssh root@195.154.249.85 'cd ~/aleph && git checkout p3-09-mps-lazy-swap-multithreading && \
  ~/.rustup/toolchains/*/bin/cargo bench -p aleph-mps --bench long_range -- --baseline main && \
  ~/.rustup/toolchains/*/bin/cargo bench -p aleph-mps --bench nn_qaoa -- --baseline main'
```
Expected: `long_range` dist4/8/11 improve materially (≈half the SWAPs + parallel SVD); dist1 ≈ flat; `nn_qaoa` flat-to-better (no long-range gates — guards against regression).

- [ ] **Step 4: Thread sweep on wide_bond**

```bash
ssh root@195.154.249.85 'cd ~/aleph && for t in 1 2 4 8 16; do \
  RAYON_NUM_THREADS=$t ~/.rustup/toolchains/*/bin/cargo bench -p aleph-mps --bench wide_bond 2>&1 | tee /tmp/wide_bond_t$t.log; done'
```
Expected: monotone speedup from t=1 to t=8/16 on `n24_chi256_d12` (AC-2 evidence). Record the table.

- [ ] **Step 5: Re-verify idleness right after measuring** (post-hoc contamination check)

```bash
ssh root@195.154.249.85 'uptime; pgrep -af "cargo bench|Runner.Worker" | grep -v $$ || echo STILL-CLEAN'
```

### Task 15: Ryzen measurement (second data point)

Box: `ssh root@49.12.173.85` (Ryzen 9 3900, 12c/24t, NO AVX-512). Its `origin` is a LOCAL bundle, not GitHub (P2-02 lesson) — ship a fresh bundle and verify HEAD.

- [ ] **Step 1: Ship the branch as a bundle**

```bash
git bundle create /tmp/p3-09.bundle origin/main p3-09-mps-lazy-swap-multithreading
scp /tmp/p3-09.bundle root@49.12.173.85:/tmp/
ssh root@49.12.173.85 'cd ~/aleph && git fetch /tmp/p3-09.bundle p3-09-mps-lazy-swap-multithreading:p3-09 && git checkout p3-09 && git log --oneline -1'
```
Expected: HEAD matches the local branch tip hash — verify before trusting any number.

- [ ] **Step 2: Thread sweep**

```bash
ssh root@49.12.173.85 'cd ~/aleph && for t in 1 2 4 8 12; do \
  RAYON_NUM_THREADS=$t cargo bench -p aleph-mps --bench wide_bond 2>&1 | tee /tmp/wide_bond_t$t.log; done'
```
Expected: speedup curve (likely shallower than EPYC — scalar path, narrower memory).

### Task 16: Perf note, BACKLOG, PR

**Files:**
- Create: `docs/perf/mps_parallel.md`
- Modify: `BACKLOG.md` (P3-09 AC checkboxes, ~line 1804)

- [ ] **Step 1: Write `docs/perf/mps_parallel.md`** with the structure:

```markdown
# P3-09: MPS lazy SWAP routing + multithreading

Date: <fill>. Boxes: EPYC 8124P (16c, AVX-512), Ryzen 9 3900 (12c/24t, scalar).
Branch `p3-09-mps-lazy-swap-multithreading` @ <hash> vs `main` @ <hash>.

## Lazy permutation (AC-1)
| bench | main (swap-back) | P3-09 (lazy) | ratio | SWAPs before/after |
(long_range dist1/4/8/11 rows; SWAP counts from swaps_applied: 2(d-1) -> d-1)

## Thread sweep, wide_bond (AC-2)
| threads | EPYC n24_chi256 | speedup | Ryzen n24_chi256 | speedup |
(t = 1/2/4/8/16; baseline = t1)

## Honest notes
(any cell that regressed or stayed flat, and why; nn_qaoa guard numbers)
```

Fill every cell with measured numbers from Tasks 14–15. Honest reporting: if 16t is flat vs 8t (bandwidth ceiling — the Phase-2 pattern), say so.

- [ ] **Step 2: Flip the two AC checkboxes** in the P3-09 section of `BACKLOG.md` (`- [ ]` → `- [x]`).

- [ ] **Step 3: Commit and push**

```bash
git add docs/perf/mps_parallel.md BACKLOG.md
git commit -m "[P3-09] Perf report: lazy SWAP + thread sweep on EPYC/Ryzen"
git push
```

- [ ] **Step 4: Open the PR**

```bash
gh pr create --title "[P3-09] MPS multithreading + lazy SWAP permutation tracking" --body "$(cat <<'EOF'
Closes #125

## Summary
- Stage 1: lazy site<->qubit permutation tracking — long-range 2q gates cost (d-1) NN SWAPs instead of 2(d-1) and amortize across consecutive gates; all reads (measure/sample/probabilities/expectation/dense) route through the permutation.
- Stage 2: faer rayon parallelism — theta-build and bond absorption as parallel gemms on zero-copy views, thin SVD/QR on faer; nalgebra remains only in the overlap/probabilities transfer contractions.

## Test results
<paste: cargo test --workspace summary; oracle counts>

## Benchmarks
<paste: long_range before/after table, swaps_applied counts, wide_bond thread-sweep EPYC + Ryzen tables from docs/perf/mps_parallel.md>

## Notes / follow-ups
<honest flat cells; transfer-contraction parallelism deferred (read path)>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 5: CI green, self-review the diff, merge** per the standard workflow (let it sit, re-review, squash-merge).

---

## Self-review checklist (done at plan-writing time)

- **Spec coverage:** lazy fields/routing (Tasks 1–5), docs (6), faer rayon (7), views (8), gemm+SVD (9), QR/absorb (10), thread invariance (11), wide-bond bench (12), EPYC+Ryzen evidence (14–15), AC mapping + delivery (16). Transfer contractions explicitly out of scope (spec §Stage 2). ✓
- **Type consistency:** `truncated_svd(faer::MatRef<Complex>, &TruncationPolicy) -> (Mat, Vec<f64>, Mat, f64)` defined in Task 9, used in Task 9's `apply_2q_adjacent`; `group_left_view`/`group_right_view`/`from_group_left_faer`/`from_group_right_faer` defined in Task 8, used in 9–10; `swaps_applied()` defined in Task 1, used in 4–5 and the PR. `apply_2q_adjacent(s_msb: usize, s_lsb: usize, u)` defined in Task 3, used in 4 and rewritten in 9 with the same signature. ✓
- **Ordering safety:** reads are routed (Task 2) *before* the lazy router lands (Task 4), so the suite stays green at every commit. ✓
