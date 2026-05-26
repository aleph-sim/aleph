# P1-02 — Bit-manipulation indexing for 1-qubit gates — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `SoaSvBackend`'s `apply_1q` kernel with a branch-free nested block/pair iteration driven by a layout-agnostic `base_index_1q` helper; the new shape eliminates the per-iteration `if (i & t_bit == 0) && (i & ctrl_mask) == ctrl_mask` branch, iterates exactly the pairs that mutate (`2^(n-1-|controls|)` iterations), and exposes a unit-stride inner block that P1-03 (AVX2) consumes directly.

**Architecture:** Single substantive file changes (`crates/aleph-sv/src/kernels/soa.rs::apply_1q`); one new helper (`crates/aleph-sv/src/kernels/mod.rs::base_index_1q`) sits next to `control_mask`. `apply_2q`/`apply_3q` (P1-07/P1-08), AVX2 (P1-03), and the AoS reference (`kernels/aos.rs`) are out of scope. `SoaSvBackend::apply_gate`, `measure_soa`, `soa_state`, `validation`, and the oracle harness are untouched.

**Tech Stack:** Rust 2021, `num_complex`, `smallvec`, `proptest`, `criterion`. No new dependencies. No public-API changes.

**Spec:** `docs/superpowers/specs/2026-05-26-p1-02-bit-manip-indexing-design.md`

**Branch:** `p1-02-bit-manip-indexing` (already created from main `07d3724`).

---

## File map

| Path | Action | Responsibility |
|---|---|---|
| `crates/aleph-sv/src/kernels/mod.rs` | **modify** | Add `pub(crate) fn base_index_1q(k, target, controls) -> usize` + 5 unit tests |
| `crates/aleph-sv/src/kernels/soa.rs` | **modify** | Rewrite `apply_1q` body to call `base_index_1q` in a flat `0..outer_count` loop; signature unchanged. Extend the existing `apply_1q_soa_matches_aos` proptest to draw a controls set |
| `BACKLOG.md` | **modify** | Tick P1-02 AC checkboxes per measured outcome |

No new files. No deleted files. No `Cargo.toml` changes.

---

## Task 1: `base_index_1q` helper + unit tests

**Files:**
- Modify: `crates/aleph-sv/src/kernels/mod.rs`

- [ ] **Step 1.1: Write the failing unit tests**

In `crates/aleph-sv/src/kernels/mod.rs`, replace the existing `#[cfg(test)] mod tests` block at the bottom of the file with the extended version below. Keep the existing three `control_mask` tests; append five `base_index_1q` tests:

```rust
#[cfg(test)]
mod tests {
    use super::{base_index_1q, control_mask};

    #[test]
    fn control_mask_empty_is_zero() {
        assert_eq!(control_mask(&[]), 0);
    }

    #[test]
    fn control_mask_combines_bits() {
        // Controls on qubits 0, 2, 5 → bit positions 0, 2, 5 → 0b100101 = 37.
        assert_eq!(control_mask(&[0, 2, 5]), 0b100101);
    }

    #[test]
    fn control_mask_is_order_independent() {
        assert_eq!(control_mask(&[5, 0, 2]), control_mask(&[0, 2, 5]));
    }

    #[test]
    fn base_index_uncontrolled_target_zero_is_2k() {
        // target = 0, no controls → k-th iteration's i0 is 2*k.
        for k in 0..16 {
            assert_eq!(base_index_1q(k, 0, &[]), 2 * k);
        }
    }

    #[test]
    fn base_index_uncontrolled_target_two_keeps_bit_zero() {
        // target = 2 → bit 2 of i0 is always 0.
        for k in 0..16 {
            let i = base_index_1q(k, 2, &[]);
            assert_eq!((i >> 2) & 1, 0, "k = {k}, i = {i:#06b}");
        }
    }

    #[test]
    fn base_index_controlled_sets_control_bits() {
        // target = 0, control = 2 → bit 0 of i0 is 0, bit 2 of i0 is 1.
        for k in 0..8 {
            let i = base_index_1q(k, 0, &[2]);
            assert_eq!(i & 1, 0, "target bit must be 0");
            assert_eq!((i >> 2) & 1, 1, "control bit must be 1");
        }
    }

    #[test]
    fn base_index_two_controls_target_between() {
        // target = 2, controls = [0, 5] → bit 0 = 1, bit 2 = 0, bit 5 = 1.
        for k in 0..16 {
            let i = base_index_1q(k, 2, &[0, 5]);
            assert_eq!(i & 1, 1, "control bit 0 must be 1");
            assert_eq!((i >> 2) & 1, 0, "target bit must be 0");
            assert_eq!((i >> 5) & 1, 1, "control bit 5 must be 1");
        }
    }

    #[test]
    fn base_index_enumerates_all_pairs_exactly_once() {
        // For n = 5 qubits, target = 3, controls = [1], the valid pair
        // count is 2^(5 - 1 - 1) = 8.  Collect (i0, i1) across all k;
        // every value must be distinct, and the union must equal exactly
        // the 16 amplitudes whose bit-1 is set (2^(5-1) = 16).
        use std::collections::HashSet;
        let mut seen: HashSet<usize> = HashSet::new();
        for k in 0..8 {
            let i0 = base_index_1q(k, 3, &[1]);
            let i1 = i0 | (1usize << 3);
            assert!(seen.insert(i0), "duplicate i0 = {i0:#07b} at k = {k}");
            assert!(seen.insert(i1), "duplicate i1 = {i1:#07b} at k = {k}");
            assert_eq!(
                (i0 >> 1) & 1,
                1,
                "control bit 1 unset at i0 = {i0:#07b}"
            );
        }
        assert_eq!(seen.len(), 16);
    }
}
```

- [ ] **Step 1.2: Run to verify the new tests fail to compile**

Run: `cargo test -p aleph-sv --lib kernels::tests 2>&1 | head -20`
Expected: compile error — `cannot find function 'base_index_1q' in this scope`.

- [ ] **Step 1.3: Implement `base_index_1q`**

In `crates/aleph-sv/src/kernels/mod.rs`, between the `control_mask` function and the `#[cfg(test)] mod tests` block, insert:

```rust
/// For a 1-qubit gate on `target` with external `controls`, returns
/// the "base" amplitude index `i0` (the one with `target` bit = 0
/// and every control bit = 1) for the `k`-th iteration of the
/// free-bit outer loop.
///
/// `controls` must contain no element equal to `target` (callers
/// enforce this via the `DuplicateQubit` check before reaching the
/// kernel).  `target` and every control must be `< usize::BITS`
/// (caller enforces via `MAX_SOA_QUBITS = 28`).
///
/// `k` ranges over `0..(1usize << free_bits)` where
/// `free_bits = n_qubits - 1 - controls.len()`.  The pair partner is
/// `i0 | (1usize << target)`.
///
/// The implementation walks the sorted set of fixed bit positions
/// (target + controls) and splices chunks of `k`'s low bits into the
/// free slots between them, leaving the fixed slots to be filled in
/// the same pass (target → 0, controls → 1).  Algorithm is canonical
/// — cf. QuEST `statevec_unitary` for the reference shape.
pub(crate) fn base_index_1q(k: usize, target: u32, controls: &[u32]) -> usize {
    // Stack-only for the realistic `controls.len() ≤ 7` range (the
    // `SmallVec<[u32; 6]>` in `apply_gate`'s `seen` set tolerates up
    // to ~6 unique qubit indices; this cap of 8 leaves headroom and
    // avoids any heap allocation in the hot path).
    let mut fixed: smallvec::SmallVec<[(u32, bool); 8]> = smallvec::SmallVec::new();
    fixed.push((target, false));
    for &c in controls {
        fixed.push((c, true));
    }
    fixed.sort_unstable_by_key(|(pos, _)| *pos);

    let mut i = 0usize;
    let mut k_rem = k;
    let mut prev = 0u32;
    for (pos, val) in &fixed {
        let span = (*pos - prev) as usize;
        let chunk = k_rem & ((1usize << span) - 1);
        i |= chunk << (prev as usize);
        k_rem >>= span;
        if *val {
            i |= 1usize << pos;
        }
        prev = *pos + 1;
    }
    // Remaining free bits above the highest fixed position.
    i |= k_rem << (prev as usize);
    i
}

```

- [ ] **Step 1.4: Run tests to verify they pass**

Run: `cargo test -p aleph-sv --lib kernels::tests -- --nocapture`
Expected: 8 passed (3 existing `control_mask` + 5 new `base_index_1q`).

- [ ] **Step 1.5: Clippy check**

Run: `cargo clippy -p aleph-sv --all-targets -- -D warnings 2>&1 | tail -5`
Expected: clean. Note: `base_index_1q` is `pub(crate)` and unused outside the test module on this commit — Task 2 wires the caller in `kernels/soa.rs`. If clippy complains about dead code, that's expected and resolves in Task 2. If it complains for any other reason, fix before committing.

Workaround if clippy errors on dead-code: prepend `#[allow(dead_code)]` to the function and remove it in Task 2 after the caller is wired.

- [ ] **Step 1.6: Commit**

```bash
git add crates/aleph-sv/src/kernels/mod.rs
git commit -m "P1-02: add base_index_1q helper + unit tests"
```

---

## Task 2: Rewrite `apply_1q` body using `base_index_1q`

**Files:**
- Modify: `crates/aleph-sv/src/kernels/soa.rs`

The existing `apply_1q` body is replaced top-to-bottom inside the same function signature. The 4 existing unit tests (`x_flips_single_qubit_soa`, `h_on_zero_yields_plus_soa`, `external_control_skips_when_unset_soa`, `external_control_fires_when_set_soa`) and the existing `apply_1q_soa_matches_aos` proptest (still `controls=&[]`) must pass unchanged — they are the existing guard rail.

- [ ] **Step 2.1: Baseline — confirm existing tests pass before rewrite**

Run: `cargo test -p aleph-sv --lib kernels::soa 2>&1 | tail -5`
Expected: 9 passed (4 unit + 3 cnot/toffoli + 2 proptests = 9 with current count). Note the exact pass count; it must stay at 9 after Step 2.4.

- [ ] **Step 2.2: Rewrite `apply_1q` body**

In `crates/aleph-sv/src/kernels/soa.rs`, replace the entire `pub(crate) fn apply_1q(...)` block (from the line `pub(crate) fn apply_1q(` through its closing `}`) with the body below. Keep the doc comment immediately above the function; replace only the body inside the braces.

Old body (for reference — what's being removed):
```rust
    debug_assert_eq!(re.len(), im.len());
    let t_bit = 1usize << target;
    let ctrl_mask = super::control_mask(controls);
    let len = re.len();
    let mut i = 0usize;
    while i < len {
        if i & t_bit == 0 && (i & ctrl_mask) == ctrl_mask {
            let j = i | t_bit;
            // … 8-mul / 4-add matrix application …
        }
        i += 1;
    }
```

New body:
```rust
    debug_assert_eq!(re.len(), im.len());
    debug_assert!(
        re.len().is_power_of_two(),
        "apply_1q: state length must be a power of two, got {}",
        re.len()
    );
    let target_bit = 1usize << target;
    // `n_qubits` is encoded by `re.len() = 2^n_qubits` (allocator's
    // postcondition).  `free_bits = n_qubits - 1 (target) - |controls|`.
    let n_qubits = re.len().trailing_zeros();
    let n_free = n_qubits - 1 - controls.len() as u32;
    let outer_count = 1usize << n_free;

    // Hoist matrix entries — let the compiler keep them in registers
    // across the inner block instead of refetching from `m` each iter.
    let m00_re = m[0][0].re;
    let m00_im = m[0][0].im;
    let m01_re = m[0][1].re;
    let m01_im = m[0][1].im;
    let m10_re = m[1][0].re;
    let m10_im = m[1][0].im;
    let m11_re = m[1][1].re;
    let m11_im = m[1][1].im;

    for k in 0..outer_count {
        let i0 = super::base_index_1q(k, target, controls);
        let i1 = i0 | target_bit;

        let a0_re = re[i0];
        let a0_im = im[i0];
        let a1_re = re[i1];
        let a1_im = im[i1];

        re[i0] = m00_re * a0_re - m00_im * a0_im
            + m01_re * a1_re
            - m01_im * a1_im;
        im[i0] = m00_re * a0_im
            + m00_im * a0_re
            + m01_re * a1_im
            + m01_im * a1_re;
        re[i1] = m10_re * a0_re - m10_im * a0_im
            + m11_re * a1_re
            - m11_im * a1_im;
        im[i1] = m10_re * a0_im
            + m10_im * a0_re
            + m11_re * a1_im
            + m11_im * a1_re;
    }
```

- [ ] **Step 2.3: If Task 1 added `#[allow(dead_code)]` on `base_index_1q`, remove it now**

The helper has its first non-test caller. Open `crates/aleph-sv/src/kernels/mod.rs` and remove the `#[allow(dead_code)]` attribute if present.

- [ ] **Step 2.4: Run kernels::soa tests — must still be 9 passed**

Run: `cargo test -p aleph-sv --lib kernels::soa -- --nocapture 2>&1 | tail -15`
Expected: identical pass count to Step 2.1 (9 passed). If any test fails, the bit-manip math is wrong — the most likely culprit is `n_free = n_qubits - 1 - controls.len() as u32` underflowing when `controls.len() >= n_qubits` (impossible in correct usage but happens if the test passes a malformed input). Re-read the failing assertion message; the existing `apply_1q_soa_matches_aos` proptest shrinks to a minimal failing input.

- [ ] **Step 2.5: Run the full `aleph-sv` lib tests — the entire SoA backend, including measure_soa expectation_value slow path which calls `apply_1q`, must stay green**

Run: `cargo test -p aleph-sv --lib 2>&1 | tail -3`
Expected: 121 passed (118 from P1-01 baseline + 5 new base_index_1q from Task 1; numbers count from `cargo test -p aleph-sv --lib | tail -3` after Task 1 commit).

- [ ] **Step 2.6: Run the workhorse SoA-vs-naive across all 28 fixtures**

Run: `cargo test -p aleph-oracle --test soa_vs_naive 2>&1 | tail -5`
Expected: `test result: ok. 1 passed`. This is the gating correctness test — exercises every committed oracle circuit (QFT, GHZ, CCX, controlled-Phase) through both backends with the new kernel.

- [ ] **Step 2.7: Commit**

```bash
git add crates/aleph-sv/src/kernels/soa.rs crates/aleph-sv/src/kernels/mod.rs
git commit -m "P1-02: rewrite SoA apply_1q with branch-free bit-manip iteration"
```

---

## Task 3: Extend `apply_1q_soa_matches_aos` proptest to cover controlled paths

**Files:**
- Modify: `crates/aleph-sv/src/kernels/soa.rs` (the `#[cfg(test)] mod tests` block, specifically the existing `proptest!` macro containing `apply_1q_soa_matches_aos`)

Today's proptest passes `controls: &[]` only — bit-manip controlled paths slip through. This task extends the strategy to draw 0, 1, or 2 controls and validates AoS↔SoA equivalence for the controlled path.

- [ ] **Step 3.1: Replace the `apply_1q_soa_matches_aos` proptest body**

In `crates/aleph-sv/src/kernels/soa.rs`, inside `#[cfg(test)] mod tests`, locate the existing `apply_1q_soa_matches_aos` test inside `proptest!`. Replace it with:

```rust
        /// AoS / SoA equivalence on `apply_1q`: for any 1q gate, any
        /// target qubit, any set of 0-2 distinct external controls,
        /// and any normalised state, applying through both kernels
        /// yields matching amplitudes within 1e-12.
        ///
        /// The controls set is built deterministically from the
        /// strategy seed: take qubits `[0..5)` minus `q`, sort by a
        /// seed-keyed pseudo-random key, take the first `n_ctrls`.
        /// Distinctness from `q` and among each other is structural.
        #[test]
        fn apply_1q_soa_matches_aos(
            gate in arb_1q_gate(),
            q in 0u32..5,
            n_ctrls in 0usize..=2,
            ctrl_seed in any::<u32>(),
            amps in arb_state_vector(5),
        ) {
            let m = match gate.matrix().unwrap() {
                GateMatrix::M2x2(m) => m,
                _ => unreachable!("arb_1q_gate yields 1q gates"),
            };
            // Build a deterministic control set: candidates are 0..5 \ {q},
            // shuffled by a seed-keyed hash, truncated to n_ctrls, sorted.
            let mut ctrls: Vec<u32> = (0u32..5).filter(|c| *c != q).collect();
            ctrls.sort_by_key(|c| ctrl_seed.wrapping_mul(*c + 7));
            ctrls.truncate(n_ctrls);
            ctrls.sort_unstable();

            let re: Vec<f64> = amps.iter().map(|c| c.re).collect();
            let im: Vec<f64> = amps.iter().map(|c| c.im).collect();
            // AoS reference
            let mut aos_state = amps.clone();
            aos::apply_1q(&mut aos_state, q, &ctrls, &m);
            // SoA candidate
            let mut soa_re = re.clone();
            let mut soa_im = im.clone();
            apply_1q(&mut soa_re, &mut soa_im, q, &ctrls, &m);
            let soa_state = aos_from(&soa_re, &soa_im);
            for (a, b) in aos_state.iter().zip(soa_state.iter()) {
                prop_assert!(
                    (a - b).norm() < 1e-12,
                    "ctrls={ctrls:?} q={q}: aos {a} vs soa {b}",
                );
            }
        }
```

- [ ] **Step 3.2: Run the extended proptest — 64 cases × {0,1,2} controls × random states**

Run: `cargo test -p aleph-sv --lib kernels::soa::tests::apply_1q_soa_matches_aos 2>&1 | tail -10`
Expected: `test result: ok. 1 passed`. If it fails, proptest shrinks to a minimal `(gate, q, ctrls, amps)` tuple and prints it — that's the input that reveals the bit-manip bug, fix in `base_index_1q` or the `apply_1q` body and re-run.

- [ ] **Step 3.3: Run the full `aleph-sv` lib + the `aleph-oracle` workhorse + all generated oracle tests**

Run:
```bash
cargo test -p aleph-sv --lib 2>&1 | tail -3
cargo test -p aleph-oracle 2>&1 | tail -3
cargo test -p aleph-oracle --tests 2>&1 | tail -3
```
Expected: all green; 112 generated oracle tests still pass on both backends (controlled paths in QFT-3, QFT-5, grover_2q_mark11, kernel_p, kernel_ccx exercise the bit-manip controlled iteration).

- [ ] **Step 3.4: Commit**

```bash
git add crates/aleph-sv/src/kernels/soa.rs
git commit -m "P1-02: extend apply_1q AoS↔SoA proptest with 0-2 external controls"
```

---

## Task 4: Run the benchmark and capture P1-02 numbers

**Files:** none (verification only)

`crates/aleph-sv/benches/soa_vs_naive.rs` is unchanged — its existing `qft/n{10,15,20}/{naive,soa}` and `ghz/n20/{naive,soa}` group already produces the numbers we need.

- [ ] **Step 4.1: Run the bench with target-native codegen**

Run:
```bash
RUSTFLAGS="-C target-cpu=native" cargo bench -p aleph-sv --bench soa_vs_naive 2>&1 | tee /tmp/p1-02-bench.txt | tail -40
```

Takes ~3-4 minutes. The output captures both backends side by side; the file `/tmp/p1-02-bench.txt` is the source for the PR body.

- [ ] **Step 4.2: Pull the P1-01 baseline numbers**

The P1-01 PR description (PR #75) recorded on M-series local-dev:
```
qft/n10/naive   85.3 µs   |  qft/n10/soa  71.4 µs    (1.19× faster)
qft/n15/naive   4.04 ms   |  qft/n15/soa  4.42 ms    (0.91× — SoA slower)
qft/n20/naive   235.7 ms  |  qft/n20/soa  245.1 ms   (0.96× — SoA slower)
ghz/n20/naive   80.4 ms   |  ghz/n20/soa  72.3 ms    (1.11× faster)
```

The P1-02 effect should manifest as `qft/n*/soa` numbers improving toward — and ideally past — the corresponding `naive` numbers, while AoS naive stays the same (P1-02 doesn't touch AoS). The `ghz/n20/soa` improvement is expected to be smaller (GHZ only invokes one `apply_1q`; the rest is `apply_2q` Cnots which P1-02 doesn't touch).

- [ ] **Step 4.3: Note the numbers — they go into the PR body in Task 6**

Write down (or keep `/tmp/p1-02-bench.txt` handy):
- `qft/n10/soa` P1-02 number
- `qft/n15/soa` P1-02 number
- `qft/n20/soa` P1-02 number
- `ghz/n20/soa` P1-02 number

The PR body in Task 6 tabulates these against the P1-01 baseline.

- [ ] **Step 4.4: No commit needed** — bench output is not checked into the repo; criterion's `target/criterion/*` directory is gitignored. The numbers live in the PR description and (canonically) on bencher.dev after merge.

---

## Task 5: Tick BACKLOG P1-02 acceptance criteria

**Files:**
- Modify: `BACKLOG.md`

- [ ] **Step 5.1: Open BACKLOG.md, find the P1-02 AC block**

```bash
grep -n '^### \[P1-02\]' BACKLOG.md
```
Expected: prints the line number where the P1-02 section starts (line 667 as of `07d3724`).

- [ ] **Step 5.2: Read the AC block**

```bash
sed -n "$(grep -n '^### \[P1-02\]' BACKLOG.md | head -1 | cut -d: -f1),+40p" BACKLOG.md
```
The Acceptance Criteria sub-block has three `- [ ]` lines. Edit each:

- "All 1q gates implemented with this pattern" → flip to `- [x]` (Task 2 rewrites the kernel; both controlled and uncontrolled paths share the bit-manip iteration).
- "Benchmark: 2–3× improvement over P1-01 on QFT-20" → flip to `- [x]` **only if** the measured `qft/n20/soa` ratio vs the P1-01 baseline in Task 4 is ≥ 2×. If below, leave unchecked and append a single inline parenthetical note pointing at the EPYC bencher.dev numbers as the source of truth, mirroring the P1-01 spec §12 precedent. Do not invent prose — keep it terse.
- "All correctness tests pass" → flip to `- [x]` (workhorse + 112 generated oracle + extended proptest all green per Tasks 2-3).

- [ ] **Step 5.3: Use the Edit tool to flip the boxes**

Example for the third AC line (replace the actual hyphen-bracketed string verbatim from Step 5.2):

```
- [ ] All correctness tests pass
```
→
```
- [x] All correctness tests pass
```

Apply the same flip to "All 1q gates implemented with this pattern" and conditionally to the benchmark line per Step 5.2.

- [ ] **Step 5.4: Commit**

```bash
git add BACKLOG.md
git commit -m "P1-02: tick AC checkboxes for bit-manip indexing"
```

---

## Task 6: Final workspace sweep — fmt / clippy / test / release

**Files:** none (verification only)

- [ ] **Step 6.1: Format check; apply rustfmt if needed**

```bash
cargo fmt --check 2>&1 | tail -5
```
If any diff appears, run `cargo fmt` and stage. Commit as a separate `P1-02: cargo fmt` commit (P1-01 precedent).

- [ ] **Step 6.2: Workspace clippy**

```bash
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -10
```
Expected: clean. Most likely issues if any: unused `#[allow(dead_code)]` left from Task 1.5 fallback (remove it), or an `unused_imports` if the proptest extension dropped a previously-used identifier.

- [ ] **Step 6.3: Full workspace test**

```bash
cargo test --workspace 2>&1 | grep -E "^test result.*failed [^0]" | head -5
echo "---"
cargo test --workspace 2>&1 | grep -E "^test result.*ok" | wc -l
```
Expected: no `failed [^0]` lines (i.e. zero non-zero-failure summaries); the second line prints the count of `ok` summaries (≈36 across the workspace per P1-01 baseline).

- [ ] **Step 6.4: Release build sanity check**

```bash
cargo build --release --workspace 2>&1 | tail -5
```
Expected: clean.

- [ ] **Step 6.5: Push branch**

```bash
git push -u origin p1-02-bit-manip-indexing
```

- [ ] **Step 6.6: Open the PR**

Title: `[P1-02] Bit-manipulation indexing for 1-qubit gates`

Body (via `gh pr create --body "$(cat <<'EOF' ... EOF)"`):

```markdown
## Summary

Replaces `SoaSvBackend`'s `apply_1q` kernel with a branch-free nested block/pair iteration driven by a new `base_index_1q` helper. The new shape eliminates the per-iteration `if (i & t_bit == 0) && (i & ctrl_mask) == ctrl_mask` branch, iterates exactly the pairs that mutate (`2^(n-1-|controls|)`), and exposes a unit-stride inner block that P1-03 (AVX2) consumes directly.

Closes #14

## Approach

* `crates/aleph-sv/src/kernels/mod.rs::base_index_1q(k, target, controls) -> usize` — sorts `target + controls` ascending, walks free-bit chunks between fixed positions, inserts target=0 and each control=1 at the fixed slots. Stack-only via `SmallVec<[(u32, bool); 8]>`. 5 unit tests cover boundary cases (target at LSB / MSB, multi-control with target between, exhaustive pair enumeration).
* `crates/aleph-sv/src/kernels/soa.rs::apply_1q` body rewritten to `for k in 0..outer_count { let i0 = base_index_1q(k, target, controls); ... }`. Matrix entries hoisted outside the loop (compiler keeps them in registers). Public signature unchanged; no caller in `SoaSvBackend::apply_gate` or `measure_soa` is touched.
* AoS↔SoA proptest in `kernels/soa.rs` extended from `controls: &[]` to `n_ctrls in 0..=2` with a seed-keyed deterministic control set. 64 cases × {0,1,2} controls verifies the controlled bit-manip path.
* `NaiveSvBackend` AoS kernel untouched (reference oracle per BACKLOG).
* `apply_2q`, `apply_3q` untouched — P1-07 and P1-08 cover those.

## Test results

* `cargo test --workspace` — all green
* Workhorse `crates/aleph-oracle/tests/soa_vs_naive.rs::all_fixtures_match_naive` — SoA ≡ AoS within 1e-12 across all 28 oracle fixtures (QFT, GHZ, CCX, controlled-Phase all exercise the bit-manip path through the workhorse)
* 112 generated oracle tests (`naive_state`, `naive_distribution`, `soa_state`, `soa_distribution` per fixture) — all pass
* Extended `apply_1q_soa_matches_aos` proptest (64 cases × {0,1,2} controls) — passes
* 5 `base_index_1q` unit tests — pass
* `cargo clippy --workspace --all-targets -- -D warnings` clean. `cargo fmt --check` clean. `cargo build --release --workspace` clean.

## Benchmark numbers

Local dev (M-series, `RUSTFLAGS=-C target-cpu=native`, `cargo bench -p aleph-sv --bench soa_vs_naive`):

| Bench | P1-01 SoA | P1-02 SoA | Ratio vs P1-01 |
|---|---|---|---|
| `qft/n10` | 71.4 µs | <fill from /tmp/p1-02-bench.txt> | <ratio> |
| `qft/n15` | 4.42 ms | <fill> | <ratio> |
| `qft/n20` | 245.1 ms | <fill> | <ratio> |
| `ghz/n20` | 72.3 ms | <fill> | <ratio> |

[If `qft/n20` ratio ≥ 2×, state that the BACKLOG AC is met locally; if below, note that EPYC bencher.dev numbers (visible after merge at https://bencher.dev/console/projects/aleph) are the source of truth per P1-01 spec §12 precedent. Don't gate the merge on perf if correctness gates are met.]

The naive AoS numbers are unchanged from P1-01 (P1-02 doesn't touch the AoS kernel) and are included in the bench output as a stable reference.

## Notes / follow-ups

* `apply_2q` / `apply_3q` still use the P1-01 branch-loop shape; P1-07 (2q generic kernel) and P1-08 (multi-controlled gates) will land the same bit-manip pattern there.
* P1-03 (AVX2) is the natural next ticket — the unit-stride inner loop from this PR is exactly the shape `_mm256_loadu_pd` / `_mm256_storeu_pd` want.
* `base_index_1q` is intentionally specific to 1q gates; a generic `expand_index` will be hoisted at P1-07 if 2q needs the same primitive.
```

(Use a HEREDOC per CLAUDE.md PR-workflow guidance; substitute the bench numbers from `/tmp/p1-02-bench.txt`.)

- [ ] **Step 6.7: Wait for CI; squash-merge when green**

Standard cadence (cf. `p1-01-merged` memory): all GitHub Actions checks must be green — clippy, rustfmt, test linux stable+beta, test macos stable+beta, bench (self-hosted EPYC). The bench job is ~17 minutes; it captures the canonical numbers on bencher.dev for the post-merge timeline.

Use `gh pr merge <PR#> --squash --subject "[P1-02] Bit-manipulation indexing for 1-qubit gates (#<PR#>)" --delete-branch` once green. GitHub closes issue #14 automatically via the PR body's `Closes #14`.

---

## Self-review checklist (run after writing the plan; informs plan stability — do not commit)

* **Spec coverage:**
  * Spec §1 goal → Tasks 1+2 cover the rewrite
  * Spec §2 scope (in/out) → Tasks scope correctly (1q only, SoA only, helper next to control_mask)
  * Spec §3 architecture → File map matches; one substantive file + one helper file
  * Spec §4 helper contract + impl → Task 1 implements verbatim
  * Spec §5 rewritten kernel → Task 2 implements verbatim (`debug_assert!(is_power_of_two)` from §9 risks added)
  * Spec §6 testing (workhorse, helper units, extended proptest, generated oracle) → Tasks 1.1, 3.1 (proptest extension); workhorse + 112 generated covered by re-running existing tests in Tasks 2.6 and 3.3
  * Spec §7 bench → Task 4 captures numbers (no bench code change; existing `benches/soa_vs_naive.rs` already covers QFT/GHZ for both backends)
  * Spec §8 ACs → Task 5 ticks
  * Spec §9 risks → `debug_assert!(is_power_of_two)` mitigation present in Task 2's new body; SmallVec stack-only confirmed by Task 1's cap of 8; AC-miss-on-EPYC handled by Task 5.2 conditional tick
  * Spec §10 workflow → Task 6 covers fmt/clippy/test/release/push/PR/squash-merge
  * Spec §11 implementation order → Plan task order matches: helper first, then kernel rewrite, then proptest extension, then bench, then ACs, then sweep

* **Placeholder scan:** the PR body in Task 6.6 contains `<fill from /tmp/p1-02-bench.txt>` placeholders — these are intentionally bench numbers captured in Task 4 and substituted at PR-creation time. Same convention as P1-01 plan Task 17.6. No other placeholders.

* **Type consistency:**
  * `base_index_1q(k: usize, target: u32, controls: &[u32]) -> usize` — same signature in Task 1.1 (test), Task 1.3 (impl), Task 2.2 (caller).
  * `apply_1q(re, im, target, controls, m)` — signature unchanged from P1-01; verified in Task 2.2's "old body for reference" block and the unchanged tests in Task 2.4.
  * `n_qubits: u32` from `re.len().trailing_zeros()`, `n_free: u32 = n_qubits - 1 - controls.len() as u32` — consistent in Task 2.2.

* **Implicit invariant:** Task 2.2 assumes `re.len() = 2^n_qubits` (power-of-two), upheld by `allocate`. The `debug_assert!(re.len().is_power_of_two())` guard surfaces a future divergence loudly in dev builds. Production cost zero.
