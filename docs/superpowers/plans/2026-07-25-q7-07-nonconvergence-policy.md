# Q7-07 Non-Convergence Fallback Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Quantify the relay-BP non-convergence rate at the shipped operating points, measure how much of the logical-error budget it accounts for, evaluate OSD-lite fallbacks against that ceiling, and record a chosen policy.

**Architecture:** Direct A/B on campaign LER is statistically impossible here (LER CI ±1.13e-4 at 10⁶ shots vs a ~0.1 % non-convergence rate), so measurement is **conditional**: (1) rate per operating point, (2) the attributable fraction `A(p)` — the hard ceiling on any fallback since it only ever acts on `valid_flag=0` shots, (3) conditional rescue on a dense retained corpus, propagated back to overall LER analytically. A pre-registered decision rule fixes the verdict before the data is seen.

**Tech Stack:** Rust 2021 (crate `aleph-qec`), rayon for parallel decode, criterion not needed (this is a statistics ticket, not a perf one), Python 3 + numpy + pynq for the KV260 board driver, Verilator for the fallback co-sim gate.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-25-q7-07-nonconvergence-policy-design.md`. Read it before Task 1.
- Branch `q7-07-nonconvergence-policy` (already created from `origin/main`, spec committed as `1982787`). No git worktrees — work in `/Users/ex/GitHub/aleph`.
- Shipped decoder operating point, identical in every task: `LEGS=6`, `ITERS=10`, `GAMMA=(-0.3, 0.9)`, `SEED=0x5E1A_4B9C`, `MSG_BITS=8`, `FRAC_BITS=3` (Q5.3).
- Block-path operating points: `p ∈ {0.003, 0.005, 0.007}`, `rounds=1`. Window-path: `p ∈ {0.001, 0.003, 0.005}`, `rounds=12`, `W=6`, `C=2`.
- No `unwrap()`/`expect()` in library code (`crates/aleph-qec/src/**`). Examples and tests may use them.
- `cargo clippy --workspace --all-targets -- -D warnings` and `cargo fmt --check` must pass before every commit.
- **No RTL change and no new bitstream in this ticket** unless Task 7's data triggers the escalation clause, which returns to the user first.
- Comments explain *why*, not *what*. Cite sources (Fossorier–Lin for OSD) adjacent to the code.

---

## File Structure

| File | Responsibility |
|---|---|
| `crates/aleph-qec/src/refvec.rs` (create) | `.ref` campaign companion-file codec — record type, v2 writer/reader, legacy rejection. Pure I/O, no decoder knowledge. |
| `crates/aleph-qec/src/lib.rs` (modify) | Register + re-export `refvec`. |
| `crates/aleph-qec/src/osd.rs` (modify) | Add residual-restricted combination sweep to `OsdDecoder`. |
| `crates/aleph-qec/src/fixed_bp.rs` (modify) | Explicit `!converged` gate in `FixedRelayBpOsd::decode_fixed_osd`. |
| `crates/aleph-qec/examples/qec_q7_bp_graph.rs` (modify) | `emit_sil_vectors` writes `.ref` v2 through `refvec`. |
| `crates/aleph-qec/examples/qec_q7_nonconv.rs` (create) | The campaign: `block` / `window` / `candidates` subcommands. Levels 1–3. |
| `hw/sw/bp_stream_banked_ler_kv260.py` (modify) | Capture status bit 19 + latency; read `.ref` v2; report `rtl_valid` / `valid_mismatch`. |
| `docs/qec/q7-07-nonconvergence-policy.md` (create) | The deliverable report. |
| `docs/perf/data/qec-q7-nonconv.csv` (create) | Raw campaign output. |

---

### Task 1: `.ref` v2 codec carrying the software validity flag

The campaign `.ref` file currently holds two u16 per shot (`true_obs`, `sw_obs`) with no header, so the board never learns what the software golden thought about validity. Add a third u16 and a magic header. The codec lives in the library (not in the example) so it is unit-testable.

**Files:**
- Create: `crates/aleph-qec/src/refvec.rs`
- Modify: `crates/aleph-qec/src/lib.rs`
- Modify: `crates/aleph-qec/examples/qec_q7_bp_graph.rs:2153-2232` (`emit_sil_vectors`)

**Interfaces:**
- Consumes: `FixedRelayBp::decode_fixed_ehat(&Syndrome) -> (Vec<u8>, Vec<bool>, bool)` and `FixedRelayBp::iters_to_valid(&Syndrome) -> (bool, u32)`, both already public in `crates/aleph-qec/src/fixed_bp.rs`.
- Produces: `aleph_qec::refvec::{RefRecord, write_ref, read_ref, REF_MAGIC, REF_VERSION}`. `RefRecord { true_obs: u16, sw_obs: u16, valid: bool, iters: u16 }`. Task 6's Python driver mirrors this exact byte layout.

**Format (all little-endian u16):** header `[REF_MAGIC, REF_VERSION, REF_WORDS_PER_SHOT, 0]`, then per shot `[true_obs, sw_obs, meta]` where `meta = (valid << 15) | (iters & 0x7FFF)`.

`REF_MAGIC = 0xA1E7` cannot collide with a legacy file's first word: legacy byte 0–1 is `true_obs`, and the gross code has 12 observables, so `true_obs <= 0x0FFF < 0xA1E7`. A legacy file is therefore detected and rejected rather than silently misparsed — the same footgun class as #478.

- [ ] **Step 1: Write the failing tests**

Create `crates/aleph-qec/src/refvec.rs` with only the tests plus the doc header (no implementation yet):

```rust
//! Binary `.ref` companion-file codec for the on-silicon LER campaigns (Q7-06 AC-2, Q7-07).
//!
//! A campaign ships two blobs: `<prefix>.syn` (raw syndrome words the DMA streams into the PL) and
//! `<prefix>.ref` (what the software golden decided, for the host to compare against). v1 held two
//! u16 per shot — `true_obs`, `sw_obs` — and nothing about *validity*, so the board could not check
//! its own `valid_flag` against the golden's. v2 adds a third u16 and a magic header.
//!
//! The header exists because #478 cost a full campaign re-run: a golden silently paired with a
//! bitstream built at another `p`. A file that cannot be identified must not be guessed at.

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<RefRecord> {
        vec![
            RefRecord { true_obs: 0x000, sw_obs: 0x000, valid: true, iters: 3 },
            RefRecord { true_obs: 0x001, sw_obs: 0x801, valid: false, iters: 60 },
            RefRecord { true_obs: 0xFFF, sw_obs: 0xFFF, valid: true, iters: 1 },
        ]
    }

    #[test]
    fn test_round_trip_preserves_every_field() {
        let recs = sample();
        let mut buf = Vec::new();
        write_ref(&mut buf, &recs).expect("write");
        let back = read_ref(&mut buf.as_slice()).expect("read");
        assert_eq!(back, recs);
    }

    #[test]
    fn test_header_is_four_words_and_payload_is_three_per_shot() {
        let mut buf = Vec::new();
        write_ref(&mut buf, &sample()).expect("write");
        assert_eq!(buf.len(), 2 * (4 + 3 * 3));
        assert_eq!(u16::from_le_bytes([buf[0], buf[1]]), REF_MAGIC);
        assert_eq!(u16::from_le_bytes([buf[2], buf[3]]), REF_VERSION);
    }

    #[test]
    fn test_legacy_v1_file_is_rejected_not_misparsed() {
        // v1 layout: two u16 per shot, no header. First word is a 12-bit observable mask.
        let legacy: Vec<u8> = [0x0001u16, 0x0001, 0x0002, 0x0002]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let err = read_ref(&mut legacy.as_slice()).expect_err("legacy must be rejected");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_unknown_version_is_rejected() {
        let mut buf = Vec::new();
        write_ref(&mut buf, &sample()).expect("write");
        buf[2..4].copy_from_slice(&99u16.to_le_bytes());
        let err = read_ref(&mut buf.as_slice()).expect_err("bad version must be rejected");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn test_truncated_payload_is_rejected() {
        let mut buf = Vec::new();
        write_ref(&mut buf, &sample()).expect("write");
        buf.truncate(buf.len() - 3);
        let err = read_ref(&mut buf.as_slice()).expect_err("truncated must be rejected");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }
}
```

Register the module in `crates/aleph-qec/src/lib.rs` — add `pub mod refvec;` next to the other `pub mod` lines, and add `pub use refvec::{read_ref, write_ref, RefRecord};` next to the other re-exports.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p aleph-qec --lib refvec 2>&1 | tail -20`
Expected: FAIL — compile errors, `cannot find type RefRecord`, `cannot find function write_ref`.

- [ ] **Step 3: Write the implementation**

Insert above the `#[cfg(test)] mod tests` block in `crates/aleph-qec/src/refvec.rs`:

```rust
use std::io::{Read, Write};

/// First header word. Chosen above `0x0FFF` so it can never be a legacy v1 file's leading
/// `true_obs` (the gross code has 12 observables), making v1 detectable rather than misparsed.
pub const REF_MAGIC: u16 = 0xA1E7;
/// Current format version.
pub const REF_VERSION: u16 = 2;
/// u16 words per shot in the payload: `true_obs`, `sw_obs`, `meta`.
pub const REF_WORDS_PER_SHOT: u16 = 3;

const HEADER_WORDS: usize = 4;

/// One shot's software-golden record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RefRecord {
    /// Truth observable-flip mask sampled with the shot.
    pub true_obs: u16,
    /// The software golden's predicted observable-flip mask.
    pub sw_obs: u16,
    /// Whether some relay-BP leg found a syndrome-valid `ê` — the software twin of the RTL
    /// `valid_flag` (`hw/bp_relay_banked.sv:968`).
    pub valid: bool,
    /// 1-based global iteration index where a first-valid stop would land, or the full schedule
    /// length if none converged (`FixedRelayBp::iters_to_valid`).
    pub iters: u16,
}

fn invalid(msg: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg.to_string())
}

/// Write the v2 header followed by one 3-word record per shot.
pub fn write_ref<W: Write>(w: &mut W, recs: &[RefRecord]) -> std::io::Result<()> {
    for word in [REF_MAGIC, REF_VERSION, REF_WORDS_PER_SHOT, 0] {
        w.write_all(&word.to_le_bytes())?;
    }
    for r in recs {
        let meta = (u16::from(r.valid) << 15) | (r.iters & 0x7FFF);
        for word in [r.true_obs, r.sw_obs, meta] {
            w.write_all(&word.to_le_bytes())?;
        }
    }
    Ok(())
}

/// Read a v2 file. Rejects a missing/unknown magic (i.e. a legacy v1 file), an unknown version,
/// an unexpected record width, and a truncated payload — never guesses.
pub fn read_ref<R: Read>(r: &mut R) -> std::io::Result<Vec<RefRecord>> {
    let mut bytes = Vec::new();
    r.read_to_end(&mut bytes)?;
    if bytes.len() % 2 != 0 {
        return Err(invalid("ref: odd byte length, not a u16 stream"));
    }
    let words: Vec<u16> = bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    if words.len() < HEADER_WORDS {
        return Err(invalid("ref: shorter than the header"));
    }
    if words[0] != REF_MAGIC {
        return Err(invalid(
            "ref: bad magic — this is a legacy v1 file (no header); regenerate it with silvectors",
        ));
    }
    if words[1] != REF_VERSION {
        return Err(invalid("ref: unsupported version"));
    }
    if words[2] != REF_WORDS_PER_SHOT {
        return Err(invalid("ref: unexpected words-per-shot"));
    }
    let payload = &words[HEADER_WORDS..];
    if payload.len() % REF_WORDS_PER_SHOT as usize != 0 {
        return Err(invalid("ref: truncated payload"));
    }
    Ok(payload
        .chunks_exact(REF_WORDS_PER_SHOT as usize)
        .map(|c| RefRecord {
            true_obs: c[0],
            sw_obs: c[1],
            valid: c[2] >> 15 == 1,
            iters: c[2] & 0x7FFF,
        })
        .collect())
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p aleph-qec --lib refvec 2>&1 | tail -20`
Expected: PASS, 5 tests.

- [ ] **Step 5: Wire the emitter to v2**

In `crates/aleph-qec/examples/qec_q7_bp_graph.rs`, inside `emit_sil_vectors`: replace the `decode_batch` call and the per-shot `.ref` writes.

Replace this line:

```rust
    let predictions = fx.decode_batch(&syndromes).expect("decode_batch");
```

with:

```rust
    // Per-shot: observable flips + validity (one decode) and the first-valid iteration index (a
    // second pass — `iters_to_valid` reports where an early-exit stop *would* land regardless of
    // the mode flag, so it cannot be folded into the first). Emitting is a one-off; the campaign
    // is the expensive half.
    use rayon::prelude::*;
    let records: Vec<(Vec<bool>, bool, u32)> = syndromes
        .par_iter()
        .map(|s| {
            let (_ehat, flips, valid) = fx.decode_fixed_ehat(s);
            let (_conv, iters) = fx.iters_to_valid(s);
            (flips, valid, iters)
        })
        .collect();
```

Replace the `sw_obs` computation and the two `ref_f.write_all` lines inside the shot loop. The loop currently ends with:

```rust
        let mut sw_obs = 0u16;
        for (o, &b) in predictions[i].observable_flips.iter().enumerate() {
            if b {
                sw_obs |= 1u16 << o;
            }
        }
        if sw_obs != true_obs {
            sw_errors += 1;
        }
        ref_f.write_all(&true_obs.to_le_bytes()).expect("write ref");
        ref_f.write_all(&sw_obs.to_le_bytes()).expect("write ref");
```

Replace with:

```rust
        let mut sw_obs = 0u16;
        for (o, &b) in records[i].0.iter().enumerate() {
            if b {
                sw_obs |= 1u16 << o;
            }
        }
        if sw_obs != true_obs {
            sw_errors += 1;
        }
        if !records[i].1 {
            sw_nonconv += 1;
        }
        ref_recs.push(aleph_qec::RefRecord {
            true_obs,
            sw_obs,
            valid: records[i].1,
            iters: records[i].2.min(0x7FFF) as u16,
        });
```

Declare `let mut ref_recs: Vec<aleph_qec::RefRecord> = Vec::with_capacity(n);` and `let mut sw_nonconv: u64 = 0;` next to `let mut sw_errors: u64 = 0;`. After the loop, before `syn_f.flush()`, add:

```rust
    aleph_qec::write_ref(&mut ref_f, &ref_recs).expect("write ref");
```

Remove the now-unused `ref_f` per-shot writes only — keep the `File::create` and `BufWriter`. Add the non-convergence rate to the stdout summary line by appending ` sw_nonconv={sw_nonconv}` to the existing `println!`. Update the `eprintln!` byte-count line to `4 + n * 6` for the `.ref` size.

- [ ] **Step 6: Verify the emitter builds and produces a readable v2 file**

Run:
```bash
cargo run --release -p aleph-qec --example qec_q7_bp_graph -- silvectors 1 0.003 200 2024 /tmp/q707smoke 0.003
python3 -c "
import numpy as np
w = np.fromfile('/tmp/q707smoke.ref', dtype='<u2')
assert w[0] == 0xA1E7 and w[1] == 2 and w[2] == 3, w[:4]
assert w.size == 4 + 3*200, w.size
print('v2 ok; nonconverged =', int(np.count_nonzero((w[4::3] >> 15) == 0)))
"
```
Expected: `v2 ok; nonconverged = <small integer>` and no assertion error.

- [ ] **Step 7: Lint, format, commit**

```bash
cargo fmt
cargo clippy -p aleph-qec --all-targets -- -D warnings
cargo test -p aleph-qec --lib refvec
git add crates/aleph-qec/src/refvec.rs crates/aleph-qec/src/lib.rs crates/aleph-qec/examples/qec_q7_bp_graph.rs
git commit -m "[Q7-07] .ref v2: carry the software validity flag and first-valid iteration

v1 held only (true_obs, sw_obs), so the board could never check its own
valid_flag against the golden's. v2 adds a magic header (rejecting a v1 file
rather than misparsing it — the #478 lesson) and a third u16 packing valid
and the first-valid iteration index."
```

---

### Task 2: Residual-restricted OSD sweep + explicit tail gating

Two changes to the fallback machinery. The **residual-restricted sweep** is the only genuinely new candidate: today `osd_solve` takes the `w` least-reliable non-pivot columns from the *whole* 144-variable basis, so at sub-threshold `p` the sweep mostly explores columns that cannot affect the violated checks. Restricting the pool to variables touching an **unsatisfied** check buys a far higher effective order for the same `2^w` budget. The **gating fix** makes `FixedRelayBpOsd`'s tail-cost measurement honest instead of relying on `OsdDecoder` internally short-circuiting.

**Files:**
- Modify: `crates/aleph-qec/src/osd.rs` (struct fields ~`:40`, `with_params` ~`:60`, `osd_solve` sweep ~`:188`, tests `:279`)
- Modify: `crates/aleph-qec/src/fixed_bp.rs:541-545` (`decode_fixed_osd`)

**Interfaces:**
- Consumes: `OsdDecoder::with_order(usize) -> Self` and `OsdDecoder::correction_from_soft(&Syndrome, &BpSoft) -> Correction`, both already public.
- Produces: `OsdDecoder::with_residual_restricted(bool) -> Self` (builder, chains after `with_order`). Task 4 uses it to build candidates.

- [ ] **Step 1: Write the failing tests**

Append inside `mod tests` in `crates/aleph-qec/src/osd.rs`:

```rust
    #[test]
    fn test_residual_restricted_is_opt_in_and_chains() {
        let dem = crate::BBCode::gross().code_capacity_dem(0.05);
        let d = OsdDecoder::new(&dem).with_order(4);
        assert!(!d.residual_restricted);
        let d = d.with_residual_restricted(true);
        assert!(d.residual_restricted);
        assert_eq!(d.order, 4);
    }

    #[test]
    fn test_residual_restricted_still_satisfies_the_syndrome() {
        // OSD's contract is a syndrome-consistent decode. Restricting which columns the
        // combination sweep explores must not break it: the pivots are still solved for H e = s,
        // only the sweep pool shrinks.
        let dem = crate::BBCode::gross().code_capacity_dem(0.05);
        let d = OsdDecoder::new(&dem).with_order(4).with_residual_restricted(true);
        let (syndromes, _truths) = crate::sample_shots(&dem, 200, 7);
        for syn in &syndromes {
            let (corr, _ran) = d.decode_osd(syn);
            assert!(
                d.check_satisfied(syn, &corr),
                "residual-restricted OSD returned a syndrome-violating decode"
            );
        }
    }

    #[test]
    fn test_residual_restricted_matches_plain_when_all_vars_are_in_the_residual() {
        // Order 0 has no sweep at all, so the restriction is a no-op there — a guard against the
        // flag accidentally changing the OSD-0 path, which is the reference candidate.
        let dem = crate::BBCode::gross().code_capacity_dem(0.05);
        let plain = OsdDecoder::new(&dem).with_order(0);
        let restricted = OsdDecoder::new(&dem).with_order(0).with_residual_restricted(true);
        let (syndromes, _truths) = crate::sample_shots(&dem, 100, 11);
        for syn in &syndromes {
            assert_eq!(
                plain.decode_osd(syn).0.observable_flips,
                restricted.decode_osd(syn).0.observable_flips
            );
        }
    }
```

Append inside `mod tests` in `crates/aleph-qec/src/fixed_bp.rs`:

```rust
    #[test]
    fn test_osd_tail_does_not_run_on_converged_shots() {
        // The tail-rate is the cost metric for the Q7-07 policy, so the gate must be explicit in
        // the caller, not an internal short-circuit of OsdDecoder we happen to inherit.
        let dem = crate::BBCode::gross().code_capacity_dem(0.01);
        let osd = FixedRelayBpOsd::new(&dem, MSG_BITS_TEST, FRAC_BITS_TEST, 0);
        let (syndromes, _truths) = crate::sample_shots(&dem, 300, 3);
        for syn in &syndromes {
            let converged = osd.fixed().decode_fixed(syn).1;
            let (_corr, tail_ran) = osd.decode_fixed_osd(syn);
            assert_eq!(tail_ran, !converged);
        }
    }
```

If `MSG_BITS_TEST` / `FRAC_BITS_TEST` do not already exist in that test module, use the literal `8, 3` instead.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p aleph-qec --lib osd:: 2>&1 | tail -20`
Expected: FAIL — `no method named with_residual_restricted`, `no method named check_satisfied`.

- [ ] **Step 3: Implement the residual-restricted sweep**

In `crates/aleph-qec/src/osd.rs`, add to the `OsdDecoder` struct:

```rust
    /// Check **rows**: the variables each detector touches. The transpose of `var_dets`, built
    /// once so the residual (unsatisfied-check) support can be computed per shot without a scan.
    det_vars: Vec<Vec<u32>>,
    /// Restrict the combination sweep to variables touching an unsatisfied check (Q7-07's
    /// "OSD-lite on the residual"). Same `2^order` solve budget, aimed at the columns that can
    /// actually repair the violated checks.
    residual_restricted: bool,
```

In `with_params`, after `var_dets` is built and before the struct literal:

```rust
        let mut det_vars: Vec<Vec<u32>> = vec![Vec::new(); num_detectors];
        for (v, dets) in var_dets.iter().enumerate() {
            for &c in dets {
                det_vars[c as usize].push(v as u32);
            }
        }
```

and add `det_vars,` and `residual_restricted: false,` to the struct literal.

Add the builder next to `with_order`:

```rust
    /// Restrict the OSD combination sweep to variables in the support of the **unsatisfied**
    /// checks. No effect at `order == 0` (there is no sweep). Q7-07 candidate 3.
    pub fn with_residual_restricted(mut self, on: bool) -> Self {
        self.residual_restricted = on;
        self
    }

    /// Whether `corr`'s error pattern reproduces `syndrome` under this decoder's parity checks.
    /// Exposed for tests and for the Q7-07 campaign's per-shot validity accounting.
    pub fn check_satisfied(&self, syndrome: &Syndrome, corr: &Correction) -> bool {
        self.residual(syndrome, &corr.error).is_empty()
    }

    /// The detectors whose parity under `ehat` disagrees with `syndrome` — the residual.
    fn residual(&self, syndrome: &Syndrome, ehat: &[u8]) -> Vec<u32> {
        let mut parity = vec![false; self.num_detectors];
        for (v, dets) in self.var_dets.iter().enumerate() {
            if ehat.get(v).copied().unwrap_or(0) == 1 {
                for &c in dets {
                    parity[c as usize] ^= true;
                }
            }
        }
        for &d in &syndrome.fired {
            if (d as usize) < self.num_detectors {
                parity[d as usize] ^= true;
            }
        }
        (0..self.num_detectors as u32)
            .filter(|&c| parity[c as usize])
            .collect()
    }
```

If `Correction` has no public `error` field carrying the per-variable decision, add the residual computation directly over `ehat` in the test instead — read `crates/aleph-qec/src/syndrome.rs` first and adapt `check_satisfied` to whatever `Correction` exposes; the `residual` helper itself takes `&[u8]` and is unaffected.

In `osd_solve`, replace the `nonpivot` construction:

```rust
        let nonpivot: Vec<usize> = order
            .iter()
            .copied()
            .filter(|&v| row_for_col[v] == usize::MAX)
            .collect();
```

with:

```rust
        // Sweep pool: non-pivot columns, optionally narrowed to those touching an unsatisfied
        // check. The sweep costs 2^w regardless of pool size, so narrowing the pool raises the
        // *effective* order — the w columns actually explored are the ones that can repair the
        // residual, rather than the globally least-reliable ones anywhere in the code.
        let restrict: Option<std::collections::HashSet<usize>> = if self.residual_restricted {
            let resid = self.residual(syndrome, bp_hard);
            Some(
                resid
                    .iter()
                    .flat_map(|&c| self.det_vars[c as usize].iter().map(|&v| v as usize))
                    .collect(),
            )
        } else {
            None
        };
        let nonpivot: Vec<usize> = order
            .iter()
            .copied()
            .filter(|&v| row_for_col[v] == usize::MAX)
            .filter(|v| restrict.as_ref().is_none_or(|s| s.contains(v)))
            .collect();
```

`is_none_or` is stable since Rust 1.82; the workspace floor is 1.89, so it is available.

- [ ] **Step 4: Implement the explicit tail gate**

In `crates/aleph-qec/src/fixed_bp.rs`, replace the body of `decode_fixed_osd`:

```rust
    pub fn decode_fixed_osd(&self, syndrome: &Syndrome) -> (Correction, bool) {
        let soft = self.fixed.decode_fixed_soft(syndrome);
        // Gate the tail here, explicitly. `OsdDecoder::correction_from_soft` also short-circuits
        // on `converged`, but the Q7-07 policy measurement costs the tail by how often this branch
        // is taken — that must not depend on a callee's internal behaviour.
        if soft.converged {
            return (self.fixed.correction_of(&soft.ehat), false);
        }
        (self.osd.correction_from_soft(syndrome, &soft), true)
    }
```

If `FixedRelayBp::correction_of` is private, make it `pub` (it is the natural companion to `decode_fixed_ehat`, which already exposes `ehat`) and give it a one-line rustdoc.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p aleph-qec --lib osd:: && cargo test -p aleph-qec --lib fixed_bp::`
Expected: PASS.

- [ ] **Step 6: Verify no regression in the existing OSD/relay gates**

Run: `cargo test -p aleph-qec 2>&1 | tail -30`
Expected: all existing tests pass, including `tests/bposd.rs` and `tests/relay_bp.rs`.

- [ ] **Step 7: Lint, format, commit**

```bash
cargo fmt
cargo clippy -p aleph-qec --all-targets -- -D warnings
git add crates/aleph-qec/src/osd.rs crates/aleph-qec/src/fixed_bp.rs
git commit -m "[Q7-07] Residual-restricted OSD sweep + explicit tail gating

The combination sweep took the w least-reliable non-pivot columns from the
whole basis, so at sub-threshold p it mostly explored columns that cannot
affect the violated checks. Restricting the pool to the unsatisfied-check
support keeps the 2^w budget but raises the effective order.

FixedRelayBpOsd now gates its tail explicitly rather than inheriting
OsdDecoder's internal short-circuit — the tail-rate is Q7-07's cost metric
and must not depend on a callee's internals."
```

---

### Task 3: Block-path campaign — Levels 1 and 2

The rate and the ceiling. Chunked so a 10⁷-shot stream never materialises at once.

**Files:**
- Create: `crates/aleph-qec/examples/qec_q7_nonconv.rs`

**Interfaces:**
- Consumes: `aleph_qec::{sample_shots, BBCode, CircuitNoise, FixedRelayBp, LogicalErrorResult}`; `FixedRelayBp::with_budget(&dem, LEGS, ITERS, GAMMA, SEED, MSG_BITS, FRAC_BITS)`; `FixedRelayBp::decode_fixed_ehat`; `FixedRelayBp::iters_to_valid`.
- Produces: the `block` subcommand, a CSV on stdout, and a corpus file per operating point at `<out_prefix>-p<ppp>.corpus`. Task 4 reads that corpus.

**Chunk seeding:** chunk `c` samples with `seed ^ (c as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)`. Deterministic, reproducible, and independent across chunks. Record it in the corpus header so Task 4 can re-derive the exact shot.

**Corpus format** (text, one retained shot per line, small by construction — ~10³ rows):

```
# mode=block rounds=1 p=0.003 shots=10000000 seed=2024 retained=1042
# dets;truth
3 17 44;0 0 0 0 0 0 0 0 0 0 0 0
```

- [ ] **Step 1: Write the example**

Create `crates/aleph-qec/examples/qec_q7_nonconv.rs`:

```rust
//! Q7-07 — non-convergence rate, attributable fraction, and fallback evaluation.
//!
//! Relay-BP occasionally emits a hard decision violating the syndrome; the RTL flags it
//! (`valid_flag`, `hw/bp_relay_banked.sv:968`) and emits it anyway. This measures what that costs.
//!
//! A direct A/B on campaign LER is statistically hopeless — at p=0.003 the LER CI is ±1.13e-4 at
//! 10⁶ shots while the non-convergence rate is order 0.1 %, so a fallback's effect sits orders of
//! magnitude under the noise floor. So the measurement is conditional:
//!
//!   L1  r(p) = P(valid = 0), with CI.
//!   L2  A(p) = (# logical errors with valid=0) / (# logical errors) — the HARD CEILING on any
//!       fallback, since a fallback only ever acts on valid=0 shots. Also P(err | valid) both ways.
//!   L3  conditional rescue on the retained valid=0 corpus (see the `candidates` subcommand),
//!       propagated back as ΔLER(p) = r(p) · [P(err|v=0) − P(err|v=0, fallback)].
//!
//! Usage:
//!   cargo run --release -p aleph-qec --example qec_q7_nonconv -- block  [rounds] [shots] [seed] [out_prefix]
//!   cargo run --release -p aleph-qec --example qec_q7_nonconv -- window [rounds] [shots] [seed]
//!   cargo run --release -p aleph-qec --example qec_q7_nonconv -- candidates <corpus-file>
//!   # block defaults:  rounds=1  shots=1000000 seed=2024 out_prefix=q7-07
//!   # window defaults: rounds=12 shots=20000   seed=2024

use aleph_qec::{sample_shots, BBCode, CircuitNoise, FixedRelayBp, LogicalErrorResult, Syndrome};
use rayon::prelude::*;

const MSG_BITS: u32 = 8;
const FRAC_BITS: u32 = 3;
const LEGS: usize = 6;
const ITERS: u32 = 10;
const GAMMA: (f64, f64) = (-0.3, 0.9);
const SEED: u64 = 0x5E1A_4B9C;

/// The Q7-06 on-silicon campaign points (rounds=1 vehicle).
const BLOCK_PS: &[f64] = &[0.003, 0.005, 0.007];
/// Shots per chunk — bounds peak memory; the stream is `shots` total across chunks.
const CHUNK: u64 = 1_000_000;
/// Level-3 corpus target: retained non-converged shots per operating point.
const CORPUS_TARGET: usize = 1000;

fn mispredicted(pred: &[bool], truth: &[bool], observables: usize) -> bool {
    (0..observables)
        .any(|o| pred.get(o).copied().unwrap_or(false) != truth.get(o).copied().unwrap_or(false))
}

/// Per-chunk seed: independent, deterministic, reproducible from `(seed, chunk)`.
fn chunk_seed(seed: u64, chunk: u64) -> u64 {
    seed ^ chunk.wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

#[derive(Default, Clone, Copy)]
struct Counts {
    shots: u64,
    nonconv: u64,
    err_total: u64,
    err_nonconv: u64,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("block") => {
            let rounds = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1usize);
            let shots = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1_000_000u64);
            let seed = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2024u64);
            let prefix = args.get(4).cloned().unwrap_or_else(|| "q7-07".to_string());
            run_block(rounds, shots, seed, &prefix);
        }
        other => {
            eprintln!("unknown subcommand {other:?}");
            eprintln!("usage: qec_q7_nonconv -- block|window|candidates ...");
            std::process::exit(2);
        }
    }
}

fn run_block(rounds: usize, shots: u64, seed: u64, prefix: &str) {
    let code = BBCode::gross();
    eprintln!(
        "# Q7-07 block path: gross [[144,12,12]] circuit-level rounds={rounds} shots={shots} \
         seed={seed} schedule={LEGS}x{ITERS} word=Q{}.{}",
        MSG_BITS - 1 - FRAC_BITS,
        FRAC_BITS
    );
    println!("p,shots,r,r_ci95,ler,ler_ci95,p_err_given_nonconv,p_err_given_conv,attributable,iters_mean,iters_p50,iters_p99,iters_max,retained");

    for &p in BLOCK_PS {
        let dem = code
            .circuit_level_dem(rounds, CircuitNoise::uniform(p))
            .expect("circuit-level DEM");
        let fx = FixedRelayBp::with_budget(&dem, LEGS, ITERS, GAMMA, SEED, MSG_BITS, FRAC_BITS);

        let mut c = Counts::default();
        let mut iters_hist: Vec<u32> = Vec::new();
        let mut corpus: Vec<(Vec<u32>, Vec<bool>, u64)> = Vec::new();

        let mut done: u64 = 0;
        let mut chunk_idx: u64 = 0;
        while done < shots {
            let n = CHUNK.min(shots - done);
            let cs = chunk_seed(seed, chunk_idx);
            let (syndromes, truths) = sample_shots(&dem, n, cs);
            let out: Vec<(bool, bool, u32)> = syndromes
                .par_iter()
                .zip(&truths)
                .map(|(syn, truth)| {
                    let (_ehat, flips, valid) = fx.decode_fixed_ehat(syn);
                    let (_conv, iters) = fx.iters_to_valid(syn);
                    (valid, mispredicted(&flips, truth, dem.observables), iters)
                })
                .collect();

            for (i, &(valid, err, iters)) in out.iter().enumerate() {
                c.shots += 1;
                iters_hist.push(iters);
                if err {
                    c.err_total += 1;
                }
                if !valid {
                    c.nonconv += 1;
                    if err {
                        c.err_nonconv += 1;
                    }
                    if corpus.len() < CORPUS_TARGET {
                        corpus.push((syndromes[i].fired.clone(), truths[i].clone(), cs));
                    }
                }
            }
            done += n;
            chunk_idx += 1;
            eprintln!(
                "p={p}: {done}/{shots} shots, nonconv {} ({:.4}%), corpus {}",
                c.nonconv,
                100.0 * c.nonconv as f64 / c.shots as f64,
                corpus.len()
            );
        }

        write_corpus(prefix, rounds, p, shots, seed, &corpus);
        report_block(p, &c, &mut iters_hist, corpus.len());
    }
}

fn report_block(p: f64, c: &Counts, iters: &mut [u32], retained: usize) {
    let r = LogicalErrorResult::new(c.shots, c.nonconv);
    let ler = LogicalErrorResult::new(c.shots, c.err_total);
    let conv_shots = c.shots - c.nonconv;
    let p_err_nc = if c.nonconv > 0 {
        c.err_nonconv as f64 / c.nonconv as f64
    } else {
        f64::NAN
    };
    let p_err_c = if conv_shots > 0 {
        (c.err_total - c.err_nonconv) as f64 / conv_shots as f64
    } else {
        f64::NAN
    };
    // The ceiling: a fallback only ever acts on valid=0 shots, so even a perfect one removes at
    // most this fraction of the logical errors.
    let attributable = if c.err_total > 0 {
        c.err_nonconv as f64 / c.err_total as f64
    } else {
        f64::NAN
    };
    iters.sort_unstable();
    let pct = |q: f64| -> u32 {
        if iters.is_empty() {
            return 0;
        }
        let i = ((iters.len() as f64 - 1.0) * q).round() as usize;
        iters[i]
    };
    let mean = iters.iter().map(|&x| x as f64).sum::<f64>() / iters.len().max(1) as f64;
    println!(
        "{p},{},{:.8},{:.8},{:.8},{:.8},{:.6},{:.8},{:.6},{:.2},{},{},{},{retained}",
        c.shots,
        r.rate,
        r.ci95,
        ler.rate,
        ler.ci95,
        p_err_nc,
        p_err_c,
        attributable,
        mean,
        pct(0.50),
        pct(0.99),
        iters.last().copied().unwrap_or(0)
    );
    eprintln!(
        "p={p}: r={:.4e} ±{:.1e} | LER={:.4e} | P(err|v=0)={p_err_nc:.4} P(err|v=1)={p_err_c:.4e} \
         | A={attributable:.4}",
        r.rate, r.ci95, ler.rate
    );
}

fn write_corpus(
    prefix: &str,
    rounds: usize,
    p: f64,
    shots: u64,
    seed: u64,
    corpus: &[(Vec<u32>, Vec<bool>, u64)],
) {
    use std::io::Write;
    let path = format!("{prefix}-p{:03.0}.corpus", p * 1000.0);
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path).expect("create corpus"));
    writeln!(
        f,
        "# mode=block rounds={rounds} p={p} shots={shots} seed={seed} retained={}",
        corpus.len()
    )
    .expect("write corpus");
    writeln!(f, "# dets;truth").expect("write corpus");
    for (dets, truth, _cs) in corpus {
        let d: Vec<String> = dets.iter().map(|x| x.to_string()).collect();
        let t: Vec<String> = truth.iter().map(|&b| u8::from(b).to_string()).collect();
        writeln!(f, "{};{}", d.join(" "), t.join(" ")).expect("write corpus");
    }
    eprintln!("# wrote {path} ({} retained shots)", corpus.len());
}
```

Note `Syndrome` is imported for Task 4; if the compiler warns it is unused at this point, leave the import out and add it in Task 4.

- [ ] **Step 2: Run a smoke campaign to verify it works**

Run:
```bash
cargo run --release -p aleph-qec --example qec_q7_nonconv -- block 1 20000 2024 /tmp/q707 2>&1 | tail -20
```
Expected: three CSV rows (p=0.003/0.005/0.007), non-zero `attributable` at the higher rates, and three `/tmp/q707-p00{3,5,7}.corpus` files written. `r` at p=0.003 should be small (order 1e-3 or below), consistent with the 99.9 % converged figure in `docs/perf/qec-q7-fixed-bp.md:968-973`.

- [ ] **Step 3: Sanity-check the corpus file**

Run: `head -3 /tmp/q707-p007.corpus && wc -l /tmp/q707-p007.corpus`
Expected: two `#` header lines then rows of the form `3 17 44;0 0 0 ...`; line count = retained + 2.

- [ ] **Step 4: Lint, format, commit**

```bash
cargo fmt
cargo clippy -p aleph-qec --all-targets -- -D warnings
git add crates/aleph-qec/examples/qec_q7_nonconv.rs
git commit -m "[Q7-07] Block-path campaign: rate, attributable fraction, corpus dump

Levels 1 and 2 of the conditional measurement. A(p) — the share of logical
errors carrying valid_flag=0 — is the hard ceiling on any fallback, since a
fallback only ever acts on those shots. Chunked sampling keeps a 10^7-shot
stream off the heap; the retained non-converged corpus feeds Level 3."
```

---

### Task 4: Level 3 — candidate evaluation on the retained corpus

**Files:**
- Modify: `crates/aleph-qec/examples/qec_q7_nonconv.rs`

**Interfaces:**
- Consumes: the corpus file from Task 3; `OsdDecoder::with_order` / `with_residual_restricted` from Task 2; `FixedRelayBp::decode_fixed_soft -> BpSoft`; `OsdDecoder::correction_from_soft`.
- Produces: the `candidates` subcommand and its CSV.

Candidates, per the spec: OSD-0, OSD-2, OSD-4, and residual-restricted OSD-2/OSD-4. Baseline = the relay-BP best-kept decision (what the RTL emits today).

- [ ] **Step 1: Add the subcommand**

Add to `main`'s match, above the `other =>` arm:

```rust
        Some("candidates") => {
            let path = args.get(1).cloned().unwrap_or_else(|| {
                eprintln!("candidates: needs a corpus file");
                std::process::exit(2)
            });
            run_candidates(&path);
        }
```

Append to the file:

```rust
struct Corpus {
    rounds: usize,
    p: f64,
    shots: u64,
    shot_rate: f64,
    entries: Vec<(Vec<u32>, Vec<bool>)>,
}

fn read_corpus(path: &str) -> Corpus {
    let text = std::fs::read_to_string(path).expect("read corpus");
    let mut rounds = 1usize;
    let mut p = 0.0f64;
    let mut shots = 0u64;
    let mut retained = 0usize;
    let mut entries = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("# mode=") {
            for kv in rest.split_whitespace() {
                let Some((k, v)) = kv.split_once('=') else { continue };
                match k {
                    "rounds" => rounds = v.parse().expect("rounds"),
                    "p" => p = v.parse().expect("p"),
                    "shots" => shots = v.parse().expect("shots"),
                    "retained" => retained = v.parse().expect("retained"),
                    _ => {}
                }
            }
            continue;
        }
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let (d, t) = line.split_once(';').expect("corpus row");
        let dets: Vec<u32> = d
            .split_whitespace()
            .map(|x| x.parse().expect("det"))
            .collect();
        let truth: Vec<bool> = t.split_whitespace().map(|x| x == "1").collect();
        entries.push((dets, truth));
    }
    assert_eq!(entries.len(), retained, "corpus header/row count disagree");
    // r(p) as measured by the run that produced this corpus — needed to propagate the conditional
    // rescue back to an overall ΔLER. Recomputed from the header, not re-measured.
    let shot_rate = 0.0; // filled by the caller from the block CSV; see run_candidates.
    let _ = shot_rate;
    Corpus { rounds, p, shots, shot_rate: 0.0, entries }
}

fn run_candidates(path: &str) {
    let corpus = read_corpus(path);
    let code = BBCode::gross();
    let dem = code
        .circuit_level_dem(corpus.rounds, CircuitNoise::uniform(corpus.p))
        .expect("circuit-level DEM");
    let fx = FixedRelayBp::with_budget(&dem, LEGS, ITERS, GAMMA, SEED, MSG_BITS, FRAC_BITS);
    let n = corpus.entries.len();
    let n_dets = dem.detectors;

    eprintln!(
        "# Q7-07 candidates on {path}: p={} rounds={} corpus={n} (from {} shots)",
        corpus.p, corpus.rounds, corpus.shots
    );
    println!("p,candidate,order,restricted,corpus,errors,p_err_given_nonconv,solves_per_shot,us_per_shot");

    // Baseline: what the RTL emits today — the best-kept decision, syndrome-violating and all.
    let base_err: Vec<bool> = corpus
        .entries
        .par_iter()
        .map(|(dets, truth)| {
            let syn = Syndrome::new(n_dets, dets.clone());
            let (_e, flips, _v) = fx.decode_fixed_ehat(&syn);
            mispredicted(&flips, truth, dem.observables)
        })
        .collect();
    report_candidate(corpus.p, "baseline", 0, false, &base_err, 0.0, 0.0);

    for (order, restricted) in [(0, false), (2, false), (4, false), (2, true), (4, true)] {
        let osd = aleph_qec::OsdDecoder::new(&dem)
            .with_order(order)
            .with_residual_restricted(restricted);
        let t0 = std::time::Instant::now();
        let errs: Vec<bool> = corpus
            .entries
            .par_iter()
            .map(|(dets, truth)| {
                let syn = Syndrome::new(n_dets, dets.clone());
                let soft = fx.decode_fixed_soft(&syn);
                let corr = osd.correction_from_soft(&syn, &soft);
                mispredicted(&corr.observable_flips, truth, dem.observables)
            })
            .collect();
        let us = 1e6 * t0.elapsed().as_secs_f64() / n as f64;
        let name = if restricted { "osd-resid" } else { "osd" };
        report_candidate(corpus.p, name, order, restricted, &errs, (1u64 << order) as f64, us);
        // Paired McNemar against the baseline: the candidates decode the SAME shots, so the
        // unpaired difference of two rates throws away most of the power.
        mcnemar(&base_err, &errs, name, order, restricted);
    }
}

fn report_candidate(
    p: f64,
    name: &str,
    order: usize,
    restricted: bool,
    errs: &[bool],
    solves: f64,
    us: f64,
) {
    let e = errs.iter().filter(|&&x| x).count();
    println!(
        "{p},{name},{order},{},{},{e},{:.6},{solves},{us:.1}",
        u8::from(restricted),
        errs.len(),
        e as f64 / errs.len().max(1) as f64
    );
}

/// McNemar's paired test on the corpus: `b` = baseline wrong & candidate right, `c` = the reverse.
/// Reports the two discordant counts and the χ² statistic (1 dof, continuity-corrected).
fn mcnemar(base: &[bool], cand: &[bool], name: &str, order: usize, restricted: bool) {
    let b = base.iter().zip(cand).filter(|(&x, &y)| x && !y).count();
    let c = base.iter().zip(cand).filter(|(&x, &y)| !x && y).count();
    let chi2 = if b + c == 0 {
        0.0
    } else {
        let d = (b as f64 - c as f64).abs() - 1.0;
        (d.max(0.0)).powi(2) / (b + c) as f64
    };
    eprintln!(
        "  {name}-{order}{}: rescued {b}, broke {c}, chi2={chi2:.2} ({})",
        if restricted { "-resid" } else { "" },
        if chi2 > 3.84 { "significant at 0.05" } else { "not significant" }
    );
}
```

Ensure `use aleph_qec::Syndrome;` is present in the import list (add it if Task 3 left it out).

- [ ] **Step 2: Run it on the smoke corpus**

Run:
```bash
cargo run --release -p aleph-qec --example qec_q7_nonconv -- candidates /tmp/q707-p007.corpus 2>&1 | tail -20
```
Expected: a `baseline` row plus five candidate rows, and stderr McNemar lines. The baseline's `p_err_given_nonconv` should be high (non-converged shots are mostly lost); OSD-0 is expected to be **no better or worse**, per the prior verdict in `docs/perf/qec-q7-fixed-bp.md:587-640`.

- [ ] **Step 3: Verify the corpus round-trips exactly**

The corpus stores fired detectors, so re-decoding must reproduce the non-convergence that got each shot retained. Add this assertion inside `run_candidates`, after `base_err` is computed:

```rust
    let still_nonconv = corpus
        .entries
        .par_iter()
        .filter(|(dets, _)| !fx.decode_fixed_ehat(&Syndrome::new(n_dets, dets.clone())).2)
        .count();
    assert_eq!(
        still_nonconv, n,
        "corpus round-trip broken: {still_nonconv}/{n} shots still non-converged"
    );
```

Run the command from Step 2 again.
Expected: no assertion failure.

- [ ] **Step 4: Lint, format, commit**

```bash
cargo fmt
cargo clippy -p aleph-qec --all-targets -- -D warnings
git add crates/aleph-qec/examples/qec_q7_nonconv.rs
git commit -m "[Q7-07] Level 3: candidate evaluation on the retained corpus

Candidates decode the same retained non-converged shots as the baseline, so
the comparison is paired (McNemar) rather than a difference of two rates —
the corpus is ~10^3 shots and the unpaired test would have almost no power.
Cost is reported in both GF(2) solves and measured microseconds, because a
candidate that wins on LER but blows the 1 us/round budget is still dead."
```

---

### Task 5: Window-path campaign

Secondary path, reported for coverage. The flag means something different here — one window, not one shot — so both normalisations are emitted, alongside the `commit_clean` / discarded-bits signal M9b argues is sharper.

**Files:**
- Modify: `crates/aleph-qec/examples/qec_q7_nonconv.rs`

**Interfaces:**
- Consumes: `aleph_qec::HwSlidingWindowBp::new(dem, detector_round, window, commit)`; `decode_stream_trace(&Syndrome) -> (Correction, StreamStats, Vec<WindowTrace>)`; `StreamStats { windows, nonconverged, residual }`; `WindowTrace { valid, commit_clean, .. }`; `BBCode::memory_x_experiment(rounds).detector_rounds()`.
- Produces: the `window` subcommand and its CSV.

- [ ] **Step 1: Add the subcommand**

Add to `main`'s match:

```rust
        Some("window") => {
            let rounds = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(12usize);
            let shots = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20_000u64);
            let seed = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2024u64);
            run_window(rounds, shots, seed);
        }
```

Append:

```rust
/// M9b's frozen streaming configuration.
const WINDOW_W: usize = 6;
const WINDOW_C: usize = 2;
/// Circuit-level rates M9b characterised the window path at.
const WINDOW_PS: &[f64] = &[0.001, 0.003, 0.005];

fn run_window(rounds: usize, shots: u64, seed: u64) {
    let code = BBCode::gross();
    eprintln!(
        "# Q7-07 window path: gross rounds={rounds} shots={shots} seed={seed} W={WINDOW_W} C={WINDOW_C}"
    );
    println!("p,shots,windows,r_per_window,r_per_shot,ler,ler_ci95,p_err_given_nonconv_shot,p_err_given_conv_shot,attributable,dirty_commit_frac,resid_frac");

    for &p in WINDOW_PS {
        let dem = code
            .circuit_level_dem(rounds, CircuitNoise::uniform(p))
            .expect("circuit-level DEM");
        let dr = code.memory_x_experiment(rounds).detector_rounds();
        let hw = HwSlidingWindowBp::new(dem.clone(), dr, WINDOW_W, WINDOW_C);
        let (syndromes, truths) = sample_shots(&dem, shots, seed);

        let rows: Vec<(usize, usize, bool, bool, usize)> = syndromes
            .par_iter()
            .zip(&truths)
            .map(|(syn, truth)| {
                let (corr, stats, trace) = hw.decode_stream_trace(syn);
                let dirty = trace.iter().filter(|t| !t.commit_clean).count();
                (
                    stats.windows,
                    stats.nonconverged,
                    mispredicted(&corr.observable_flips, truth, dem.observables),
                    dirty > 0,
                    stats.residual,
                )
            })
            .collect();

        let windows: u64 = rows.iter().map(|r| r.0 as u64).sum();
        let nonconv_w: u64 = rows.iter().map(|r| r.1 as u64).sum();
        let nonconv_shots = rows.iter().filter(|r| r.1 > 0).count() as u64;
        let errs = rows.iter().filter(|r| r.2).count() as u64;
        let err_nonconv = rows.iter().filter(|r| r.2 && r.1 > 0).count() as u64;
        let dirty_shots = rows.iter().filter(|r| r.3).count() as u64;
        let resid_shots = rows.iter().filter(|r| r.4 > 0).count() as u64;
        let ler = LogicalErrorResult::new(shots, errs);
        let conv_shots = shots - nonconv_shots;

        println!(
            "{p},{shots},{windows},{:.8},{:.6},{:.8},{:.8},{:.6},{:.6},{:.6},{:.6},{:.6}",
            nonconv_w as f64 / windows.max(1) as f64,
            nonconv_shots as f64 / shots as f64,
            ler.rate,
            ler.ci95,
            if nonconv_shots > 0 { err_nonconv as f64 / nonconv_shots as f64 } else { f64::NAN },
            if conv_shots > 0 { (errs - err_nonconv) as f64 / conv_shots as f64 } else { f64::NAN },
            if errs > 0 { err_nonconv as f64 / errs as f64 } else { f64::NAN },
            dirty_shots as f64 / shots as f64,
            resid_shots as f64 / shots as f64
        );
        eprintln!(
            "p={p}: per-window r={:.4} | per-shot r={:.4} | LER={:.3e} | A={:.4} | dirty-commit {:.4}",
            nonconv_w as f64 / windows.max(1) as f64,
            nonconv_shots as f64 / shots as f64,
            ler.rate,
            if errs > 0 { err_nonconv as f64 / errs as f64 } else { f64::NAN },
            dirty_shots as f64 / shots as f64
        );
    }
}
```

Add `HwSlidingWindowBp` to the `use aleph_qec::{...}` import list.

- [ ] **Step 2: Run a smoke campaign**

Run:
```bash
cargo run --release -p aleph-qec --example qec_q7_nonconv -- window 12 2000 2024 2>&1 | tail -12
```
Expected: three rows. Cross-check against M9b (`docs/perf/qec-q7-fixed-bp.md:1400-1410`): `r_per_shot` should land near 0.118 / 0.668 / 0.960 at p = 0.001 / 0.003 / 0.005, and `r_per_window` must be substantially **lower** than `r_per_shot` (a shot has many windows). If `r_per_shot` is far off those figures, stop and investigate before continuing — the configuration does not match M9b's.

- [ ] **Step 3: Lint, format, commit**

```bash
cargo fmt
cargo clippy -p aleph-qec --all-targets -- -D warnings
git add crates/aleph-qec/examples/qec_q7_nonconv.rs
git commit -m "[Q7-07] Window-path campaign: per-window and per-shot rates

The flag counts windows here, not shots, so M9b's 12/67/96 % figures are the
per-shot 'at least one bad window' normalisation and overstate the per-decode
problem. Both are emitted, alongside the commit_clean/discarded-bits signal
M9b argues is the sharper health metric."
```

---

### Task 6: Board driver — capture `valid_flag` and read `.ref` v2

**Files:**
- Modify: `hw/sw/bp_stream_banked_ler_kv260.py:85-135`

**Interfaces:**
- Consumes: the `.ref` v2 layout from Task 1 (`hw/bp_stream_banked_core.sv:28` gives the status word: `[31:20]=obs_flip`, `[19]=valid_flag`, `[15:0]=latency_cycles`).
- Produces: two new report columns, `rtl_valid` and `valid_mismatch`, plus a hard gate on the latter.

`hw/sw/bp_stream_banked_kv260.py:216` already reads bit 19 the same way — use it as the reference for the shift/mask.

- [ ] **Step 1: Return the full status word from `run_chunk`**

Replace the last line of `run_chunk`:

```python
        return (np.asarray(ob[:n]) >> 20) & obs_mask  # rtl_obs per shot
```

with:

```python
        return np.asarray(ob[:n]).copy()  # full status word: [31:20]=obs, [19]=valid, [15:0]=cycles
```

- [ ] **Step 2: Parse `.ref` v2 and unpack the status word**

Replace the `.ref` load block:

```python
        ref = np.fromfile(prefix + ".ref", dtype=np.uint16)
        n = syn.size // NS
        assert ref.size == 2 * n, "%s.ref size mismatch (%d vs %d)" % (prefix, ref.size, 2 * n)
        true_obs = ref[0::2].astype(np.uint32)
        sw_obs = ref[1::2].astype(np.uint32)
```

with:

```python
        # .ref v2 (aleph_qec::refvec): header [magic, version, words_per_shot, 0] then
        # [true_obs, sw_obs, meta] per shot, meta = (valid << 15) | iters.
        ref = np.fromfile(prefix + ".ref", dtype="<u2")
        n = syn.size // NS
        if ref.size < 4 or ref[0] != 0xA1E7:
            raise SystemExit(
                "%s.ref is a legacy v1 file (no header). Regenerate it: "
                "qec_q7_bp_graph -- silvectors <rounds> <p> <n> <seed> %s <decoder_p>"
                % (prefix, prefix))
        if ref[1] != 2 or ref[2] != 3:
            raise SystemExit("%s.ref: unsupported version/width %d/%d" % (prefix, ref[1], ref[2]))
        body = ref[4:]
        assert body.size == 3 * n, "%s.ref size mismatch (%d vs %d)" % (prefix, body.size, 3 * n)
        true_obs = body[0::3].astype(np.uint32)
        sw_obs = body[1::3].astype(np.uint32)
        sw_valid = (body[2::3] >> 15).astype(np.uint32)
```

- [ ] **Step 3: Collect the status words and unpack**

Replace the collection loop and the metric block:

```python
        rtl_obs = np.empty(n, dtype=np.uint32)
        t0 = time.perf_counter()
        off = 0
        while off < n:
            m = min(chunk, n - off)
            rtl_obs[off:off + m] = run_chunk(syn[off * NS:(off + m) * NS], m)
            off += m
        dt = time.perf_counter() - t0
```

with:

```python
        status = np.empty(n, dtype=np.uint32)
        t0 = time.perf_counter()
        off = 0
        while off < n:
            m = min(chunk, n - off)
            status[off:off + m] = run_chunk(syn[off * NS:(off + m) * NS], m)
            off += m
        dt = time.perf_counter() - t0

        rtl_obs = (status >> 20) & obs_mask
        rtl_valid = (status >> 19) & 1
        rtl_cycles = status & 0xFFFF
        valid_mismatch = int(np.count_nonzero(rtl_valid != sw_valid))
        rtl_nonconv = int(np.count_nonzero(rtl_valid == 0))
```

- [ ] **Step 4: Report and gate**

After the existing per-point `print` of `(%.2f s, ...)`, add:

```python
        print("           (valid: rtl_nonconv=%d (%.4f%%), sw_nonconv=%d, mismatch=%d; "
              "cycles mean=%.1f max=%d)"
              % (rtl_nonconv, 100.0 * rtl_nonconv / n,
                 int(np.count_nonzero(sw_valid == 0)), valid_mismatch,
                 float(rtl_cycles.mean()), int(rtl_cycles.max())))
        if valid_mismatch:
            all_pass = False
```

And extend the final verdict string so a `valid_mismatch` failure is named:

```python
    print("\nAC-2 RESULT:", "PASS (RTL LER within CI of software golden; valid_flag matches at every point)"
          if all_pass else "FAIL (see rows)")
```

Update the module docstring's metric list to add `valid_mismatch = count(rtl_valid != sw_valid)  -- Q7-07 gate, must be 0`, and the usage line to note that `.ref` must be v2.

- [ ] **Step 5: Syntax-check the driver offline**

The board is not needed to catch a typo. Run:
```bash
python3 -m py_compile hw/sw/bp_stream_banked_ler_kv260.py && echo "compiles"
```
Expected: `compiles`.

- [ ] **Step 6: Verify the v2 parser against a real file offline**

Run:
```bash
cargo run --release -p aleph-qec --example qec_q7_bp_graph -- silvectors 1 0.005 5000 2024 /tmp/q707v2 0.005
python3 - <<'PY'
import numpy as np
ref = np.fromfile('/tmp/q707v2.ref', dtype='<u2')
assert ref[0] == 0xA1E7 and ref[1] == 2 and ref[2] == 3
body = ref[4:]
n = body.size // 3
sw_valid = (body[2::3] >> 15)
print('n=%d sw_nonconv=%d' % (n, int(np.count_nonzero(sw_valid == 0))))
PY
```
Expected: `n=5000 sw_nonconv=<small integer>`.

- [ ] **Step 7: Commit**

```bash
git add hw/sw/bp_stream_banked_ler_kv260.py
git commit -m "[Q7-07] LER driver: capture valid_flag (status bit 19) and read .ref v2

run_chunk masked off everything below bit 20 and threw away the flag the whole
ticket is about. It now returns the full status word and the driver unpacks
obs / valid / latency_cycles, gating on rtl_valid == sw_valid. Legacy v1 .ref
files are rejected with the regeneration command rather than misparsed."
```

---

### Task 7: Run the campaigns, confirm on silicon, write the report

Everything before this was instrumentation. This task produces the AC evidence.

**Files:**
- Create: `docs/perf/data/qec-q7-nonconv.csv`
- Create: `docs/qec/q7-07-nonconvergence-policy.md`
- Modify: `docs/qec/BACKLOG.md:1439-1441` (tick the two AC boxes)

**Interfaces:**
- Consumes: everything from Tasks 1–6.
- Produces: the ticket deliverable and the PR.

- [ ] **Step 1: Verify the bench box is idle before measuring**

Per CLAUDE.md's performance guidelines and `feedback-check-server-clean`:
```bash
ssh root@195.154.249.85 'uptime; pgrep -af "cargo bench|bencher run|Runner.Worker" || echo "clean"'
```
Expected: load ≈ 0 and `clean`. If a CI run is active, wait — a shared box silently inflates timing measurements, and this ticket reports microseconds per candidate.

- [ ] **Step 2: Run the block campaign at scale**

On the EPYC box, 10⁷ shots per point so the p=0.003 corpus reaches its 10³ target even at a ~1e-4 rate:
```bash
cargo run --release -p aleph-qec --example qec_q7_nonconv -- block 1 10000000 2024 q7-07 \
  > docs/perf/data/qec-q7-nonconv-block.csv 2> q7-07-block.log
```
Expected: three CSV rows and three `q7-07-p00{3,5,7}.corpus` files. Check `retained` in each row — if it is far below 1000 at p=0.003, the rate is lower than assumed and that is itself a finding; record the actual retained count rather than raising the shot count indefinitely.

- [ ] **Step 3: Run the candidate evaluation on each corpus**

```bash
for f in q7-07-p003.corpus q7-07-p005.corpus q7-07-p007.corpus; do
  cargo run --release -p aleph-qec --example qec_q7_nonconv -- candidates "$f"
done > docs/perf/data/qec-q7-nonconv-candidates.csv 2> q7-07-candidates.log
```
Expected: six rows per corpus (baseline + five candidates) and McNemar lines on stderr.

- [ ] **Step 4: Run the window campaign**

```bash
cargo run --release -p aleph-qec --example qec_q7_nonconv -- window 12 20000 2024 \
  > docs/perf/data/qec-q7-nonconv-window.csv 2> q7-07-window.log
```
Expected: three rows; `r_per_shot` near M9b's 0.118 / 0.668 / 0.960.

- [ ] **Step 5: Board confirmation**

First check the board and overlay are actually there:
```bash
ssh root@192.168.88.174 'ls -la ~/bp_kv260_stream_banked.bit 2>/dev/null || echo NO-OVERLAY; uptime'
```

If the overlay is present, generate v2 vectors at p=0.005 and run ~10⁵ shots:
```bash
cargo run --release -p aleph-qec --example qec_q7_bp_graph -- silvectors 1 0.005 100000 2024 p005v2 0.005
scp p005v2.syn p005v2.ref root@192.168.88.174:~/
ssh root@192.168.88.174 'cd ~ && sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 \
  bp_stream_banked_ler_kv260.py bp_kv260_stream_banked.bit p005v2'
```
**Gate: `mismatch=0`.** That is the ticket's only new hardware claim.

If the overlay is missing or the board is unreachable, fall back to the off-board gate instead and say so explicitly in the report:
```bash
make -C hw bpbanked-highweight
```
Expected: 2000/2000 bit-identical (this gate already compares `valid_flag`, per `hw/tb_bp_banked.cpp:104`).

- [ ] **Step 6: Apply the pre-registered decision rule**

From the spec, without amendment:
- `A(p) < 5 %` everywhere → **do-nothing-but-flag**.
- `A(p) ≥ 5 %` **and** a candidate's McNemar χ² > 3.84 in its favour **and** its measured µs/shot fits the 1 µs/round budget → that candidate wins.
- Wins on LER, breaks latency → **rejected-on-latency**, with the arithmetic shown.

If a candidate wins and it is an OSD variant, the implementation is a PS-side tail on `!valid_flag` shots — no RTL change. **If the data instead points at a hardware-side policy, stop and return to the user before touching any RTL** (spec § "Implementation boundary").

- [ ] **Step 7: Write the report**

Create `docs/qec/q7-07-nonconvergence-policy.md` following the shape of `docs/qec/q7-06-ac1-batched-dma.md`: a status line naming both ACs, what was measured and why the conditional design was necessary, the per-operating-point rate table, the attributable-fraction table with the ceiling stated in words, the candidate table (LER, solves/shot, µs/shot, McNemar), the window-path table with both normalisations, the board-confirmation result, the chosen policy with the decision rule quoted, and a reproduce section with the exact commands from Steps 2–5.

State the ceiling explicitly, e.g.: *"At p=0.003, A = X %, so no fallback whatsoever — including a perfect oracle — can reduce LER by more than X %."* That sentence is the ticket's main result if the verdict is do-nothing-but-flag.

Tick both AC boxes in `docs/qec/BACKLOG.md:1439-1441`.

- [ ] **Step 8: Verify before claiming completion**

```bash
cargo test -p aleph-qec 2>&1 | tail -5
cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -5
cargo fmt --check && echo "fmt ok"
```
Expected: tests pass, clippy clean, `fmt ok`. Confirm every table in the report is populated from a committed CSV — no number typed by hand.

- [ ] **Step 9: Commit and open the PR**

```bash
git add docs/qec/q7-07-nonconvergence-policy.md docs/perf/data/qec-q7-nonconv-*.csv docs/qec/BACKLOG.md
git commit -m "[Q7-07] Non-convergence rate, attributable ceiling, and the chosen policy

Both ACs. <one line naming the verdict and the ceiling.>"
git push -u origin q7-07-nonconvergence-policy
gh pr create --title "[Q7-07] Non-convergence fallback policy (valid_flag=0 path)" --body "..."
```

The PR body must contain `Closes #458` — **the issue number, not the PR number** (P0-06/07/08/11 all merged with the wrong reference and had to be closed by hand). Include: the approach summary, the rate and attributable tables, the candidate table, test results, the board-confirmation outcome, and anything left out.

---

## Self-Review

**Spec coverage.** Level 1 → Task 3. Level 2 (`A(p)`) → Task 3 (`report_block`). Level 3 → Task 4. Both decoder paths → Tasks 3 and 5. Latency constraint → Task 4 (`us_per_shot`) and Task 7 Step 6. Candidates OSD-0/OSD-w/residual-restricted → Task 2 (implementation) + Task 4 (evaluation). Emitter `.ref` change → Task 1. Explicit tail gating → Task 2. Driver change → Task 6. Verification (three-API agreement, `.ref` round-trip, board gate) → Task 1 Step 1, Task 2 Step 1, Task 6, Task 7 Step 5. Pre-registered decision rule → Task 7 Step 6. Deliverables → Task 7.

One spec item needed a task it did not obviously have: the spec's "unit test that `valid`, `converged`, and `iters_to_valid`'s bool agree on the same syndromes". Task 2's `test_osd_tail_does_not_run_on_converged_shots` covers `decode_fixed` vs `decode_fixed_soft` indirectly. Add this explicit test to Task 2 Step 1's `fixed_bp.rs` block:

```rust
    #[test]
    fn test_three_validity_apis_agree() {
        let dem = crate::BBCode::gross().code_capacity_dem(0.05);
        let fx = FixedRelayBp::new(&dem, 8, 3);
        let (syndromes, _truths) = crate::sample_shots(&dem, 300, 5);
        for syn in &syndromes {
            let a = fx.decode_fixed(syn).1;
            let b = fx.decode_fixed_ehat(syn).2;
            let c = fx.decode_fixed_soft(syn).converged;
            let d = fx.iters_to_valid(syn).0;
            assert_eq!((a, b, c), (b, c, d), "validity APIs disagree");
        }
    }
```

Note this test uses `FixedRelayBp::new` (the 4-leg default), not the 6×10 shipped budget — `iters_to_valid` is budget-independent in its contract, and using the default constructor keeps the test cheap.

**Placeholder scan.** No TBD/TODO. Every code step carries real code. Two steps are deliberately conditional on facts the implementer must check in-tree rather than guess: Task 2 Step 3's `Correction` field name, and Task 3 Step 1's `Syndrome` import. Both name the file to read and the adaptation to make.

**Type consistency.** `RefRecord { true_obs, sw_obs, valid, iters }` is written in Task 1 and read identically by the Python in Task 6 (`body[2::3] >> 15`). `with_residual_restricted(bool) -> Self` is defined in Task 2 and called in Task 4. `decode_stream_trace -> (Correction, StreamStats, Vec<WindowTrace>)` in Task 5 matches `crates/aleph-qec/src/relay_window.rs:660`. `mispredicted(&[bool], &[bool], usize)` takes an observable-flip slice throughout Tasks 3–5 (note it differs from `qec_q7_stream_sweep`'s version, which takes `&Correction`). `chunk_seed` is defined and used only in Task 3.
