# M9c gather-crossbar fix — Step 2 (Beneš permutation networks for sites 2–4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the three residual runtime-indexed crossbars/demuxes in `hw/bp_relay_banked_bram_m.sv` (site 2 e_cm read-gather, site 3 e_cm read-address scatter, site 4 m_cm write scatter) with **ROM-configured Beneš permutation networks**, so the banked relay-BP core fits and routes on the KV260 (`xck26`, 117 120 LUT) while staying **bit-exact and latency-exact** to `FixedRelayBp` in Verilator co-sim.

**Architecture:** Each of sites 2–4 maps a ≤1-access-per-bank-per-group var↔check correspondence — a **partial permutation**, not arbitrary fan-out. Today each is expressed as `array[rom_value]` (a ROM value indexing a `BP_N`-scale register array), which Vivado must synthesize as a full NEB/NHB-way crossbar (measured Step-1 residual: **CLB LUTs = 2,232,451 = 1906 %**, entirely these crossbars; the non-crossbar core fits at DSP 23 %/BRAM 22 %/FF 9 %). A Beneš network realises any permutation with **O(N·log N) 2×2 switches** whose control bits are ROM outputs feeding mux *selects* (the allowed pattern — never a register-array index). Per the design decisions taken 2026-07-14: **three site-specific fabrics** (each minimal for its N/payload), **pipelined proactively** with latency absorbed so the co-sim latency stays exact, and **all three sites landed together before one 8 h EPYC synth**.

**Tech Stack:** Rust (`aleph-qec` lib + `qec_q7_bp_graph` example emitter), SystemVerilog (Verilator 5 co-sim + Vivado 2024.2 OOC synth on EPYC).

**Design spec:** `docs/superpowers/specs/2026-07-13-m9c-gather-crossbar-fix-design.md` (§ "Step 2").
**Predecessor:** `docs/superpowers/plans/2026-07-13-m9c-gather-fix-step1.md` (Step 1 landed as `bd99812`; residual measured Jul 13 23:25 = 1906 % LUT ⇒ this plan).

## Global Constraints

- **Bit-exact:** 40/40 decision-equal vs `FixedRelayBp` at **both** bankings (8/24 and 16/48). Gate = `make -C hw bpbankedbramm`. No synth before Verilator is green.
- **Latency-exact:** worst-case latency printed by the co-sim (`worst latency = N cycles`) must stay **2206 @16/48** and **3871 @8/24** — unchanged from Step 1. Any Beneš pipeline stage MUST be absorbed by equal alignment delay on all co-launched paths (present/select/sbit/data) so the number does not move. This is a hard gate alongside bit-exactness.
- **No ROM-value-as-register-index anywhere in the new RTL.** Beneš control bits feed 2:1 mux *selects* only. The `rom_contract` elaboration guard must re-derive every network's routing at time-0 and `$fatal`/`fails++` on any mismatch (house `verify_banking` pattern).
- **Preserve `-flatten_hierarchy none` module boundaries** — the stamped cells are load-bearing for Vivado area-opt. New fabrics are their own modules, instantiated once per site.
- **Config (16/48 gross, the config that must fit):** `NEB = BP_BANK_W*BP_CHK_DEG = 16*25 = 400`; `NHB = 2*NEB = 800`; `NVB = BP_BANK_V*BP_VAR_DEG = 48*6 = 288`. Padded Beneš sizes: **site 2 & 3 → M=512**, **site 4 → M=1024**. Row-addr width `BWC = $clog2(BP_GC)`; message width `MSG_BITS`. The RTL must stay parameterized in these localparams (never hard-code 400/800/288 — the 8/24 banking uses smaller values and must also pass).
- **Bench host:** EPYC `root@195.154.249.85`, Vivado `/tools/Xilinx/Vivado/2024.2` (`source settings64.sh`), **serial `set_param general.maxThreads 1`**, 123 GB RAM + 128 GB `/data/swapfile`. One synth ≈ 8–9 h. Staging dir `/data/kv260fit/`, `ooc_serial.tcl` → RESULT line + `util_banked.rpt`.

---

## File Structure

| File | Responsibility |
|---|---|
| `crates/aleph-qec/src/benes.rs` (**create**) | Pure Beneš routing library: `benes_control(perm) -> Vec<bool>` (looping algorithm) + `benes_apply(ctrl, data) -> Vec<T>` (network simulator, used only by tests). Fully unit-tested against brute-force. No I/O, no graph knowledge. |
| `crates/aleph-qec/src/lib.rs` (**modify**) | `pub mod benes;` |
| `crates/aleph-qec/examples/qec_q7_bp_graph.rs` (**modify**) | In `print_rom_rows`: build the three per-group partial permutations from `Banking`, call `benes_control`, emit `BP_ROM_BENES_ECMRD/_ECMADDR/_MCMWR` packed-row ROMs. In `verify_banking`: re-route and assert control bits reproduce the matching. |
| `hw/bp_benes.sv` (**create**) | `bp_benes_switch` (payload-generic 2×2) + three fabrics `bp_benes_ecm_read` / `bp_benes_ecm_addr` / `bp_benes_mcm_wr`, each `#(N, W, PIPE)`, ROM-control-fed, pipelined. |
| `hw/tb_bp_benes.cpp` (**create**) | Standalone Verilator TB: random control + data through each fabric, assert bit-exact routing + latency == PIPE. |
| `hw/Makefile` (**modify**) | Add `bpbenes` target (module unit test) and add `bp_benes.sv` to the `bpbankedbramm`/`-lint` compile lists. |
| `hw/bp_relay_banked_bram_m.sv` (**modify**) | Replace sites 2/3/4 crossbars with the three fabric instances + leaf demuxes; add alignment registers; extend `rom_contract`. |
| `docs/perf/qec-q7-fixed-bp.md` (**modify**) | § M9c: record placed LUT/LUTRAM/BRAM/DSP/Fmax after synth + fit verdict. |

**Direction of each network** (source → destination; `s = i*BP_VAR_DEG+d` var-tap index, `b` = bank id):

| Site | Fabric | Inputs (N) | Outputs | Per-tap permutation `P` | Payload `W` | Leaf op |
|---|---|---|---|---|---|---|
| 2 e_cm read | `bp_benes_ecm_read` | NEB banks (pad 512) | NVB taps | tap `s` reads bank `var_ebsel[s]` ⇒ route bank→tap (inverse of site 3) | `2*MSG_BITS` (qa,qb pair) | `e_in = var_eport_r[s] ? qb : qa` |
| 3 e_cm addr | `bp_benes_ecm_addr` | NVB taps (pad 512) | NEB banks | tap `s` → bank `var_ebsel[s]` | `BWC + 1 + 1` (row, eport, valid) | at bank `b`: `valid ? (eport ? rb=row : ra=row) : ra=rb=0` |
| 4 m_cm wr | `bp_benes_mcm_wr` | NVB slots (pad 1024) | NHB half-banks | slot `s` → half-bank `scat_hb[s]` | `1 + BWC + MSG_BITS` (valid, row, data) | at hb `b`: `we=valid; wa=row; wd=data` |

Sites 2 and 3 share **one** matching (var reads exactly the banks it addressed): the emitter computes it once and emits the read-network control as the routing for `bank→tap`, the addr-network control for `tap→bank` (same matching, opposite direction). Site 4 is an independent matching (var-slot → m_cm half-bank).

> **⚠️ DESIGN UPDATE (post-Task-2, verified correct by review — SUPERSEDES the e_cm rows above for Tasks 3–5):**
> e_cm banks are **dual-port (a/b)**. Within a var-group up to `BANK_CAP=2` edges legitimately share a
> bank, disambiguated by `BP_EDGE_EPORT`. So sites 2 & 3 are **NOT a single permutation** — each is
> **two independent per-port size-512 Beneš networks** (port 0 = qa/ra, port 1 = qb/rb). Task 2 already
> emits the control accordingly: `BP_ROM_BENES_ECMADDR`/`_ECMRD` rows are **`BP_BENES_ECM_PORTS(=2) *
> BP_BENES_ECM_COLS(=17) * (BP_BENES_ECM_M(=512)/2)` = 8704 bits**, with **port 0 packed low, port 1
> packed high** (each port a full `ECM_COLS*(ECM_M/2)=4352`-bit control block). m_cm (site 4) is
> **unchanged — one size-1024 network** (`hb=eb*2+beta` already injective; `BP_ROM_BENES_MCMWR` = 9728 bits).
>
> **Consequences for Tasks 3–5:**
> - **Task 3:** the three fabric *module types* are unchanged and stay generic `#(N,W,PIPE)`; the e_cm
>   read/addr payloads become **per-port** (read `W=MSG_BITS`, addr `W=BWC+1` valid+row — the `eport`
>   bit is no longer carried, it's implicit in which network) not the `2*MSG_BITS`/`BWC+2` in the table.
> - **Task 4 (site 2 read):** instantiate `bp_benes_ecm_read` **twice** — `u_benes_rd0` fed
>   `qa_ecm`+`ctrl[4351:0]`, `u_benes_rd1` fed `qb_ecm`+`ctrl[8703:4352]`. Leaf:
>   `e_in_i[d] = var_eport_r[idx] ? rd1_dout[idx] : rd0_dout[idx]`.
> - **Task 5 (site 3 addr):** instantiate `bp_benes_ecm_addr` **twice** — port 0 → `ra_ecm`, port 1 →
>   `rb_ecm`, each fed its half of `BP_ROM_BENES_ECMADDR`. Site 4 stays one `bp_benes_mcm_wr`.
> - Slice the ROM halves with the emitted localparams (`BP_BENES_ECM_PORTS`, `BP_BENES_ECM_COLS`,
>   `BP_BENES_ECM_M`) — never hard-code 4352/8703.

---

### Task 1: Beneš routing library (`crates/aleph-qec/src/benes.rs`) — pure, TDD

**Files:**
- Create: `crates/aleph-qec/src/benes.rs`
- Modify: `crates/aleph-qec/src/lib.rs` (add `pub mod benes;`)
- Test: inline `#[cfg(test)] mod tests` in `benes.rs`

**Interfaces:**
- Consumes: nothing (pure).
- Produces:
  - `pub fn benes_control(perm: &[usize]) -> Vec<bool>` — `perm` is a **full bijection** on `0..M` (`M = perm.len()`, a power of two), `perm[i]` = output index that input `i` routes to. Returns the switch control bits in **column-major, top-to-bottom** order: `2*log2(M)-1` columns × `M/2` switches, `ctrl[col*(M/2)+sw]`, `false` = bar (straight), `true` = cross.
  - `pub fn benes_columns(m: usize) -> usize` — `2*log2(m)-1`.
  - `pub fn benes_apply(ctrl: &[bool], input: &[usize]) -> Vec<usize>` — network simulator (test/guard oracle): applies the control bits to `input` (length M) and returns the routed output. Used by the emitter's guard and the tests; NOT on the RTL path.
  - `pub fn complete_partial(dest: &[Option<usize>], m: usize) -> Vec<usize>` — extend a partial injection (`dest[i]=Some(b)` or `None` for padding) on `0..M` outputs to a full bijection by filling unused inputs with the unused outputs in ascending order. Panics if `dest` has a duplicate target (violates the ≤1-per-bank invariant).

- [ ] **Step 1: Write the failing test (network simulator + round-trip on known perms)**

Create `crates/aleph-qec/src/benes.rs` with only the test module + `unimplemented!()` stubs so it compiles-and-fails:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    // Apply control bits, assert the network realises exactly `perm`.
    fn check(perm: &[usize]) {
        let ctrl = benes_control(perm);
        assert_eq!(ctrl.len(), benes_columns(perm.len()) * (perm.len() / 2));
        let ident: Vec<usize> = (0..perm.len()).collect();
        let routed = benes_apply(&ctrl, &ident);
        // input i must land at output perm[i]  =>  routed[perm[i]] == i
        let mut got = vec![usize::MAX; perm.len()];
        for (i, &o) in perm.iter().enumerate() {
            got[o] = routed[o]; // routed[o] is the input delivered to output o
        }
        for (i, &o) in perm.iter().enumerate() {
            assert_eq!(routed[o], i, "perm {:?}: output {} got input {} not {}", perm, o, routed[o], i);
        }
        let _ = got;
    }

    #[test] fn identity_8()  { check(&[0,1,2,3,4,5,6,7]); }
    #[test] fn reverse_8()   { check(&[7,6,5,4,3,2,1,0]); }
    #[test] fn swap_pairs_8(){ check(&[1,0,3,2,5,4,7,6]); }

    #[test]
    fn random_bijections() {
        // deterministic LCG (no dev-dep); cover the production sizes.
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut rng = || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; state };
        for &m in &[2usize, 4, 8, 16, 512, 1024] {
            for _ in 0..64 {
                let mut p: Vec<usize> = (0..m).collect();
                for i in (1..m).rev() { let j = (rng() as usize) % (i + 1); p.swap(i, j); }
                check(&p);
            }
        }
    }

    #[test]
    fn partial_completes_and_routes() {
        // 3 real sources into 8 outputs (banks): s0->5, s1->2, s2->6
        let dest = vec![Some(5), Some(2), Some(6), None, None, None, None, None];
        let full = complete_partial(&dest, 8);
        assert_eq!(full[0], 5); assert_eq!(full[1], 2); assert_eq!(full[2], 6);
        assert_eq!(full.iter().copied().collect::<std::collections::BTreeSet<_>>().len(), 8);
        check(&full);
    }

    #[test]
    #[should_panic]
    fn duplicate_target_rejected() {
        complete_partial(&[Some(3), Some(3), None, None], 4);
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p aleph-qec --lib benes`
Expected: FAIL — `unimplemented!()` panics (or does-not-compile until stubs exist). Add `pub fn` stubs returning `unimplemented!()` so it compiles and fails at runtime, then re-run to confirm FAIL.

- [ ] **Step 3: Implement the routing (Beneš looping algorithm)**

Prepend to `benes.rs` (before the test module). Algorithm: Beneš 1965 / Lee's *looping algorithm* (K. Y. Lee, "On the Rearrangeability of 2(log2 N)−1 Stage Permutation Networks", IEEE ToC 1985). Recurse: input column pairs `(2i,2i+1)`, output column pairs `(2j,2j+1)`; assign each edge to the upper or lower `B(M/2)`, following forced loops; recurse on the two half-permutations; stitch the three control slices column-major.
```rust
//! Beneš rearrangeable permutation network routing.
//!
//! A Beneš network on M = 2^k inputs realises any permutation with 2·log2(M)−1
//! columns of M/2 2×2 switches. `benes_control` returns the switch settings for a
//! given bijection; `benes_apply` simulates the fabric (test / gen-time guard oracle).
//! Ref: Beneš (1965); Lee, IEEE ToC C-34(5), 1985 (looping algorithm).

pub fn benes_columns(m: usize) -> usize {
    assert!(m.is_power_of_two() && m >= 2);
    2 * (m.trailing_zeros() as usize) - 1
}

pub fn complete_partial(dest: &[Option<usize>], m: usize) -> Vec<usize> {
    assert!(m.is_power_of_two());
    assert_eq!(dest.len(), m);
    let mut used = vec![false; m];
    let mut out = vec![usize::MAX; m];
    for (i, d) in dest.iter().enumerate() {
        if let Some(b) = *d {
            assert!(b < m, "target {b} out of range {m}");
            assert!(!used[b], "duplicate Beneš target {b} (violates <=1-per-bank invariant)");
            used[b] = true;
            out[i] = b;
        }
    }
    let mut free: Vec<usize> = (0..m).filter(|&b| !used[b]).collect();
    for slot in out.iter_mut() {
        if *slot == usize::MAX {
            *slot = free.pop().expect("free list underflow: dest not an injection");
        }
    }
    out
}

/// Control bits, column-major (`ctrl[col*(m/2) + switch]`), false = bar, true = cross.
pub fn benes_control(perm: &[usize]) -> Vec<bool> {
    let m = perm.len();
    assert!(m.is_power_of_two() && m >= 2, "Beneš size must be a power of two >= 2");
    let cols = benes_columns(m);
    let mut ctrl = vec![false; cols * (m / 2)];
    route(perm, &mut ctrl, 0, m, 0, cols);
    ctrl
}

// Recursive router. `perm` is a bijection on 0..n (n = 2^t). `col0` = index of this
// block's INPUT column in the global `ctrl`; `ncols` = 2·log2(n)−1 columns owned here.
// The global fabric has `stride = M/2` switches per column; sub-blocks occupy a
// contiguous half-range of switch rows (`row0`).
fn route(perm: &[usize], ctrl: &mut [bool], col0: usize, n: usize, row0: usize, gcols: usize) {
    let stride = /* switches per global column */ ctrl.len() / gcols;
    if n == 2 {
        // single column: bar if perm==[0,1], cross if [1,0]
        ctrl[col0 * stride + row0] = perm[0] == 1;
        return;
    }
    let half = n / 2;
    let inv = {
        let mut q = vec![0usize; n];
        for (i, &o) in perm.iter().enumerate() { q[o] = i; }
        q
    };
    // subnet assignment via the looping algorithm
    let mut in_set = vec![Option::<bool>::None; half];  // input switch i -> upper(false)/lower(true) for input 2i
    let mut out_set = vec![Option::<bool>::None; half];
    // process each output switch; start an unassigned one on the upper subnet
    let mut start = 0;
    while start < half {
        if out_set[start].is_some() { start += 1; continue; }
        let mut o = 2 * start;              // an output terminal
        let mut go_upper = false;           // this terminal -> upper
        loop {
            let osw = o / 2;
            out_set[osw] = Some(if o % 2 == 0 { go_upper } else { !go_upper });
            let i = inv[o];                  // input feeding output o
            let isw = i / 2;
            let i_upper = go_upper;          // same subnet as the output it feeds
            in_set[isw] = Some(if i % 2 == 0 { i_upper } else { !i_upper });
            let i_partner = if i % 2 == 0 { i + 1 } else { i - 1 };
            let o2 = perm[i_partner];        // partner input must take the OTHER subnet
            let o2sw = o2 / 2;
            if out_set[o2sw].is_some() { break; }
            o = o2;
            go_upper = !i_upper;             // partner output -> other subnet
        }
        start += 1;
    }
    // build the two half-permutations (upper handles subnet=false, lower=true)
    let mut up = vec![0usize; half];
    let mut lo = vec![0usize; half];
    for isw in 0..half {
        let iu = in_set[isw].unwrap();
        // input 2*isw goes to subnet iu, 2*isw+1 to !iu
        for (bit, inp) in [(iu, 2 * isw), (!iu, 2 * isw + 1)] {
            let o = perm[inp];
            let osw = o / 2;
            let ou = out_set[osw].unwrap();
            debug_assert_eq!(ou == false, bit == false, "subnet parity mismatch");
            let sub_in = isw;
            let sub_out = osw;
            if !bit { up[sub_in] = sub_out; } else { lo[sub_in] = sub_out; }
        }
    }
    // write THIS block's input + output columns
    for isw in 0..half {
        ctrl[col0 * stride + row0 + isw] = in_set[isw].unwrap();          // input column
        ctrl[(col0 + gcols_block(n) - 1) * stride + row0 + isw] = out_set[isw].unwrap(); // output column
    }
    // recurse: upper subnet occupies rows [row0 .. row0+half/2), lower the next half/2,
    // both starting one column in and spanning the inner (gcols_block(n)-2) columns.
    route(&up, ctrl, col0 + 1, half, row0, gcols);
    route(&lo, ctrl, col0 + 1, half, row0 + half / 2, gcols);
}

fn gcols_block(n: usize) -> usize { 2 * (n.trailing_zeros() as usize) - 1 }

/// Simulate the fabric with `ctrl` on `input` (length M). Returns output[o] = input routed to o.
pub fn benes_apply(ctrl: &[bool], input: &[usize]) -> Vec<usize> {
    let m = input.len();
    let cols = benes_columns(m);
    let stride = m / 2;
    let mut buf = input.to_vec();
    for c in 0..cols {
        let mut next = buf.clone();
        for sw in 0..stride {
            let (a, b) = switch_wires(c, sw, m); // the two wire indices this switch touches
            if ctrl[c * stride + sw] { next[a] = buf[b]; next[b] = buf[a]; }
            else                     { next[a] = buf[a]; next[b] = buf[b]; }
        }
        buf = next;
    }
    buf
}
```
Also implement `switch_wires(col, sw, m)`, the butterfly wiring for a Beneš (outer columns = Beneš shuffle, inner columns recurse). Derive it from the same recursion so `benes_apply` and `route` agree; the round-trip test (Step 1) is what proves they agree — if `switch_wires` and `route` disagree, `random_bijections` fails. Keep iterating `switch_wires`/`route` until the test is green; **do not proceed while any case fails**.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p aleph-qec --lib benes`
Expected: PASS — all of `identity_8`, `reverse_8`, `swap_pairs_8`, `random_bijections` (M up to 1024), `partial_completes_and_routes`, `duplicate_target_rejected`.

- [ ] **Step 5: Clippy + fmt clean**

Run: `cargo clippy -p aleph-qec --all-targets -- -D warnings && cargo fmt --check`
Expected: exit 0.

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-qec/src/benes.rs crates/aleph-qec/src/lib.rs
git commit -m "[Q7-04] M9c step 2.1: Beneš routing library (looping algorithm, TDD)

Pure benes_control/benes_apply/complete_partial for the ROM-configured
permutation networks replacing sites 2-4 crossbars. Round-trip tested vs
brute-force fabric simulation for M up to 1024 incl. partial injections.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HNwUEvqsNoAqv1ykqHHNHn"
```

---

### Task 2: Emitter — compute matchings, emit control ROMs, guard (`qec_q7_bp_graph.rs`)

**Files:**
- Modify: `crates/aleph-qec/examples/qec_q7_bp_graph.rs` — `print_rom_rows` (~L1202) + `verify_banking` (~L869)
- Test (regression): `cargo run --release --example qec_q7_bp_graph -- circgraph 1 0.003 16 48` emits without panic; the guard in `verify_banking` runs at emit time.

**Interfaces:**
- Consumes: `Banking { edge_eb: Vec<usize>, edge_hb: Vec<usize>, ... }`, `FixedHwView`, the per-group presence already computed for `var_ebsel`/`scat_hb`/`var_eport`, `aleph_qec::benes::{benes_control, complete_partial, benes_columns}`.
- Produces three new packed-row ROM tables (house `emit_rom_table` + `RomRow`):
  - `BP_ROM_BENES_ECMRD` — depth `BP_GV`, width `benes_columns(NEB_POW2)*(NEB_POW2/2)` bits/group (bank→tap control).
  - `BP_ROM_BENES_ECMADDR` — depth `BP_GV`, same width (tap→bank control).
  - `BP_ROM_BENES_MCMWR` — depth `BP_GV`, width `benes_columns(NHB_POW2)*(NHB_POW2/2)` bits/group.
  - Plus localparams `BP_BENES_ECM_M` (= NEB padded to pow2), `BP_BENES_MCM_M` (= NHB padded), `BP_BENES_ECM_COLS`, `BP_BENES_MCM_COLS` printed alongside the existing config localparams (~L434).

- [ ] **Step 1: Write the failing guard test (route round-trips the matching at emit time)**

In `verify_banking`, after the existing `edge_eb`/`edge_hb` asserts (~L1006), add per-group re-routing asserts:
```rust
// M9c Step 2: the Beneš control we will emit must realise exactly the per-group matching.
use aleph_qec::benes::{benes_control, benes_apply, complete_partial};
for g in 0..gv {
    // tap s = i*var_deg + d ; site 3/2 matching: tap present? -> bank edge_eb[e]
    let mut dest_ecm = vec![None; ecm_m]; // ecm_m = NEB padded to pow2
    for (s, e) in group_taps(g) {          // (tap index, edge id) present this group
        dest_ecm[s] = Some(b.edge_eb[e]);
    }
    let full = complete_partial(&dest_ecm, ecm_m);
    let ctrl_addr = benes_control(&full);              // tap -> bank
    let routed = benes_apply(&ctrl_addr, &(0..ecm_m).collect::<Vec<_>>());
    for (s, e) in group_taps(g) {
        assert_eq!(routed[b.edge_eb[e]], s, "ECM addr route g={g} s={s} bank={}", b.edge_eb[e]);
    }
    // read network is the same matching inverted; emitter derives ctrl_read from inv(full)
}
```
(Reuse the existing group/tap iteration already present in `print_rom_rows`; factor a small `group_taps(g)` helper if not already available.)

- [ ] **Step 2: Run to verify it fails**

Run: `cargo run --release --example qec_q7_bp_graph -- circgraph 1 0.003 16 48 > /dev/null`
Expected: FAIL — `ecm_m`/`group_taps` undefined (compile error) or the derive-read path not yet written. Confirms the guard is wired.

- [ ] **Step 3: Implement the padded sizes + matchings + control emission in `print_rom_rows`**

Add near the existing `var_ebsel`/`scat_hb` build (~L1265-1321):
```rust
let ecm_m = (neb as usize).next_power_of_two();   // 400 -> 512
let mcm_m = (nhb as usize).next_power_of_two();    // 800 -> 1024
let ecm_cols = benes_columns(ecm_m);
let mcm_cols = benes_columns(mcm_m);
let mut benes_ecmrd  = Vec::with_capacity(gv);
let mut benes_ecmaddr = Vec::with_capacity(gv);
let mut benes_mcmwr  = Vec::with_capacity(gv);
for g in 0..gv {
    // --- e_cm matching: tap s -> bank edge_eb[e] (partial injection, <=1 per bank/group)
    let mut dest_ecm = vec![None; ecm_m];
    let mut dest_mcm = vec![None; mcm_m];
    for (s, e) in group_taps(g) {
        dest_ecm[s] = Some(b.edge_eb[e] as usize);
        dest_mcm[s] = Some(b.edge_hb[e] as usize);   // site 4 half-bank
    }
    let full_ecm = complete_partial(&dest_ecm, ecm_m);
    let full_mcm = complete_partial(&dest_mcm, mcm_m);
    // addr scatter (tap->bank) and read gather (bank->tap = inverse)
    let mut inv_ecm = vec![0usize; ecm_m];
    for (i, &o) in full_ecm.iter().enumerate() { inv_ecm[o] = i; }
    let c_addr = benes_control(&full_ecm);
    let c_read = benes_control(&inv_ecm);
    let c_mcm  = benes_control(&full_mcm);
    benes_ecmaddr.push(pack_bits(&c_addr));
    benes_ecmrd.push(pack_bits(&c_read));
    benes_mcmwr.push(pack_bits(&c_mcm));
}
```
Add a `pack_bits(&[bool]) -> RomRow` helper (LSB-first, mirrors `RomRow::set` 1-bit fields — see `print_rom_rows`'s existing 1-bit fills). Then emit (near L1336):
```rust
emit_rom_table("BP_ROM_BENES_ECMRD",  "BP_GV", &benes_ecmrd);
emit_rom_table("BP_ROM_BENES_ECMADDR","BP_GV", &benes_ecmaddr);
emit_rom_table("BP_ROM_BENES_MCMWR",  "BP_GV", &benes_mcmwr);
```
And print the new localparams alongside the config block (~L434):
```rust
println!("localparam int BP_BENES_ECM_M    = {ecm_m};");
println!("localparam int BP_BENES_MCM_M    = {mcm_m};");
println!("localparam int BP_BENES_ECM_COLS = {ecm_cols};");
println!("localparam int BP_BENES_MCM_COLS = {mcm_cols};");
```

- [ ] **Step 4: Run — emit succeeds, guard passes at both bankings**

Run:
```bash
cargo run --release --example qec_q7_bp_graph -- circgraph 1 0.003 16 48 | grep -c "BP_ROM_BENES"
cargo run --release --example qec_q7_bp_graph -- circgraph 1 0.003 8 24  > /dev/null && echo "8/24 OK"
```
Expected: `3` (three tables emitted), `8/24 OK`, no panic from `verify_banking` (the emit-time guard proves control ⇒ matching for every group at both bankings).

- [ ] **Step 5: Clippy + fmt**

Run: `cargo clippy -p aleph-qec --all-targets -- -D warnings && cargo fmt --check`
Expected: exit 0.

- [ ] **Step 6: Commit**

```bash
git add crates/aleph-qec/examples/qec_q7_bp_graph.rs
git commit -m "[Q7-04] M9c step 2.2: emit Beneš control ROMs + gen-time route guard

print_rom_rows computes the per-group tap<->bank (sites 2/3) and tap->half-bank
(site 4) matchings, routes them via benes_control, emits BP_ROM_BENES_ECMRD/
ECMADDR/MCMWR packed-row ROMs. verify_banking re-routes and asserts the control
reproduces the matching for every group at both bankings.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HNwUEvqsNoAqv1ykqHHNHn"
```

---

### Task 3: SystemVerilog Beneš fabrics + standalone unit test (`hw/bp_benes.sv`, `tb_bp_benes.cpp`)

> **⚠️ Read the "DESIGN UPDATE (post-Task-2)" note above first.** The three fabric module *types* below are
> correct and generic; only the e_cm read/addr **payload widths** change to per-port (read `W=MSG_BITS`,
> addr `W=BWC+1`). The fabric modules themselves are unaffected — they stay `#(N,W,PIPE)` generic.

**Files:**
- Create: `hw/bp_benes.sv` — `bp_benes_switch` + `bp_benes_ecm_read` + `bp_benes_ecm_addr` + `bp_benes_mcm_wr`
- Create: `hw/tb_bp_benes.cpp`
- Modify: `hw/Makefile` — add `bpbenes` target

**Interfaces:**
- `bp_benes_switch #(parameter int W)( input logic sel, input logic [W-1:0] a_in, b_in, output logic [W-1:0] a_out, b_out )` — `sel` ? crossed : straight. Pure comb.
- Each fabric: `#(parameter int N, W, PIPE)( input clk, input logic [N-1:0][W-1:0] din, input logic [COLS*(N/2)-1:0] ctrl, output logic [N-1:0][W-1:0] dout )` where `COLS = 2*$clog2(N)-1`. `din[i]` = payload on input wire `i`; `ctrl` = flattened column-major control (matches `benes_control`); `dout[o]` = payload delivered to output wire `o`. `PIPE` registers spread evenly across columns; latency = `PIPE` cycles. The butterfly wiring per column MUST match `benes_apply`/`switch_wires` from Task 1 (that identity is what the TB checks).

- [ ] **Step 1: Write the failing standalone TB (random control+data, bit-exact route + latency)**

Create `hw/tb_bp_benes.cpp`. It (a) reimplements `benes_apply` in C++ (same wiring as Task 1's `switch_wires`), (b) drives random `ctrl` + random `din`, (c) waits `PIPE` cycles, (d) asserts `dout` equals the C++ reference for each of the three fabrics. Structure mirrors `hw/tb_var_update.cpp` (generic module, TB reimplements the reference, ≥10000 random cases). Build one top per fabric via `--top-module`. Include a latency assert: first correct output appears exactly `PIPE` cycles after `din`/`ctrl` applied.

- [ ] **Step 2: Add the Make target and run to see it fail**

In `hw/Makefile`, add to `.PHONY` and define:
```make
bpbenes:
	$(VERILATOR) --cc --exe --build -Wall --Mdir obj_benes_read --top-module bp_benes_ecm_read \
		-GN=512 -GW=32 -GPIPE=3 bp_benes.sv tb_bp_benes.cpp -o sim_benes_read
	./obj_benes_read/sim_benes_read
	$(VERILATOR) --cc --exe --build -Wall --Mdir obj_benes_addr --top-module bp_benes_ecm_addr \
		-GN=512 -GW=10 -GPIPE=3 bp_benes.sv tb_bp_benes.cpp -o sim_benes_addr
	./obj_benes_addr/sim_benes_addr
	$(VERILATOR) --cc --exe --build -Wall --Mdir obj_benes_wr --top-module bp_benes_mcm_wr \
		-GN=1024 -GW=24 -GPIPE=4 bp_benes.sv tb_bp_benes.cpp -o sim_benes_wr
	./obj_benes_wr/sim_benes_wr
```
Run: `make -C hw bpbenes`
Expected: FAIL — `bp_benes.sv` not yet created (compile error).

- [ ] **Step 3: Implement `bp_benes.sv`**

Write the switch primitive and the three fabrics. Each fabric is a `generate` over `COLS` columns × `N/2` switches with the Beneš butterfly wiring (outer columns = Beneš shuffle permutation, inner columns recurse — the same wiring `switch_wires` encodes in Task 1). Insert a pipeline register after column `floor(c * PIPE / COLS)` boundaries so exactly `PIPE` registers sit on every path. Keep the three fabrics separate modules (site-specific, per the design decision) but each `include`s the shared `bp_benes_switch`. Iterate wiring until Step 4 is green.

- [ ] **Step 4: Run the unit test — bit-exact routing + latency == PIPE**

Run: `make -C hw bpbenes`
Expected: all three sims print PASS (≥10000 random cases each, `dout` == C++ reference, first-correct-output latency == PIPE). **Do not proceed while any fabric mis-routes** — this is the module-level correctness proof before the fabric touches the core.

- [ ] **Step 5: Lint**

Run: `$(VERILATOR) --lint-only -Wall --top-module bp_benes_mcm_wr -GN=1024 -GW=24 -GPIPE=4 hw/bp_benes.sv` (and the other two tops)
Expected: exit 0, no warnings.

- [ ] **Step 6: Commit**

```bash
git add hw/bp_benes.sv hw/tb_bp_benes.cpp hw/Makefile
git commit -m "[Q7-04] M9c step 2.3: parameterized Beneš fabrics + standalone unit test

bp_benes_switch + three site-specific fabrics (ecm_read/ecm_addr/mcm_wr),
#(N,W,PIPE), ROM-control-fed, evenly pipelined. tb_bp_benes drives 10k random
control+data cases per fabric; routing bit-exact vs C++ benes_apply, latency==PIPE.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HNwUEvqsNoAqv1ykqHHNHn"
```

---

### Task 4: Wire site 2 (e_cm read-gather) into the core + pipeline-align — co-sim gate

> **⚠️ Read the "DESIGN UPDATE (post-Task-2)" note above first — it SUPERSEDES the single-instance wiring in
> this task.** Instantiate `bp_benes_ecm_read` **twice**: `u_benes_rd0` fed `qa_ecm` + the low ROM half
> (`BP_ROM_BENES_ECMRD[g]` bits `[ECM_COLS*(ECM_M/2)-1 : 0]`), `u_benes_rd1` fed `qb_ecm` + the high half.
> Each carries `W=MSG_BITS` (NOT the `2*MSG_BITS` pair). Leaf: `e_in_i[d] = var_eport_r[idx] ? rd1_dout[idx]
> : rd0_dout[idx]`. Slice ROM halves via `BP_BENES_ECM_PORTS`/`BP_BENES_ECM_COLS`/`BP_BENES_ECM_M`.

**Files:**
- Modify: `hw/bp_relay_banked_bram_m.sv` — site 2 read (~L884-898), instantiate `bp_benes_ecm_read`, add alignment registers, `rom_contract` (already re-derived at emit time; add the fabric's ROM wiring)
- Modify: `hw/Makefile` — add `bp_benes.sv` to `bpbankedbramm` + `bpbankedbramm-lint` compile lists

**Interfaces:**
- Consumes: `qa_ecm[NEB]`, `qb_ecm[NEB]` (bank read outputs), `BP_ROM_BENES_ECMRD` (per-group control), `var_eport_r`, `var_epres_r`.
- Produces: `e_in_i[d]` per var-tap — **same value/type as today**, delivered via the fabric instead of `qb_ecm[var_ebsel_r[idx]]`.

- [ ] **Step 1: Capture green baseline + baseline latency**

Run: `make -C hw bpbankedbramm 2>&1 | tee /tmp/base.txt; grep -E "== W=|40/40|worst latency" /tmp/base.txt`
Expected: both bankings 40/40; `worst latency = 2206` (16/48) and `3871` (8/24). Record — these are the exact gates.

- [ ] **Step 2: Instantiate the read fabric + leaf select (replace site 2)**

Pack `{qa_ecm, qb_ecm}` (padded to `BP_BENES_ECM_M` with zeros) into `din`, wire `BP_ROM_BENES_ECMRD[group]` → `ctrl`, take `dout[s]` for tap `s`, split the pair, and select by `var_eport_r`:
```systemverilog
  // M9c site 2: e_cm read-gather via ROM-configured Beneš (replaces qb_ecm[var_ebsel_r[idx]] crossbar)
  logic [BP_BENES_ECM_M-1:0][2*MSG_BITS-1:0] ecm_rd_din, ecm_rd_dout;
  always_comb for (int b = 0; b < BP_BENES_ECM_M; b++)
      ecm_rd_din[b] = (b < NEB) ? {qb_ecm[b], qa_ecm[b]} : '0;
  bp_benes_ecm_read #(.N(BP_BENES_ECM_M), .W(2*MSG_BITS), .PIPE(BENES_PIPE_ECM))
      u_benes_rd (.clk(clk), .din(ecm_rd_din), .ctrl(benes_ecmrd_r), .dout(ecm_rd_dout));
```
In the var gather, replace the site-2 read line (~L890):
```systemverilog
            e_in_i[d] = var_eport_r[idx] ? ecm_rd_dout[idx][2*MSG_BITS-1 -: MSG_BITS]
                                         : ecm_rd_dout[idx][MSG_BITS-1:0];
```
Add `benes_ecmrd_r` = registered `BP_ROM_BENES_ECMRD[read-group]` twin (mirrors the existing `*_r` ROM registers). Set `BENES_PIPE_ECM` (localparam, start = 3) and add **`BENES_PIPE_ECM`-deep alignment registers** on the co-launched `present_i`/`m_in_i`/`var_eport_r`/`sbit` paths so every operand of `var_update` arrives on the same cycle (Step 4 latency gate confirms).

- [ ] **Step 3: Add `bp_benes.sv` to the co-sim compile lists**

In `hw/Makefile`, both `bpbankedbramm` and `bpbankedbramm-lint`: add `bp_benes.sv` before `bp_relay_banked_bram_m.sv` in the source list.

- [ ] **Step 4: Run co-sim — 40/40 both bankings + latency EXACTLY unchanged**

Run: `make -C hw bpbankedbramm 2>&1 | grep -E "== W=|40/40|worst latency|FAIL"`
Expected: both bankings 40/40; `worst latency = 2206` (16/48) and `3871` (8/24) — **identical to Step 1**. If latency moved, adjust the alignment-register depth on the sibling paths (not the fabric PIPE) until it matches; if decisions differ, the pair packing/`eport` select or the `benes_ecmrd` direction (read = inverse) is wrong — fix and re-run. Minutes per iteration.

- [ ] **Step 5: Lint**

Run: `make -C hw bpbankedbramm-lint`
Expected: exit 0.

- [ ] **Step 6: Commit**

```bash
git add hw/bp_relay_banked_bram_m.sv hw/Makefile
git commit -m "[Q7-04] M9c step 2.4: site 2 e_cm read-gather via Beneš (kill 400-way crossbar)

qb_ecm[var_ebsel_r[idx]] -> bp_benes_ecm_read fabric fed by BP_ROM_BENES_ECMRD;
leaf eport-selects the (qa,qb) pair. PIPE absorbed by sibling alignment regs:
40/40 both bankings, latency 2206/3871 unchanged.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HNwUEvqsNoAqv1ykqHHNHn"
```

---

### Task 5: Wire sites 3 & 4 (scatters) into the core + pipeline-align — co-sim gate

> **⚠️ Read the "DESIGN UPDATE (post-Task-2)" note above first — it SUPERSEDES the single-instance site-3
> wiring.** Instantiate `bp_benes_ecm_addr` **twice**: port 0 (low ROM half of `BP_ROM_BENES_ECMADDR[g]`)
> drives `ra_ecm`, port 1 (high half) drives `rb_ecm`. Each carries `W=BWC+1` (valid+row; the `eport` bit
> is implicit in which network). Site 4 (`bp_benes_mcm_wr`) is UNCHANGED — one size-1024 network,
> `W=1+BWC+MSG_BITS`. Slice ROM halves via `BP_BENES_ECM_PORTS`/`BP_BENES_ECM_COLS`/`BP_BENES_ECM_M`.

**Files:**
- Modify: `hw/bp_relay_banked_bram_m.sv` — site 3 (e_cm read-addr scatter, ~L748-764) + site 4 (m_cm write scatter, ~L725-746)

**Interfaces:**
- Consumes: `var_erow_r`, `var_eport_r`, `var_epres_r` (site 3); `scat_row_r`, `scat_pres_r`, `scat_lam_r`, `var_m_out` (site 4); `BP_ROM_BENES_ECMADDR`, `BP_ROM_BENES_MCMWR`.
- Produces: `ra_ecm[b]`/`rb_ecm[b]` (site 3), `we_mcm[b]`/`wa_mcm[b]`/`wd_mcm[b]` (site 4) — **same values as today**, delivered via fabrics instead of `array[rom_value] = ...` scatters.

- [ ] **Step 1: Replace site 3 (e_cm addr scatter) with `bp_benes_ecm_addr`**

Feed per-tap `{valid=var_epres_r[s], eport=var_eport_r[s], row=var_erow_r[s]}` (padded to `BP_BENES_ECM_M`) into the addr fabric; at output bank `b` demux:
```systemverilog
  logic [BP_BENES_ECM_M-1:0][BWC+2-1:0] ecm_ad_din, ecm_ad_dout;
  always_comb for (int s = 0; s < BP_BENES_ECM_M; s++)
      ecm_ad_din[s] = (s < NVB) ? {var_epres_r[s], var_eport_r[s], var_erow_r[s]} : '0;
  bp_benes_ecm_addr #(.N(BP_BENES_ECM_M), .W(BWC+2), .PIPE(BENES_PIPE_ECM))
      u_benes_ad (.clk(clk), .din(ecm_ad_din), .ctrl(benes_ecmaddr_r), .dout(ecm_ad_dout));
  always_comb for (int b = 0; b < NEB; b++) begin
      ra_ecm[b] = '0; rb_ecm[b] = '0;
      if (state_pipe_valid && ecm_ad_dout[b][BWC+1]) begin           // valid bit
          if (ecm_ad_dout[b][BWC]) rb_ecm[b] = ecm_ad_dout[b][BWC-1:0];
          else                     ra_ecm[b] = ecm_ad_dout[b][BWC-1:0];
      end
  end
```
(`state_pipe_valid` = the `state==S_VAR` gate delayed by `BENES_PIPE_ECM` to match the fabric latency.)

- [ ] **Step 2: Replace site 4 (m_cm write scatter) with `bp_benes_mcm_wr`**

Feed per-tap `{valid=scat_pres_r[s], row=scat_row_r[s], data=(scat_is_init?scat_lam_r[s]:var_m_flat[s])}` (padded to `BP_BENES_MCM_M`) into the wr fabric; at output half-bank `b`:
```systemverilog
  logic [BP_BENES_MCM_M-1:0][1+BWC+MSG_BITS-1:0] mcm_wr_din, mcm_wr_dout;
  always_comb for (int s = 0; s < BP_BENES_MCM_M; s++)
      mcm_wr_din[s] = (s < NVB)
          ? {scat_pres_r[s], scat_row_r[s], (scat_is_init ? signed'(scat_lam_r[s]) : var_m_flat[s])}
          : '0;
  bp_benes_mcm_wr #(.N(BP_BENES_MCM_M), .W(1+BWC+MSG_BITS), .PIPE(BENES_PIPE_MCM))
      u_benes_wr (.clk(clk), .din(mcm_wr_din), .ctrl(benes_mcmwr_r), .dout(mcm_wr_dout));
  always_comb for (int b = 0; b < NHB; b++) begin
      we_mcm[b] = mcm_we_gate_d && mcm_wr_dout[b][1+BWC+MSG_BITS-1];   // valid, gate delayed by PIPE
      wa_mcm[b] = mcm_wr_dout[b][MSG_BITS +: BWC];
      wd_mcm[b] = signed'(mcm_wr_dout[b][MSG_BITS-1:0]);
  end
```
Add `benes_ecmaddr_r`/`benes_mcmwr_r` registered ROM twins; `BENES_PIPE_MCM` localparam (start = 4); delay `mcm_we_gate`/`scat_is_init` by the matching PIPE (`_d`). **Note `var_m_flat`/`var_m_out` timing:** site 4 previously read `var_m_out[i][d]` combinationally in the same cycle as the write; the fabric adds PIPE latency, so the write now lands PIPE cycles later — the m_cm write-address (`mcm_ra_r`) read side and the write-enable schedule must be consistent. Verify via the latency gate; if the write/read ordering breaks decisions, delay the *write group* consistently (the co-sim catches this precisely).

- [ ] **Step 3: Run co-sim — 40/40 both bankings + latency exactly unchanged**

Run: `make -C hw bpbankedbramm 2>&1 | grep -E "== W=|40/40|worst latency|FAIL"`
Expected: both bankings 40/40; `worst latency = 2206` / `3871`. Iterate the PIPE-alignment (`_d` gates, write-group delay) until both bit-exact AND latency-exact. Minutes per iteration — this is where the "absorb latency" work converges.

- [ ] **Step 4: Streaming wrapper sanity (the wrapped core changed)**

Run: `make -C hw bpstream 2>&1 | grep -iE "PASS|FAIL|latency" | head`
Expected: green if `bpstream` wraps `_m`; if it wraps the flat core (per Step-1 plan Task 1 Step 6), it is unaffected — record which.

- [ ] **Step 5: Lint + full workspace test**

Run: `make -C hw bpbankedbramm-lint && cargo test -p aleph-qec`
Expected: lint exit 0; `aleph-qec` tests (incl. `benes`) pass.

- [ ] **Step 6: Commit**

```bash
git add hw/bp_relay_banked_bram_m.sv
git commit -m "[Q7-04] M9c step 2.5: sites 3+4 e_cm addr + m_cm write scatters via Beneš

ra/rb_ecm[var_ebsel_r[idx]] and we/wa/wd_mcm[scat_hb_r[idx]] demuxes replaced by
bp_benes_ecm_addr / bp_benes_mcm_wr fabrics (BP_ROM_BENES_ECMADDR/MCMWR). PIPE
absorbed by delayed gates: 40/40 both bankings, latency 2206/3871 unchanged. All
three runtime crossbars now eliminated.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HNwUEvqsNoAqv1ykqHHNHn"
```

---

### Task 6: EPYC synth (all 3 sites) + fit verdict + doc

**Files:**
- Uses: `hw/bp_relay_banked_bram_m.sv`, `hw/bp_benes.sv`, `hw/check_minsum.sv`, `hw/var_update.sv`, generated `hw/bb_gross_tanner.svh` (16/48)
- Modify: `docs/perf/qec-q7-fixed-bp.md` § M9c

**Interfaces:**
- Consumes: the committed Task-5 core.
- Produces: placed **CLB LUT/LUTRAM/BRAM/DSP + Fmax** for `bp_relay_banked_bram_m` on `xck26`, and the fit go/no-go.

- [ ] **Step 1: Regenerate the 16/48 header and stage the fit inputs**

```bash
cd /Users/ex/GitHub/aleph
cargo run --release --example qec_q7_bp_graph -- circgraph 1 0.003 16 48 > hw/bb_gross_tanner.svh
scp hw/bp_relay_banked_bram_m.sv hw/bp_benes.sv hw/check_minsum.sv hw/var_update.sv hw/bb_gross_tanner.svh \
    root@195.154.249.85:/data/kv260fit/
```
Expected: 5 files copied. (Add `bp_benes.sv` to the glob the tcl reads — `ooc_serial.tcl` reads all `*.sv` in the dir, so no tcl change needed.)

- [ ] **Step 2: Launch the serial OOC synth (detached)**

```bash
ssh root@195.154.249.85 'cd /data/kv260fit && source /tools/Xilinx/Vivado/2024.2/settings64.sh && \
  mv -f fit.log fit_step1.log 2>/dev/null; \
  nohup vivado -mode batch -source ooc_serial.tcl -tclargs 5.0 m9c_step2 bp_relay_banked_bram_m \
    > fit.log 2>&1 & echo $! > fit.pid; echo "PID $(cat fit.pid)"'
```
Expected: PID printed, `fit.log` growing. **~8–9 h.** Poll with `ssh root@195.154.249.85 'grep RESULT /data/kv260fit/fit.log; ps -p $(cat /data/kv260fit/fit.pid) -o etime= 2>/dev/null || echo DONE'`.

- [ ] **Step 3: Record placed numbers + verdict**

On completion:
```bash
ssh root@195.154.249.85 'grep RESULT /data/kv260fit/fit.log; \
  sed -n "1,45p" /data/kv260fit/util_banked.rpt | grep -iE "CLB LUTs|LUT as Memory|CLB Registers|CARRY8|Block RAM|RAMB|DSP|F7|F8"'
```
Success = **fits xck26 with margin (target ≤ ~80 % CLB LUT)**, Fmax ≥ the sustained-rate clock. Compare to the Step-1 residual (2,232,451 LUT = 1906 %): the three Beneš fabrics replace ~1.15 M crossbar LUTs with ~(512·17 + 512·17 + 1024·19) ≈ 37 k switches ⇒ expect a **>20× LUT drop**.

- [ ] **Step 4: Write the verdict to the perf doc**

Append to `docs/perf/qec-q7-fixed-bp.md` § M9c: the Step-1 (1906 %) → Step-2 placed LUT/Fmax delta, the fit verdict, and — **if it fits** — the next action (re-fit the streaming W=6 wrapper to unblock AC-3 on-silicon). If it still overflows, record the residual breakdown (which resource) as the next-investigation seed.

- [ ] **Step 5: Commit the doc**

```bash
git add docs/perf/qec-q7-fixed-bp.md
git commit -m "[Q7-04] M9c step 2.6: EPYC synth verdict — Beneš networks placed util + fit

Records bp_relay_banked_bram_m xck26 placed CLB LUT/Fmax after sites 2-4 Beneš
conversion; Step-1 1906% -> [measured]. [Fit / no-fit] verdict + next action.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HNwUEvqsNoAqv1ykqHHNHn"
```

---

## Self-Review

**Spec coverage** (`2026-07-13-m9c-gather-crossbar-fix-design.md` § Step 2):
- "ROM-configured Beneš, 2×2-switch control from per-group ROM" → Tasks 1–3 (routing lib, control ROM emission, RTL fabric). ✅
- "Site 2 e_cm read; Sites 3–4 address/we scatters" → Tasks 4 (site 2), 5 (sites 3+4). ✅
- "≤1 access per bank per group ⇒ partial permutation the network realises exactly" → `complete_partial` + `verify_banking` guard (Task 1 Step 3, Task 2 Step 1). ✅
- "O(N·log N) switches vs O(N²) crossbar; one pipeline register if Fmax regresses" → pipelined proactively (`PIPE` param), latency absorbed (Tasks 4/5 latency gate). ✅
- "Emitter Beneš routing deterministic + guarded (verify_banking/rom_contract)" → Task 2 emit-time guard. ✅
- "Bit-exact (Beneš realises identity permutation of values) + latency unchanged" → Global Constraints + every co-sim step asserts 40/40 AND `worst latency` == 2206/3871. ✅
- "Success = fits xck26 ≤~80 % LUT, placed numbers in perf doc; streaming re-fit unblocks AC-3" → Task 6. ✅

**Placeholder scan:** No TBD/TODO. The two deliberately-iterative spots (`switch_wires` wiring in Task 1 Step 3; RTL butterfly in Task 3 Step 3) are gated by a concrete round-trip test that defines "done" precisely — that is convergence against an oracle, not a placeholder. All commands and expected outputs are literal.

**Type consistency:** `benes_control`/`benes_apply`/`complete_partial`/`benes_columns` signatures identical across Tasks 1→2. RTL names `bp_benes_ecm_read`/`_ecm_addr`/`_mcm_wr`, ports `din`/`ctrl`/`dout`, params `N`/`W`/`PIPE`, localparams `BP_BENES_ECM_M`/`BP_BENES_MCM_M`/`BP_BENES_ECM_COLS`/`BP_BENES_MCM_COLS`, ROMs `BP_ROM_BENES_ECMRD`/`ECMADDR`/`MCMWR` consistent across Tasks 2→3→4→5. Payload widths match the file-structure table (`2*MSG_BITS`, `BWC+2`, `1+BWC+MSG_BITS`). Latency gates (2206/3871) identical everywhere.

**Risks carried from spec:** (1) latency absorption is the convergence work in Tasks 4/5 — mitigated by the minutes-fast co-sim gate; (2) Fmax after pipelining is only known at Task 6 — `PIPE` is a localparam so a re-synth with deeper pipeline is a one-line change if Fmax is short; (3) site-4 write/read ordering under added latency is the subtlest correctness point (Task 5 Step 2 note) — the bit-exact gate catches it.
