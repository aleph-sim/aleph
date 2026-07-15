# M9c Step 4 (Q7-04) — e_cm read-address fabric → BRAM ROM

**Status:** design (2026-07-15). Follows the merged M9c Step-2 Beneš deliverable (PR #464,
`f816e22`). One focused fabric-elimination lever; part of the post-Beneš area campaign toward a
KV260 fit (`docs/perf/qec-q7-fixed-bp.md` § M9c).

## Problem

The Step-2 Beneš core (`hw/bp_relay_banked_bram_m.sv`) synthesises to **239,750 CLB LUTs =
204.7 %** of the KV260 (XCK26, 117,120 LUTs) — a 9.3× cut from the Step-1 crossbar, but still
2.05× over. The residual LUT is the three Beneš fabrics (`docs/perf/qec-q7-fixed-bp.md`, Step-2
table):

| fabric | N | LUTs | carries |
|--------|---|------|---------|
| `u_benes_wr` (m_cm write) | 1024 | 75,985 | runtime message data (row+payload) |
| `u_benes_rd0`+`rd1` (e_cm read gather) | 512×2 | 51,456 | runtime message data |
| **`u_benes_ad0`+`ad1` (e_cm read-addr scatter)** | 512×2 | **31,474** | **static per-group row addresses** |

The addr fabric is the odd one out: it routes **no runtime data**. Its input
(`bp_relay_banked_bram_m.sv:833`) is

```systemverilog
ecm_ad_din[s] = (s < NVB) ? {var_epres_r[s], var_erow_r[s]} : '0;
```

where `var_epres_r`/`var_erow_r` are registered outputs of `BP_ROM_VAR_EPRES` / `BP_ROM_VAR_EROW`
— pure ROM constants indexed by the var-group cursor. Its control (`benes_ecmaddr_q`) is likewise a
ROM (`BP_ROM_BENES_ECMADDR`, indexed by `var_rd`). Both operands are static per var-group `g`, so
the fabric output

```systemverilog
ra_ecm[b] = ad0_dout[b][BWC] ? ad0_dout[b][BWC-1:0] : '0;   // port-A read row for bank b
rb_ecm[b] = ad1_dout[b][BWC] ? ad1_dout[b][BWC-1:0] : '0;   // port-B read row for bank b
```

is a **pure function of `g`**. A 31,474-LUT rearrangeable permutation network is computing a fixed
lookup table.

## Approach

Replace the two `bp_benes_ecm_addr` instances (and their `BP_ROM_BENES_ECMADDR` control ROM) with a
single sync-read data ROM `BP_ROM_ECM_READROW[g]` that holds the already-resolved
`{ra_ecm[b], rb_ecm[b]}` for every bank `b` of every group `g`. No switches; the permutation is
baked into the table at generation time.

The read (`rd0/rd1`) and write (`u_benes_wr`) fabrics are **unchanged** — they carry genuine
runtime message data and must route.

### Considered alternatives (rejected)

- **Waksman-optimise the addr fabric instead.** At N=512 Waksman removes N−1=511 switches ≈ 5.9 %
  of the fabric — leaves ~29.6k LUT where the ROM leaves ~0. Rejected: an order of magnitude less
  win for the same verification cost.
- **Time-share addr with the read fabric** (the doc's "read/addr realise a permutation and its
  inverse" note). Rejected for this PR: at the fabric's II=1 the addr and read networks are both
  busy every cycle (read ctrl is `benes_ecmrd_q` delayed `BENES_PIPE_ECM` deep — the addr result
  feeds the read path 3 cycles later), so they cannot share physical switches without halving
  throughput. The ROM approach subsumes the intent (the addr network disappears entirely) at lower
  risk. Left as a separate investigation gated on the core's group-scheduling analysis.

## Design

### 1. Emitter (`crates/aleph-qec/examples/qec_q7_bp_graph.rs`)

Currently `print_rom_rows` (the `emit_rom` path) builds `benes_ecmaddr[g]` by
`benes_control(&complete_partial(dest_ecm[port]))` for each group and emits it as
`BP_ROM_BENES_ECMADDR`. Replace that emit with a **data ROM** built directly from the same
`dest_ecm[port][s]` matching that `benes_group_matchings` already returns:

For each var-group `g`, each bank `b ∈ 0..NEB`, each port `p ∈ {0,1}`:
- find the tap `s` with `dest_ecm[p][s] == b` (at most one — the ≤1-per-bank invariant
  `complete_partial` already asserts);
- if such `s` exists **and** the group's slot `s` is present (`var_epres[g][s]`): emit
  `var_erow[g][s]` (`BWC` bits);
- else emit `0`.

Pack per row as `readrow[g] = { {rb[NEB-1], ra[NEB-1]}, …, {rb[0], ra[0]} }`, `2·BWC` bits per
bank, `NEB·2·BWC` bits per row, `GV` rows. Emit as `BP_ROM_ECM_READROW` + localparam
`BP_ECM_READROW_W = NEB*2*BWC`. Drop the `BP_ROM_BENES_ECMADDR` emit and its `benes_ecmaddr`
accumulation.

**Gen-time guard (unchanged oracle).** Keep computing the addr permutation via
`complete_partial`/`benes_control`/`benes_apply` in the existing `verify_banking` guard, and assert
that applying it to `{epres, erow}` inputs reproduces the emitted `readrow[g]` table bank-for-bank.
The new ROM is thus proved against the already-trusted Beneš apply oracle — no independent
re-derivation of the mapping. `$fatal` on mismatch, exactly as the current guard does.

### 2. RTL (`hw/bp_relay_banked_bram_m.sv`)

Remove:
- `bp_rom_benes_ecmaddr_bqm` module + its instance `u_rom_benes_ecmaddr`;
- `benes_ecmaddr_q`, `ecm_ad_din`, instances `u_benes_ad0`/`u_benes_ad1`, and the
  `ra_ecm`/`rb_ecm` combinational select block that reads `ad0_dout`/`ad1_dout`.

Add:
- `bp_rom_ecm_readrow_bqm`: `(* rom_style = "block" *) logic [BP_ECM_READROW_W-1:0] rom [BP_GV]`,
  loaded from `BP_ROM_ECM_READROW`, sync-read, addressed by `BQM_AWV'(var_rd)` — the **identical
  address** that drove `u_rom_benes_ecmaddr` (line 771), so group timing is unchanged;
- unpack the row into `ra_ecm[b]`/`rb_ecm[b]` (`BWC` bits each, port-A low / port-B high per bank).

**Latency-match (bit-exactness).** The old addr path was `PIPE = BENES_PIPE_ECM` (3) cycles
(`din_t → dout_{t+3}`). A `rom_style="block"` sync read is 1 cycle. Register the ROM output through
`BENES_PIPE_ECM − 1` additional stages so the resolved `ra_ecm/rb_ecm` arrive at exactly the same
cycle as before. Then **every downstream offset is untouched**: `benes_ecmrd_q_d` stays 3 deep, the
var-operand twins stay 6, the S_VAR schedule shift stays +3, `BENES_ECM_LAT` stays 6. The change is
strictly local to the addr stage.

### 3. Size / resource

- ROM: `GV × NEB·2·BWC` bits. Shipped gross config (`ecm_m = 512 ⇒ NEB ≤ 512`, `BWC = clog2(GC)`,
  `GV` var groups). Concrete numbers come from the emitted localparams; upper bound ≈
  `201 × 512·2·6 ≈ 1.2 Mbit ≈ ~34 BRAM36`.
- The removed `BP_ROM_BENES_ECMADDR` control ROM was `GV × (2·ECM_COLS·(ECM_M/2))` =
  `201 × 8704 ≈ 1.75 Mbit`. Net BRAM is therefore **flat-to-freed**, not added; and 31,474 LUT are
  removed. Current BRAM is 223/288 (77 %) with ~65 BRAM36 free — comfortable either way.
- Expected result: **239,750 → ~208,300 LUT ≈ 177.9 %** (−13.1 pts). Still NO-FIT; this is one
  increment in the campaign (`partial-perm` on the 76k m_cm write net is the next and largest
  lever).

## Verification

1. **Gen-time guard** (above): emitted `readrow` table proved bank-for-bank against `benes_apply`
   on the addr permutation, `$fatal` on mismatch, for every banking.
2. **Verilator co-sim**: bit-exact **40/40 vs `FixedRelayBp`**, unchanged harness, at both bankings
   (8/24 and 16/48) — the same acceptance the Step-2 Beneš core met. Worst-case decode latency must
   be **identical** to the pre-change core (2810 cycles) since the latency-match keeps the pipeline
   depth byte-for-byte.
3. **Rust unit test**: a direct test that the emitter's `readrow` builder agrees with
   `benes_apply(benes_control(dest), {epres,erow})` on random group matchings (round-trip, mirrors
   `benes.rs::random_bijections`).
4. **OOC synth probe** on `xck26` (like the Step-3 area probe, ~8 min): synth the addr stage
   before/after to confirm the ~31k-LUT removal and no Fmax regression on the e_cm operand path.

## Out of scope

- Waksman variant (marginal at these N — separate follow-up if ever worth it).
- read/addr time-share (needs the core group-scheduling analysis; separate).
- partial-permutation right-sizing of the m_cm write / e_cm read fabrics (the next, larger lever).
- Real-time throughput: unchanged and still far over the 2 µs budget (a reach/fit lever, not speed;
  see `docs/perf/qec-q7-fixed-bp.md` § M9b honest framing).

## Files

- `crates/aleph-qec/examples/qec_q7_bp_graph.rs` — emit `BP_ROM_ECM_READROW`, drop
  `BP_ROM_BENES_ECMADDR`; guard.
- `hw/bp_relay_banked_bram_m.sv` — swap addr fabric for the readrow ROM + latency-match regs.
- `crates/aleph-qec/tests/` — emitter round-trip unit test.
- `docs/perf/qec-q7-fixed-bp.md` — § M9c Step-4 result note (after synth).
