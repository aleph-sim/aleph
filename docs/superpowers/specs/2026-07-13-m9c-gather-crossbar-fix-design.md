# M9c gather-crossbar fix — design

**Issue:** Q7-04 M9c. **Goal:** make the banked relay-BP core (`hw/bp_relay_banked_bram_m.sv`)
**fit and route on the KV260** (`xck26`, 117 120 LUT) so the streaming sustained-rate (AC-3) can be
built, while staying **bit-exact** to `FixedRelayBp` / the M8 LUT core in Verilator co-sim (40/40).
One core must serve both rounds=1 and the W=6 streaming window.

## Root cause (measured on EPYC, GROSS config W=16, CHK_DEG=25 → NHB=800, NEB=400, NVB=288)

The 8 h EPYC synth cleared all logic-opt and OOM'd at Technology Mapping on **~1.17 M inferred
muxes**. They come from **ROM values used as register-array *indices*** — the exact pattern
`hw/bp_relay_banked.sv`'s header rule forbids ("ROM outputs may feed data or real-memory addresses
only, never register-array indices"). The M9b BRAM-ify introduced them. Per site:

| # | site (`bp_relay_banked_bram_m.sv`) | expression | why it explodes | class |
|---|---|---|---|---|
| 1 | m_cm CHK read (~L834) | `qmcm[chk_hbsel_r[idx]]` | `chk_hbsel_r[idx] = idx*2+beta`; only `beta` (1 bit) varies, but the 10-bit ROM value is **opaque** to Vivado → full **800-way** mux × 400 taps ≈ **235 k** | **tap-structured** |
| 2 | e_cm VAR read (~L880) | `qa/qb_ecm[var_ebsel_r[idx]]` | var reads the e_cm banks of *its checks* (scattered) → genuine **400-way** crossbar × 288 taps | permutation |
| 3 | e_cm read-addr scatter (L754) | `ra/rb_ecm[var_ebsel_r[idx]] = var_erow_r[idx]` | 4-bit row demuxed into a 400-entry array by runtime index × 288 | permutation |
| 4 | m_cm write scatter (L735) | `we/wa/wd_mcm[scat_hb_r[idx]]` | var-slot writes to its check-tap's half-bank (scattered) → 800-way demux × 288 | permutation |

`m_vm` is the in-file proof of the correct shape: `qmvm[idx]` with `idx` a **compile-time constant**
(writer==reader partition) → no mux. The fix makes m_cm/e_cm read/write the same way.

## Design principle

Turn every **ROM-value-as-index** into either (a) a **compile-time-constant index + a small runtime
select** where the index is structured (site 1), or (b) a **ROM-configured permutation network**
where it is a genuine var↔check permutation (sites 2–4). Messages/values are unchanged → decisions
stay bit-exact; only *where* each lives and *how* it is routed changes.

## Incremental plan (de-risks the ~8 h EPYC synth loop)

Land the cheap, huge win first, **measure**, then do the hard part only for the real residual.

### Step 1 — m_cm CHK read: 2:1 beta-split (concrete first deliverable)

Rewrite site 1 so the tap base is a constant and only `beta` is runtime:

```systemverilog
// before: 800-way mux (Vivado can't see chk_hbsel_r[idx] ∈ {idx*2, idx*2+1})
m_in_j[k] = chk_epres_r[idx] ? qmcm[chk_hbsel_r[idx]] : '0;
// after: 2:1 mux, qmcm[idx*2], qmcm[idx*2+1] are compile-time constants
m_in_j[k] = chk_epres_r[idx] ? (beta_r[idx] ? qmcm[idx*2 + 1] : qmcm[idx*2]) : '0;
```

- `idx = j*BP_CHK_DEG + k` is a genvar/loop constant (per-tap), so `idx*2`/`idx*2+1` fold to wires.
- `beta_r[idx]` is a **1-bit** ROM (the low bit of the old `chk_hbsel`, i.e. `BP_EDGE_HB[e] & 1`).
  Emitter emits a `BP_ROM_CHK_BETA` (1 bit/tap/group) replacing `BP_ROM_CHK_HBSEL` (10 bit/tap/group)
  for this site — or, simplest and lowest-risk: **keep `chk_hbsel_r` but assert `chk_hbsel_r[idx] ∈
  {idx*2, idx*2+1}` and derive `beta = chk_hbsel_r[idx][0]`** (no new ROM, an elaboration guard proves
  the invariant, and the RTL reads `beta = chk_hbsel_r[idx] - idx*2`). Prefer the derive-in-RTL form:
  zero emitter change, the invariant is already true by construction (`HB = eb*2+beta`, `eb = idx`).
- Expected: **~235 k muxes → ~400**. Bit-exact by construction (same two banks, same select bit).

**Verify:** `make -C hw bpbankedbramm` (Verilator 40/40 vs golden, both bankings) — minutes.
**Then synth on EPYC** (serial `maxThreads 1`, 128 GB + swap) → read the util/mux count. Decide if
sites 2–4 are still needed (they likely are — sites 2–4 ≈ the 691 k 4-bit muxes — but measure).

### Step 2 — e_cm gather + scatters: ROM-configured Beneš permutation (contingent on step-1 measurement)

Sites 2–4 are genuine var↔check permutations (≤1 access per bank per group ⇒ a **permutation**, not
arbitrary fan-out). Replace each runtime-indexed crossbar/demux with a **Beneš network** whose
2×2-switch control bits come from a per-group ROM the emitter computes by Beneš routing:

- **Site 2** (e_cm read): route the 400 e_cm bank outputs → 288 var-tap operands through a Beneš net;
  `var_eport` stays the a/b port select at the leaf.
- **Sites 3–4** (address/we scatters): route the 288 writer (row/we/data) tuples → their target banks
  through a Beneš net (write direction). Because ≤1 writer per bank per group, this is a partial
  permutation the network realises exactly.
- Cost: O(N·log N) 2×2 switches (~288·9 ≈ 2.6 k switches/net) vs the O(N²) crossbar (~115 k). Net
  combinational depth ≈ log₂N stages — may need **one pipeline register** (schedule-neutral, folded
  into the existing M8 gather register plane) if Fmax regresses; note as a timing risk.

The exact per-site network sizing and whether a pipeline stage is needed are **decided after the
step-1 measurement**, using the real residual mux/LUT count — not guessed now.

## Emitter changes (`crates/aleph-qec/examples/qec_q7_bp_graph.rs`)

- Step 1: **none** if we derive `beta` in RTL (preferred). Otherwise a 1-bit `BP_ROM_CHK_BETA` table.
- Step 2: extend `solve_banking`/`print_rom_rows` to (a) confirm the ≤1-access-per-bank-per-group
  invariant already guaranteed by `cap`, (b) run a **Beneš routing** per group over the (writer→bank)
  and (bank→reader) partial permutations, (c) emit the control-bit ROMs as packed-row literals (house
  pattern), with a `rom_contract`-style time-0 guard recomputing routes and `$fatal` on mismatch.

## Bit-exactness

No message value changes. Step 1: same two candidate banks, same select bit → identical read. Step 2:
a permutation network delivers exactly the same value to exactly the same consumer as the crossbar
did (Beneš realises the identity permutation of *values*), verified by the `rom_contract` guard + the
existing `bpbankedbramm` co-sim (40/40 vs `FixedRelayBp`, both bankings, worst-latency unchanged).

## Verification plan

1. **Verilator co-sim after each step** — `make -C hw bpbankedbramm` (40/40, both bankings) and the
   streaming gates (`make -C hw bpstream`) must stay green. This is the correctness gate; **no synth
   before it passes.**
2. **Incremental EPYC synth** — after step 1, and again after step 2, run the OOC fit
   (`ooc_serial.tcl`, `bp_relay_banked_bram_m`, xck26) on EPYC and record placed LUT/LUTRAM/BRAM/DSP +
   Fmax. Success = **fits xck26 with margin** (target ≤ ~80 % LUT, matching the M8 LUT core's 80 %).
3. **Streaming re-fit** — once the rounds=1 core fits, re-confirm the W=6 streaming wrapper fits
   (it reuses this core), unblocking the M9c bitstream + on-silicon sustained-rate (AC-3).

## Risks / open questions

- **Step 1 alone may or may not be enough.** If the residual (sites 2–4) still overflows, step 2 is
  required. Measurement after step 1 decides — that's the point of sequencing.
- **Beneš adds combinational depth** → possible Fmax hit needing a pipeline register (schedule-neutral
  but must re-verify latency = 2206 cyc). Decide with post-step-1 timing.
- **Emitter Beneš routing must be deterministic + guarded** (house `verify_banking`/`rom_contract`
  pattern) so the ROM control bits are provably correct at generation time.
- **8 h synth per measurement** — keep steps small and Verilator-gated so each synth is a real signal,
  not a debug round-trip.

## Success criteria

- `bp_relay_banked_bram_m` fits `xck26` with margin (≤ ~80 % LUT), placed numbers in
  `docs/perf/qec-q7-fixed-bp.md`.
- 40/40 bit-exact vs `FixedRelayBp` at both bankings; streaming gates green; latency unchanged.
- Streaming W=6 core fits → M9c bitstream buildable → AC-3 (on-silicon sustained-rate) unblocked.
