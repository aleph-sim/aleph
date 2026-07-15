# M9c Step 3 — serial-gather relay-BP core: design

**Issue:** Q7-04 M9c (KV260 fit). **Goal:** make the banked relay-BP core **fit and route on the
KV260** (`xck26`, 117 120 LUT) by replacing the parallel gather/scatter with a **memory-based serial
gather**, while staying **bit-exact** to `FixedRelayBp` in Verilator co-sim (40/40, both bankings).

## Why (measured)

Two prior attempts did not fit:

| core | CLB LUTs | vs 117 120 |
|------|----------|------------|
| M9b crossbar (`bp_relay_banked_bram_m`, Step 1) | 2 232 451 | **1906 %** |
| M9c Beneš (Step 2, `docs/perf/qec-q7-fixed-bp.md` § M9c) | 239 750 | **204.7 %** |

Step 2 eliminated the runtime-mux crossbars (F7/F8 muxes 238 %/231 % → 0.9 %/0 %) for a **9.3× LUT
cut**, but the **Beneš permutation networks themselves** are ~159 k LUT (66 % of the total) — because
the core gathers all ~288 edges *in parallel* every cycle, which inherently needs 512/1024-wide
permutation fabrics (`u_benes_wr` alone is 76 k). Incremental fabric levers (Waksman ~5 %, beta-split
~10 %, read/addr time-share ~41 k) plateau ~1.3–1.7× over budget — a parallel gather cannot fit.

**Decision (user, 2026-07-15):** guarantee the KV260 fit; accept the throughput hit. The ~1 µs/round
sustained target was already out of reach worst-case (spec's honest-expectations; `docs/perf/qec-q7-fixed-bp.md`
line ~1349) — M9c's deliverable is a **routable, on-silicon-buildable** core, not sub-µs streaming.

## Principle

Replace the parallel gather (400/800 banks read at once + a permutation network) with a **serial,
memory-based gather**: hold the messages in **P physical BRAM banks**, read **P messages per cycle**
over `ceil(N/P)` cycles into an **in-order buffer**, then feed the *unchanged* parallel
`check_minsum`/`var_update`. The scatter (writing results back) is the mirror. No plane is ever a
runtime-indexed register mux — the ROM feeds **BRAM address ports**, the bank-select and buffer
position are **cycle-counter** driven (compile-time-sequenced constants).

## Architecture

```
                per cycle-step c (0 .. STEPS-1):
  addr ROM ──► [bank 0].addr = rom_a[c][0] ─► qa0 ─┐
  (BRAM)       [bank 1].addr = rom_a[c][1] ─► qa1 ─┤ P reads,
               ...                                  ├─► buffer[c*P + b]   (b = 0..P-1, counter index)
               [bank P-1].addr= rom_a[c][P-1]─► qaP─┘
                                                     │  after STEPS cycles →
                                                     ▼
                        buffer holds the gathered edges in TAP ORDER
                                                     │
                                                     ▼
                         unchanged parallel check_minsum / var_update
                                                     │  results
                                                     ▼
  scatter: drain buffer[c*P+b] ─► [bank b].wr(addr = rom_a'[c][b])   (mirror, mux-free)
```

### Components (each a focused unit)

1. **`bp_serial_mem` — P-banked message store.** Replaces the 400 `bp_ecm_cell` / 800 `bp_mcm_cell` /
   288 `bp_mvm_cell` per-bank BRAMs with **P wide BRAM banks** per message class (e_cm, m_cm, m_vm).
   Each bank is a true-dual-port BRAM (`RAMB36`): its address is a ROM value; its bank-index is the
   round-robin counter. **Interface:** `read(step) → [P]×MSG_BITS` (bank b at `rom_a[step][b]`),
   `write(step, [P]×MSG_BITS)`.
2. **`bp_gather_seq` — gather/scatter sequencer FSM.** Drives the cycle counter `c ∈ 0..STEPS-1`,
   presents `rom_a[c]` to the P banks, and writes the P outputs to `buffer[c*P +: P]`. On scatter,
   reverses: reads `buffer[c*P +: P]`, writes the banks at `rom_a'[c]`. **Interface:** `start`,
   `busy`, `done`; produces the full gather buffer when `done`.
3. **`bp_gather_buf` — in-order buffer.** A register file of `N` = (per-group edge count) × MSG_BITS,
   written at position `c*P+b` (counter index → no demux), read as a parallel `[N]` bus by the
   consumers. Small (N ≤ 288 words × MSG_BITS).
4. **Unchanged:** `check_minsum`, `var_update` — they consume/produce the parallel buffer exactly as
   today. This is a hard constraint: **no change to the arithmetic modules** (they are already
   oracle-verified).

### Offline conflict-free banking (emitter — `qec_q7_bp_graph.rs`)

The **precondition** that makes every plane mux-free: each edge is assigned to a **(bank b, step c)
slot** such that **no two edges of the same group share a (b, c) slot** (≤ 1 read per bank per step).
The emitter extends `solve_banking` to compute this assignment and emit two address-ROM tables per
message class:
- `rom_a[c][b]` — the intra-bank address (row) to read/write at step c, bank b.
- The **tap-order mapping** is implicit: the solver orders slots so `buffer[c*P+b]` is exactly the tap
  the consumer expects (same role the Beneš control played, now a static slot assignment).
- A **time-0 `rom_contract` guard** (house `verify_banking` pattern) re-derives the assignment and
  `$fatal`s on any (b, c) collision or tap-order mismatch — so correctness is proven at generation
  time, not just in co-sim.

Conflict-free assignment always exists for `P ≤ (banks)` because the original per-edge banking already
guarantees ≤ 1 access per *physical* bank per group; the solver simply folds the (up to) 400 logical
banks onto P physical banks across `ceil(N/P)` steps (a bin-packing / edge-colouring the solver runs
offline, deterministic + guarded).

## P (serialization factor): max that fits

`STEPS = ceil(N / P)` cycles per gather; larger P → fewer steps (better throughput), more BRAM. **P is
pinned during the implementation plan from the KV260 BRAM budget** (144 `RAMB36` / 288 `RAMB18`), less
what the message stores + control ROMs already consume (Step-2 used 223 RAMB tiles — so the P-bank
consolidation must also *reduce* BRAM pressure vs the 400/800 tiny-bank layout, or trade to `RAMB18`).
Target: the **largest P whose e_cm/m_cm/m_vm banks + buffer + FSM fit ≤ ~80 % LUT and ≤ ~90 % BRAM**,
minimizing STEPS. The plan pins P from a first synth probe; the RTL keeps P a `localparam` so it is a
one-line retune.

## Data flow / schedule

Each processing group, formerly ~1 gather-cycle, becomes: **`STEPS` serial-read cycles → parallel
compute (check_minsum/var_update, unchanged pipeline) → `STEPS` serial-write cycles** (write may
overlap the next group's reads if the banks free up — a plan-time pipelining question). Total decode
latency grows ≈ `× (STEPS + compute)/(1 + compute)`; measured, not assumed. The control FSM
(`S_INIT/S_CHECK/S_VAR`) gains the inner serial-step loop; early-exit/convergence logic is unchanged.

## Bit-exactness

No message value changes — the serial gather delivers exactly the same messages to exactly the same
consumers, only spread over `STEPS` cycles instead of 1. Gate: `make -C hw bpbankedbramm` = 40/40
decision-equal vs `FixedRelayBp` at **both** bankings (8/24, 16/48); the gen-time `rom_contract` guard
proves the slot assignment; latency is **allowed to grow** (recorded, not fixed).

## Verification plan

1. **Verilator co-sim after each stage** — `make -C hw bpbankedbramm` 40/40 both bankings is the
   correctness gate; no synth before green.
2. **New standalone TBs** for `bp_gather_seq` + `bp_serial_mem` (random conflict-free schedules,
   bit-exact readback), mirroring the `bpbenes`/`varupdate` module-TB pattern.
3. **EPYC OOC synth** (`ooc_serial.tcl`, `xck26`) — record placed LUT/BRAM/DSP/Fmax. Success =
   **fits xck26 ≤ ~80 % LUT** with the P chosen. The synth loop is now ~25 min (no OOM), so P can be
   tuned across a few probes.
4. **Throughput note** — record the resulting worst/median latency in µs at Fmax; document AC-3 as
   **met on fit (routable/buildable)**, throughput degraded vs the parallel core, per the user decision.

## Success criteria

- New serial-gather core (sibling to `bp_relay_banked_bram_m`) **fits `xck26` ≤ ~80 % LUT**, placed
  numbers in `docs/perf/qec-q7-fixed-bp.md` § M9c.
- 40/40 bit-exact vs `FixedRelayBp` at both bankings; `check_minsum`/`var_update` unchanged.
- Beneš fabrics (`bp_benes.sv`) no longer instantiated by this core (Step-2 core kept as the
  parallel/over-budget sibling for the record).
- Streaming W=6 wrapper re-fits on the serial core → M9c bitstream buildable → AC-3 unblocked
  **on-silicon (fit), throughput documented honestly**.

## Risks / open questions

- **BRAM pressure vs LUT relief.** Consolidating 400/800 tiny banks into P wide banks must not blow
  the BRAM budget (Step-2 already at 77 %). The P-bank layout must pack messages efficiently
  (`RAMB18` vs `RAMB36`, depth sharing). Decided by the first synth probe (plan Task: pin P).
- **Serial-write pipelining.** Whether the scatter overlaps the next group's reads (throughput) or is
  a separate phase (simpler, slower) — a plan-time schedule decision, co-sim-gated on latency.
- **Emitter conflict-free solver** must be deterministic + guarded (house pattern) so the slot ROMs
  are provably collision-free at generation time.
- **Throughput** may exceed the ~1 µs median — accepted; recorded honestly. If it is far worse than
  expected, P and the write-overlap are the levers before reopening scope.
