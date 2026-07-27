# Decoder appliance — interface specification (v1 draft)

This is the contract between the decoder and whatever drives it. It is the most important document on
the product track: a lab that integrates against v1 must not have to rewrite when v2 (large FPGA) or
v3 (ASIC module) arrives, so the surface described here is meant to outlive all three.

**Status: v1 documents what is BUILT and MEASURED. Anything not yet built is marked "not implemented"
and carries no promise.** Nothing here is aspirational without saying so.

-----

## 1. What the decoder does

Given a syndrome for the gross bivariate-bicycle code `[[144,12,12]]` under a circuit-level noise
model, it returns a correction and the predicted logical-observable flips, plus a flag saying whether
the decode converged.

| quantity | symbol in RTL | width |
|---|---|---|
| syndrome in | `syndrome_in[BP_C]` | `BP_C = 144` bits |
| correction out | `corr_out[BP_N]` | `BP_N = 864` bits |
| observable flips | `obs_flip[BP_OBS-1:0]` | `BP_OBS = 12` bits |
| converged | `valid_flag` | 1 bit |
| latency | `latency_cycles` | cycles of the decoder clock |

`BP_C`, `BP_N`, `BP_OBS` are **not** hardcoded — they come from the generated graph header
(`bb_gross_tanner.svh`). A different code changes them. Any driver must read them from the build
rather than assume 144/864/12.

### `valid_flag` is a heralding signal, and that is load-bearing

Q7-07 measured, over 10⁶ shots per point at p = 0.003 / 0.005 / 0.007, that essentially **every**
logical error is a shot with `valid_flag = 0`: the attributable fraction `A(p)` is 1.0000 / 0.9973 /
0.9960, and `P(error | valid_flag = 1)` is ≤ 3.0e-6 / 1.9e-5 / 1.2e-4.

**Consequence for integrators:** a converged decode is almost never wrong, and the decoder's logical
error rate is essentially its non-convergence rate. If your architecture can do anything useful with a
"decode failed" herald — retry, flag the round, fall back — `valid_flag` is where the value is. Do not
discard it. See `docs/qec/q7-07-nonconvergence-policy.md`.

-----

## 2. Transport A — AXI4-Lite (implemented, measured)

The register map below is as-built in `hw/bp_axi_wrap_banked.sv`. `NS = ceil(BP_C/32)`,
`NC = ceil(BP_N/32)`; byte addresses, 32-bit data.

| offset | access | contents |
|---|---|---|
| `0x00` CTRL | W | bit0 `START` (self-clearing), bit1 `EARLY_EXIT` (sticky: 1 = stop at first valid) |
| `0x04` STATUS | R | bit0 `BUSY`, bit1 `DONE` (sticky), bit2 `VALID` (= `valid_flag`) |
| `0x08` LATENCY | R | last decode latency in cycles (32-bit) |
| `0x0C` OBS | R | `obs_flip[BP_OBS-1:0]` |
| `0x10` IDCODE | R | `0x4250_0003` — `'BP'`, v3 = the banked-core wrapper |
| `0x40` SYND0.. | RW | syndrome, `NS` words; word *i* = `syndrome[i*32 +: 32]`, low `BP_C` bits used |
| `0x80` CORR0.. | R | correction, `NC` words; word *i* = `correction[i*32 +: 32]`, low `BP_N` bits used |

**Read IDCODE first.** It is how a driver tells overlays apart; a different core presents a different
value and the register map is not guaranteed to match.

### Bit order — the thing everyone gets wrong

Bit *j* of the logical vector lives at **bit `j mod 32` of word `j div 32`**, i.e. little-endian within
each 32-bit word, word 0 holding the lowest indices. This has bitten this project before; it is
spelled out rather than left to inference.

### Sequence

1. Write `SYND0..` (`NS` words).
2. Write CTRL with `START` (and `EARLY_EXIT` if wanted).
3. Poll STATUS until `DONE`.
4. Read `VALID`, `OBS`, `CORR0..` (`NC` words), `LATENCY`.

`EARLY_EXIT = 0` gives the full best-kept schedule and is what every golden comparison in this
repository uses. `EARLY_EXIT = 1` stops at the first syndrome-valid decision: much lower median
latency, same worst case.

-----

## 3. Transport B — AXI-DMA batch (implemented, measured)

AXI4-Lite is a per-decode PS round trip and the PS dominates it. The batched DMA path streams many
syndromes through without the processor in the per-decode loop; Q7-06 measured **≥ 100×** the
throughput of the polled path on silicon (`docs/qec/q7-06-ac1-batched-dma.md`).

Use this for any throughput-bound or Monte-Carlo workload. Use AXI4-Lite for single-shot,
lowest-complexity integration and bring-up.

-----

## 4. Transport C — low-latency serial — **NOT IMPLEMENTED**

For use as an *external* decoder box next to someone else's control electronics, neither transport
above is right: both assume the decoder is inside your own FPGA fabric, behind a processor. The
missing piece is a point-to-point serial link (LVDS, or Aurora over SFP+) carrying syndrome in and
correction out with bounded latency.

**This does not exist.** It is specified here because it is the interface a control-electronics vendor
would actually wire up, and because deciding it late would break the stability promise in §6. Anyone
who needs it should say so before it is designed — the shape should be driven by a real integration,
not guessed.

-----

## 5. Latency

Latency is `cycles / f_clk`. Cycles depend on the core and its configuration, not on the transport:

| core | configuration | cycles | source |
|---|---|---|---|
| banked | 16/48 | 2085 | measured, `bpbankedscale` |
| banked | 32/96 | 1283 | measured |
| banked | 64/192 | 913 | measured |
| banked | 144/864 (full-parallel) | 544 | **model only — does not generate**, see Task B1 |
| **unrolled (M4)** | full-parallel | **181** | measured, `bpunrollcirc`, bit-exact |

Exact model for the banked core: `cycles = LEGS·ITERS·(GC+GV+7) + (2·GV+GC+1)`. The `LEGS·ITERS·7`
per-iteration pipeline tail is **banking-invariant**, which is why 4× banking buys 2.28×, not 4×.

The unrolled core has no such tail: it spends 3 cycles per sweep regardless of graph size, so
`cycles = LEGS·ITERS·3 + 1`.

Measured on silicon, KV260 at 133.332 MHz, banked 16/48: **15.64 µs worst case, 0.85 µs median with
early exit.**

-----

## 6. Stability promise

Across v1 (KV260) → v2 (large FPGA) → v3 (ASIC module):

- The **register map in §2 will not change meaning.** New registers may be added at unused offsets.
- **IDCODE changes whenever the map changes.** Check it; do not assume.
- The **data formats in §1 and the bit order in §2 will not change.**
- `valid_flag` semantics will not change.
- Latency **will** change — that is the point of the newer versions — so never hardcode a timeout in
  cycles or microseconds. Poll `DONE`, and size timeouts from `LATENCY` plus margin.

Anything not on this list is not promised.

-----

## 7. What this is not

- Not a surface-code MWPM decoder. It decodes qLDPC (bivariate-bicycle) via relay belief propagation.
- Not code-agnostic at runtime: the Tanner graph is baked into the build. A different code means a
  different bitstream.
- Not a syndrome-extraction system. Syndromes come from your control electronics; this decodes them.
- Not sub-microsecond today on the shipped v1 build. See §5 for what is measured on what.
