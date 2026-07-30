# Decoder appliance

A real-time decoder for the gross bivariate-bicycle qLDPC code `[[144,12,12]]`, on hardware you can
buy for about $300.

**Status: v1 is being packaged.** The RTL, the golden model and the silicon results all exist and are
in this repository; what is missing is the last mile that turns them into something a stranger can
deploy in an afternoon. That gap is the whole point of this directory — see "What is still missing".

-----

## Versions

| version | hardware | config | worst-case latency | already real-time for |
|---|---|---|---|---|
| **v1** | Kria KV260, ~$300 | banked 16/48 | **15.64 µs** (0.85 µs median, early exit) | neutral atoms, trapped ions — round times 100 µs–ms |
| v2 | large FPGA ($10–30 k) | banked 144/864, 543 cycles | **~4.7 µs projected** (synthesis-only, calibrated) | most platforms except the fastest superconducting loops |
| v3 | ASIC module | banked 144/864, 543 cycles | 0.79–0.91 µs projected at 600–686 MHz | superconducting, ~1 µs rounds |

v1 is the important row: **for two of the four qubit modalities this is finished and merely
undistributed.** If your syndrome rounds are 100 µs apart, a $300 board already decodes them in real
time, today, with the numbers above measured on silicon rather than projected.

v2 and v3 are **not built systems**. 543 cycles for the full-parallel banked 144/864 configuration is
bit-exact against the golden model at 40/40 (`bpbankedscale`, 2026-07-30), but the *clock* that turns
cycles into microseconds is not measured for either row.

v2's ~4.7 µs is the least-bad kind of projection: out-of-context synthesis put 144/864 at 154 MHz, and
the same sweep contains 16/48 — a design we have actually routed and run on silicon — which lets us
measure that the tool overstates achieved Fmax by 1.33× on this core. De-rating by that gives ~115 MHz.
It is still synthesis, with no placement behind it, on a part nobody has built this design on
(`docs/perf/q7-02-fullparallel-fpga.md`). v3's range comes from the one ASIC-node number we have,
686 MHz on ASAP7, which carries an unresolved gated-clock caveat (`docs/perf/q7-02-asap7-timing.md`).

**Do not quote the v2/v3 rows as capability.** An earlier revision of this file projected v2/v3 from
the fully-unrolled core at 181 cycles and 200–300 MHz. That was wrong: the unrolled core was
subsequently synthesised and needs **838 % of the KV260's LUTs at 30.7 MHz** — it does not fit and is
not fast, so 181 cycles buys nothing. The rows above are the surviving configuration.

## What is actually proven

- **Bit-exact against the software golden model.** 25+ Verilator co-simulation targets in
  `hw/Makefile`; the fast ones are enforced in CI on every change (`.github/workflows/hw.yml`).
- **0 mismatches in 10⁶ × 3 shots on silicon** (Q7-06, matched-prior campaign at p = 0.003/0.005/0.007).
- **`valid_flag` heralds essentially every logical error** (Q7-07) — see `interface-spec.md` §1, this
  is the most useful property of the thing and the easiest to accidentally throw away.
- **Measured, not projected, latency on the KV260**: 15.64 µs worst, 0.85 µs median with early exit,
  at 133.332 MHz, timing met.

## What this is not

It decodes qLDPC bivariate-bicycle codes via relay belief propagation. It is not a surface-code MWPM
decoder, it is not runtime-reconfigurable to another code (the Tanner graph is baked into the build),
and it does not extract syndromes — it consumes them. Full list in `interface-spec.md` §7.

For honest comparison: Riverlane ships a hardware decoder at **<1 µs per round** for the surface code,
and NVIDIA ships relay-BP itself on GPUs through CUDA-Q QEC over NVQLink. We are not first and do not
claim to be. What is different here is that the design, the data and the results are open, and the
hardware costs $300 rather than $30,000.

## Getting it

- **RTL and simulation:** this repository. `make -C hw bpbanked` runs the banked core against the
  golden in about six minutes and needs only Verilator ≥ 5.050 and a Rust toolchain.
- **Bitstream:** not yet published as a release artefact — see below.
- **Driver:** `hw/sw/bp_stream_banked_kv260.py` (PYNQ) is the working starting point.

## What is still missing before v1 can be called shipped

Honest list, in the order it blocks people:

1. **A published bitstream.** The working images live on the development board, not in a release.
   Nobody outside can deploy without rebuilding from RTL, which needs Vivado and hours.
2. **`deploy.sh`** — one command from a bare KV260 to a running self-test. Today the steps are spread
   across `hw/sw/` scripts and institutional memory.
3. **A fresh-board test.** The deployment has never been walked through by someone who did not build
   it. Until that happens the claim "deployable in an afternoon" is untested, and the number of
   external deployments — which is the gate for everything downstream on the silicon track — cannot
   honestly be counted.
4. **A support policy.** What is answered, how fast, what "stable" means. One page, not yet written.

Items 1–3 are the difference between a repository and a product, and they are cheap. They are not done
yet, and this file says so rather than implying otherwise.

## Licence

Everything under `hw/` is Apache-2.0 (`hw/LICENSE`) — permissive, with an explicit patent grant so
that pulling this RTL into silicon does not require a legal negotiation. The Rust crates are MIT.
