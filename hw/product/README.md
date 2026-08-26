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
| v2 | large FPGA ($10–30 k) | banked **64/192**, 913 cycles | **6.07 µs — implemented and measured** | most platforms except the fastest superconducting loops |
| v3 | ASIC module | banked 144/864, 543 cycles | **0.88 µs at ASAP7 7 nm predictive** — see the node caveat below | superconducting, ~1 µs rounds |

v1 is the important row: **for two of the four qubit modalities this is finished and merely
undistributed.** If your syndrome rounds are 100 µs apart, a $300 board already decodes them in real
time, today, with the numbers above measured on silicon rather than projected.

**v2's number is measured, not projected.** 64/192 was placed and routed on an AMD Virtex UltraScale+
HBM VU47P — the part in AWS's F2 instances — at **150.4 MHz using 19 % of the device**, giving 6.07 µs
for its 913 cycles (`docs/perf/q7-02-fullparallel-fpga.md`). What is *not* built is a board: this is an
implementation result on a rented part, not a product you can buy today.

v2 deliberately ships **64/192 rather than the full-parallel 144/864**, and the reason is worth stating
because it is counter-intuitive. 144/864 also fits — 76.3 % of the same device — but it routes at only
97.3 MHz, so its 543 cycles come out at 5.58 µs. Fewer cycles, proportionally worse clock: **1.09×
faster for 4× the area.** 64/192 gives 92 % of the latency in a quarter of the part, leaving the rest
free for a host interface.

**v3's 0.88 µs is a real place-and-route result on the wrong node.** The full-parallel 144/864
configuration was implemented end to end on ASAP7 — 543 cycles at 614.59 MHz, 0.869 mm², **zero DRC
violations** (`docs/perf/q7-02-b3-asap7-fullparallel.md`). It is not arithmetic and not an
extrapolation.

But **ASAP7 is a predictive academic 7 nm PDK, and the chip this project can actually afford is
28 nm.** Nothing here measures that gap, and 0.88 µs has only 12 % of margin: a node penalty of 1.2×
gives 1.06 µs and the sub-microsecond claim is gone. So the honest statement is *"sub-microsecond at
7 nm predictive"*, and anyone quoting v3 should quote it that way.

Two further caveats on v3, both real: the run skipped timing repair (`repair_timing` segfaults on this
netlist), leaving 35,616 hold violations that slowing the clock does not fix; and there is no 28 nm PDK
access yet, which is a funding-and-agreements gate rather than an engineering step.

One thing the FPGA result predicted wrongly, recorded because it was load-bearing: on FPGA this core
lost 35 % of its clock going full-parallel, which looked like evidence that the geometry itself was the
problem. On an ASIC the same step costs 10 %. The FPGA penalty was fixed routing tracks and multi-die
crossings, not the design.

**Do not quote the v3 row as capability.** An earlier revision of this file projected v2/v3 from the
fully-unrolled core at 181 cycles and 200–300 MHz. That was wrong: the unrolled core was subsequently
synthesised and needs **838 % of the KV260's LUTs at 30.7 MHz** — it does not fit and is not fast, so
181 cycles buys nothing. The rows above are the surviving configuration, and v2 has since been
implemented rather than projected.

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
- **On a board:** first bring the board up — **`BRINGUP.md`**, which is Ubuntu 22.04 (*not* 24.04),
  a firmware check, a build toolchain the stock image lacks, and Kria-PYNQ pinned to v3.0. Each of
  those four is a wall someone hits if they follow the upstream instructions instead. Then
  `sudo ./deploy.sh`: it fetches the published release, checks every artefact against its SHA-256,
  programs the PL and requires 40/40 bit-exact against the golden before declaring success. It needs no
  Rust toolchain and installs nothing outside `/opt/aleph-decoder`. **Until the release in item 1 below
  exists there is nothing for it to fetch.**
- **Driver:** `hw/sw/bp_stream_banked_kv260.py` (PYNQ) — what `deploy.sh` runs, and the starting point
  for your own integration.

## What is still missing before v1 can be called shipped

Honest list, in the order it blocks people:

1. **A published bitstream.** The working images live on the development board, not in a release.
   Nobody outside can deploy without rebuilding from RTL, which needs Vivado and hours. The procedure
   for publishing one is written (`RELEASING.md`); it has not been run.
2. ~~**`deploy.sh`**~~ — **written and now executed end to end.** On 2026-08-24 it took a freshly
   flashed card to a verified decoder: preflight, artefact fetch, SHA-256 check, PL programming, and
   `CORRECTNESS: PASS (40/40 batched decodes match golden on KV260 silicon)` at 4.34e4 decodes/s. The
   fetch path is still unexercised — the files were placed locally, because item 1 has not happened.
3. **A deployment walked through end to end.** **Done on a spare card, 2026-08-24**, and it earned its
   keep: **nine** blockers (the ninth on the 2026-08-25 second reflash), none of them in our code,
   all now written into `BRINGUP.md` and `RELEASING.md`. The download page defaults to an Ubuntu release Kria-PYNQ cannot install on; a
   multi-hour first-boot upgrade holds the dpkg lock; the stock image has no C compiler and no Boost
   headers; the firmware step every guide insists on is usually unnecessary and is the only one that
   can brick the board; the installer is interactive and exits non-zero over an unrelated demo package;
   pip resolves a numpy that breaks PYNQ mid-install and at runtime; pip's isolated build environment
   pulls a `wheel` that demands a `packaging` newer than jammy ships, so PyAudio fails before it
   compiles — both now held by one `PIP_CONSTRAINT` file rather than patched after the fact; and
   shipping the `.bit` without its `.hwh` presents as dead DMA hardware rather than a missing file.

   Two different claims, and only the weaker one is currently reachable:
   - *the procedure is complete* — **achieved.** A fresh card, deployed using only these documents,
     reaching a 40/40 self-test. Every step that existed only in somebody's memory is now written down.
   - *a stranger can deploy it* — someone who has never seen this repository does it unaided. **This
     needs a second board or a second person, and we have neither.**

   Only the second is what the plan's Task P1 Step 4 asks for. The count of external deployments gates
   every silicon decision downstream, so the difference is recorded rather than blurred.
4. **A support policy.** What is answered, how fast, what "stable" means. One page, not yet written.

Items 1–3 are the difference between a repository and a product, and they are cheap. They are not done
yet, and this file says so rather than implying otherwise.

## Licence

Everything under `hw/` is Apache-2.0 (`hw/LICENSE`) — permissive, with an explicit patent grant so
that pulling this RTL into silicon does not require a legal negotiation. The Rust crates are MIT.
