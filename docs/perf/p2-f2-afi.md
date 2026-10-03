# Track P2 — appliance v2-large: 64/192 on AWS F2, 6.09 µs worst case, measured on the FPGA

> Open-silicon program (`docs/qec/open-silicon-program.md`), Track P, Task P2 Step 1b. Built 2026-10-02 on
> AWS with the FPGA Developer AMI (Vivado 2025.2), run 2026-10-03 on an f2.6xlarge. **Measured on the
> FPGA:** the banked 64/192 relay-BP decoder decodes 2040/2040 golden syndromes bit-exactly at 150 MHz,
> 913 cycles every time, **6.09 µs worst case**. The AFI is public: any AWS account with F2 access can
> load `agfi-0155529c6b08a03d0` and rerun it.
>
> **Update 2026-10-03 (§7): interface v2.** A rebuild hardens the OCL interface and adds a batched
> host path over PCIS. On the FPGA it reaches **160,000 decodes/s (6.23 µs per decode)**, which is
> 98 % of what the core can do on its own (6.09 µs). Over OCL the same image does 88,000/s. It also
> gives the utilisation of the integrated design. The v2 image is public: `agfi-073cc3dc45ee25cd8`.

## 1. What was built

Phase B (`docs/perf/q7-02-fullparallel-fpga.md` §9) placed and routed the bare 64/192 core on the VU47P
out of context: 150.4 MHz, 19 % of the part, 6.07 µs. That had no host interface and could not be loaded
anywhere. This step wraps the same core as an F2 Custom Logic (CL) and turns it into an Amazon FPGA Image.

```
host --OCL BAR0, AXI4-Lite, clk_main_a0 250 MHz--> PUSH --xpm_fifo_async--> AXI4-Stream --+
                                                                                          | bp_stream_banked_core
host <--OCL BAR0-- POP <--xpm_fifo_async-- AXI4-Stream <----------------------------------+ (clk_core)
```

- **The decoder is the same RTL as the KV260 and ZCU104 images.** `bp_stream_banked_core` and
  `bp_relay_banked` are copied unchanged out of `hw/`. Only the header is generated for 64/192
  (`circgraph 1 0.003 64 192`).
- **Host interface: OCL only.** The F2 small shell gives the CL no DMA. The host writes the 5 syndrome
  words of an experiment to PUSH and reads one result word from POP. The result word has the same bit
  layout as on the KV260: obs [31:20], valid_flag [19], latency in cycles [15:0]. Register map:
  `hw/f2/design/aleph_bp_ocl.sv`.
- **The decoder clock comes from the CL's own MMCM, not from an AWS clock recipe.** The shell's 100 MHz
  `clk_hbm_ref` × 15 gives a 1500 MHz VCO, divided by 10 for 150 MHz or by 12 for 125 MHz. The frequency
  is fixed in the bitstream, so nothing on the host can program a different one. That rules out the trap
  the ZCU104 build had to engineer around, where PYNQ applied divisors to the wrong PLL
  (`docs/perf/p2-appliance-v2.md` §3). The clock recipes recorded in the AFI manifest (A1/B2/C0/H2) are
  the HDK defaults and drive nothing in this CL.
- **The clock is self-measured.** Free-running counters in both domains (CORE_CYC, MAIN_CYC) give the
  host the real decoder clock against the shell's fixed 250 MHz. The driver prints it, and with
  `--max-mhz` refuses to decode above the closure clock.
- **Latency is counted by the core.** The figure comes from result-word bits [15:0], so PCIe and the OCL
  round trip are not in it.

## 2. Gate before spending money: xsim of the whole front end

Vivado 2024.2 xsim on the EPYC box, with the real XPM FIFO/CDC models and the MMCM unisim model, both
part-independent. The testbench (`hw/f2/verif/tb_aleph_bp_ocl.sv`) drives AXI4-Lite the way the shell
may, with address and data in different cycles.

| geometry | golden decodes | worst latency | clk_core measured | after soft reset |
|---|---|---|---|---|
| 64/192 | **40/40** bit-exact | 913 cycles | 125.00 MHz | 5/5 |
| 16/48 | **40/40** bit-exact | 2085 cycles | 125.00 MHz | 5/5 |

xsim is slow on this design. Elaborating the 64/192 core takes ~30 min, and the 40 decodes take a few
hours of wall clock, because the core is one large combinational block evaluated every cycle. Run it
detached (`hw/f2/verif/run_xsim.sh`).

## 3. Build: both clocks close timing

`aws_build_dcp_from_cl.py --cl cl_aleph_bp --mode small_shell` (HDK 2.3.4, shell 0x10212415),
default directives (`SSI_SpreadLogic_high` / `AggressiveExplore`). Both clocks were built in parallel on
one z1d.6xlarge (`hw/f2/aws/run_build.sh`).

| clock | WNS (setup) | WHS (hold) | worst group | build time | AFI |
|---|---|---|---|---|---|
| 150 MHz | **+0.031 ns** | +0.010 ns | `clk_core_unbuf` | 139 min | `agfi-0155529c6b08a03d0` (**public**) |
| 125 MHz | +0.136 ns | +0.010 ns | `clk_core_unbuf` | 92 min | `agfi-0fe0762a0ce11104c` (private) |

- **The timing report analyses the decoder clock at its real period** (`clk_core_unbuf`, 8.000 ns at
  125 MHz). A pass is therefore not an artefact of an unconstrained clock.
- The other groups have ample slack: `clk_main_a0` +0.71 / +0.11 ns, `clk_hbm_ref` +8.8 ns.
- At 150 MHz the router passed through WNS −0.056 ns before closing at +0.031 ns, which matches Phase B's
  150.4 MHz OOC Fmax. **150 MHz is this core's ceiling on the VU47P with default directives.** 125 MHz is
  the margin build.
- The CRITICAL WARNINGs in the build log come from the shell's DDR timing constraints referencing a DDR
  controller this CL does not instantiate (`EN_DDR = 0`). This is expected for a DDR-less CL.
- The HDK flow writes timing reports but no utilisation report. Phase B's post-route numbers for the same
  core still apply: 247,434 CLB LUTs = 19.0 % of the VU47P, 593 DSP, 0 BRAM.

AFI generation took ~30 min per image. Cost: build instance ~2.3 h ≈ $5.

## 4. On the FPGA

f2.6xlarge (us-east-1a, AMD EPYC 7R13 host, kernel 6.8.0-1021-aws), `aws-fpga` SDK (`sdk_setup.sh`),
AFI loaded with `fpga-load-local-image`; the device enumerates as `1d0f:f0a1`. Driver:
`hw/f2/sw/bp_f2_ocl.py`. Two vector sets: the standard 40-shot golden (`circvectors 1 0.003 40 2024`)
and a fresh 2000-shot set (`circvectors 1 0.003 2000 7777 0.003`).

| AFI | decoder clock measured | 40 golden | 2000 golden | worst latency (core-counted) |
|---|---|---|---|---|
| 150 MHz | **150.000 MHz** | **40/40** | **2000/2000** | **913 cycles = 6.09 µs** |
| 125 MHz | **125.000 MHz** | **40/40** | **2000/2000** | 913 cycles = 7.30 µs |

- Every decode takes 913 cycles: full schedule, `early_exit = 0`, as in the v1 and v2 images.
- **The host round trip over OCL MMIO is ~15 µs per decode** (5 posted writes + POP polls). This is a
  property of the host loop, not of the decoder. Batching through PCIS/DMA would hide it, and that is not
  part of this step.
- Clock guard negative check: `--max-mhz 100` on the 125 MHz image prints the FAIL message and exits 1
  without decoding.
- Test instance ~0.5 h ≈ $1. Both instances were terminated and nothing is left running.

### A host-driver bug that only real hardware showed

The first silicon run decoded **1/40** correctly, with the right clock and the right latency. The cause:
`struct.pack_into` / `unpack_from` on the BAR mmap let Python move the 4 bytes in pieces, and on F2 that
reaches the CL as several narrower AXI-Lite transactions. Each PUSH became several FIFO words, and each
POP read popped extra results. Measured directly: 5 PUSH writes through `struct` produced 3 decodes'
worth of beats, and 5 writes through `ctypes.c_uint32` produced exactly 1. The driver now does every
access as one aligned 32-bit load or store, and that fix gives the 2040/2040 above. xsim could not catch
this, because its AXI master only ever issues full-word transactions.

The RTL slave ignores `wstrb`. That is why a partial write turned into a whole bogus beat instead of
being dropped. Rejecting PUSH writes with `wstrb != 4'hF` is a cheap hardening for the next rebuild.

## 5. Verdict

- **Step 1b done.** A public AFI of 64/192 on the VU47P decodes bit-exactly on the FPGA at 150 MHz,
  6.09 µs worst case. That is 2.6× faster than the v1 image as deployed (22.94 µs) and 1.4× faster than
  v2 on a ZCU104 (8.29 µs, not yet run on a board).
- **Phase B's out-of-context projection held.** It predicted 6.07 µs at 150.4 MHz. With the shell
  integrated, the number is 6.09 µs at 150 MHz.
- **Still not sub-microsecond.** The geometry and clock that would get there are unchanged from Phase B
  §13: the ASIC case, Task B3.
- **Not done here, done in §7:** utilisation from the integrated build; a PCIS batch path; the `wstrb`
  hardening. Also still open: making the 125 MHz image public (one call, if wanted).

## 6. Reproduce

Run the published image (any F2 instance, from the `aws-fpga` repo on branch `f2`):

```bash
source sdk_setup.sh
sudo fpga-load-local-image -S 0 -I agfi-0155529c6b08a03d0
sudo python3 hw/f2/sw/bp_f2_ocl.py bp_circ_vectors.txt --max-mhz 150
```

Rebuild it: see `hw/f2/README.md` (stage → xsim → `run_build.sh` on the FPGA Developer AMI →
`create-fpga-image`).

## 7. Interface v2: hardened OCL, a PCIS batch path, integrated utilisation (2026-10-03)

This was one rebuild for the three follow-ups above. The decoder core and the clocking are unchanged.

### What changed

- **`wstrb`.** A register write acts only if all four byte strobes are set. Any other write changes
  nothing and sets the sticky `STATUS.err_strb`, so a split host store can no longer become bogus PUSH
  beats (§4).
- **Reads have no side effects.** Every F2 BAR is prefetchable (`AWS_Fpga_Pcie_Memory_Map.md`), so POP
  became a *peek*: it returns the oldest result together with a 2-bit seq (bits [17:16]). An explicit
  POP_ACK write carrying that seq pops it. An ack with any other seq is a no-op, so a repeated or stale
  ack cannot pop twice.
- **Debug registers.** PUSH_WORDS, RESULTS, PCIS_ACC, PCIS_DROP, sticky error bits, VERSION = 2 and
  SLOTS. The v1 image reads `0xDEAD_BEEF` at VERSION, and the driver uses that to keep talking to it.
- **The PCIS batch path** (`hw/f2/design/aleph_pcis_ring.sv`, BAR4). The small shell still has no DMA
  (`xdma_shell` is still "coming soon" upstream), so a batch has to arrive as host stores. The design
  assumes those stores are unreliable in the ways a write-combining mapping allows:
  - The host writes each experiment as one 64-byte store into one of 128 slots: the syndrome plus a
    32-bit tag. Each slot keeps a byte mask and commits only when all 32 low bytes have arrived, so a
    store that the CPU splits or reorders still yields exactly one experiment.
  - Results land in place in a `{tag, result}` ring. Reading the ring has no side effects, so
    prefetching or speculative reads cannot lose a result.
  - The CL never back-pressures PCIS, which the shell times out after 8 µs. Flow control is the host's:
    at most one experiment in flight per slot.
  - Protocol: `hw/f2/README.md`.
- **Reports.** `run_build.sh` opens the routed checkpoint on the build box and writes utilisation,
  clock, CDC and timing-summary reports (`build/scripts/aleph_reports.tcl`).

### Gates before the build

- **PCIS path alone, xsim with a stand-in decoder** (`verif/run_ring_xsim.sh`).
  - 601 experiments per seed, 8/8 seeds pass. The traffic covers whole 64-byte stores, halves, 4-byte
    pieces interleaved and reordered across slots, two-beat bursts, narrow and burst ring reads, and
    refused writes.
  - 4/4 injected bugs are caught: commit without a full mask, wrong ring lane, wrong burst address
    step, side-effecting input-window reads.
- **The PCIS path alone, OOC on xczu7ev -2 at 250 MHz:** WNS +0.444 ns.
- **The whole front end in xsim with the real core** (`verif/run_xsim.sh`).
  - At 64/192 and 150 MHz and at 16/48 and 125 MHz: the new checks pass (VERSION / SLOTS, a
    partial-strobe PUSH pushes nothing, peek is stable, a wrong-seq ack pops nothing), then 40/40
    golden over OCL with peek/ack.
  - Then the PCIS phase: the same 40 vectors as slot stores (every third split in two) read back from
    the ring, 40/40, counters exact, then 5/5 after a soft reset back to OCL mode. **PASS at both
    geometries** (64/192: worst 913 cycles at 150.02 MHz; 16/48: 2085 cycles at 125 MHz). The 64/192 run
    took ~6 h of wall clock.

### Build

Same flow and directives as §3: one z1d.6xlarge, both clocks in parallel.

| clock | WNS (setup) | WHS (hold) | build time | AFI | AGFI |
|---|---|---|---|---|---|
| 150 MHz | **+0.004 ns** | +0.010 ns | 190 min | `afi-078824ae6a563c721` | `agfi-073cc3dc45ee25cd8` (**public**) |
| 125 MHz | +0.169 ns | +0.010 ns | 100 min | `afi-03c9f8dd6777f3a2c` | `agfi-050f06c5308709a6d` (private) |

- **150 MHz closed with 4 ps to spare** (v1: 31 ps). On the way, the router passed through −0.41 ns
  and then −0.003 ns before it closed. This confirms §3: 150 MHz is the ceiling of this core on the
  VU47P with default directives. 125 MHz remains the margin build.

**Utilisation of the integrated design** (routed 125 MHz build, VU47P):

| cell | CLB LUTs | FFs | BRAM | DSP |
|---|---|---|---|---|
| whole CL (`WRAPPER/CL`) | **245,315 (18.85 %)** | 77,733 (2.99 %) | 12 RAMB36 + 2 RAMB18 | **593 (6.57 %)** |
| decoder core (`u_core`) | 240,663 | 72,408 | 0 | 593 |
| PCIS batch path (`u_pcis`) | 1,721 | 4,574 | 12 RAMB36 | 0 |
| two async FIFOs + clock counters + registers | 410 | 750 | 2 RAMB18 | 0 |
| CL top level (shell tie-offs, `sh_ddr` stub) | 2,525 | 1 | 0 | 0 |

- The 150 MHz build: 248,124 LUTs (19.06 %).
- Phase B's out-of-context figure for the bare core was 247,434 LUTs and 593 DSP. In context the core
  is the same size to within 3 %.
- **The interface costs about 2 % of the CL.**

**CDC report.** Every CRITICAL entry with a visible path has the shell's encrypted logic on both ends
(`<hidden>`), except one:

- CDC-10 flags the OR of reset sources (`~rst_main_n | soft_rst | ~mmcm_locked`) in front of the core's
  `xpm_cdc_async_rst`. The worst a glitch there can do is assert the core's asynchronous reset once
  more, while a reset input is changing anyway.
- The CDC-15 warnings sit inside the XPM async FIFOs, as expected.

### On the FPGA

Same f2.6xlarge type and SDK as §4. Driver `hw/f2/sw/bp_f2_ocl.py` (now with `--pcis`, `--repeat`,
`--window`).

**Correctness, both v2 images:**

| run | 150 MHz | 125 MHz |
|---|---|---|
| OCL (peek + ack), 40 golden | **40/40** | **40/40** |
| OCL, 2000 golden | **2000/2000** | **2000/2000** |
| OCL, 2000 × 5 | **10,000/10,000** | **10,000/10,000** |
| PCIS, 40 and 2000 golden | **40/40, 2000/2000** | **40/40, 2000/2000** |
| PCIS, 2000 × 10 at each window 1 / 2 / 4 / 8 / 16 / 64 / 128 | **7 × 20,000/20,000** | **7 × 20,000/20,000** |

- On every run the counters agree exactly: PUSH_WORDS = 5 × decodes, RESULTS = PCIS_ACC = decodes,
  PCIS_DROP = 0, no error bit set.
- `fpga-describe-local-image -M` shows `dma-pcis-timeout=0` and `ocl-slave-timeout=0`.
- The clock guard still works: `--max-mhz 100` → FAIL, exit 1.
- The new driver on the **v1 public image** (`agfi-0155529c6b08a03d0`) falls back to pop-on-read and
  passes 2000/2000. Asked for `--pcis`, it refuses with exit 1.

**Host throughput (150 MHz image, 913 cycles per decode = 6.09 µs of core time):**

| path | µs per decode | decodes/s | fraction of the core's own rate |
|---|---|---|---|
| OCL, one experiment at a time | 11.36 | 88,000 | 54 % |
| PCIS, window 1 | 10.69 | 93,600 | 57 % |
| PCIS, window 2 | 6.30 | 158,700 | 97 % |
| **PCIS, window ≥ 4** | **6.23** | **160,500** | **98 %** |

- **With two or more experiments in flight, the core is the bottleneck.**
  - The remaining 0.14 µs per decode (21 cycles) is the core taking in its 5 input words and the
    result word crossing the clock domains. The core is not pipelined across experiments
    (`bp_stream_banked_core.sv`), so 160,000/s is the ceiling of this image whatever the host does.
  - At 125 MHz the same pattern holds: 7.46 µs per decode against 7.30 µs of core time.
- **PCIS gives 1.8× the throughput of OCL.**
  - The OCL path is faster than in §4: 11.4 µs against 15.1 µs. That gain is the driver's: it packs the
    syndrome words once, up front. The same driver on the v1 image also measures 10.8 µs.
  - The 40-vector PCIS runs show ~44 µs per decode, because the one-off mmap and setup time is spread
    over only 40 decodes.
- **Latency is unchanged:** 913 cycles, 6.09 µs worst case at 150 MHz.

**Cost.** The build instance ran ~3.2 h (≈ $7) and the test instance ~0.6 h (≈ $1.2). Both are
terminated. The S3 bucket was deleted after the DCP tarballs and AFI logs were copied off. Nothing
billable is left.

### Verdict

- **All three follow-ups are done.**
- **Throughput is now set by the decoder, not the host link:** 160,000 decodes/s on one F2 FPGA,
  bit-exact.
- **Two levers remain for more throughput, and neither is in the interface.** Pipelining the core
  across experiments would overlap one decode with the next experiment's input. Instantiating several
  cores would fit easily: one core is 19 % of the device.
- **Latency is still 6.09 µs:** the sub-microsecond path is unchanged (§5).

