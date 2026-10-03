# Track P2 — appliance v2-large: 64/192 on AWS F2, 6.09 µs worst case, measured on the FPGA

> Open-silicon program (`docs/qec/open-silicon-program.md`), Track P, Task P2 Step 1b. Built 2026-10-02 on
> AWS with the FPGA Developer AMI (Vivado 2025.2), run 2026-10-03 on an f2.6xlarge. **Measured on the
> FPGA:** the banked 64/192 relay-BP decoder decodes 2040/2040 golden syndromes bit-exactly at 150 MHz,
> 913 cycles every time, **6.09 µs worst case**. The AFI is public: any AWS account with F2 access can
> load `agfi-0155529c6b08a03d0` and rerun it.

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
- **Not done:** utilisation from the integrated build; a DMA/PCIS batch path; the `wstrb` hardening; and
  making the 125 MHz image public (it can be made public with one call if wanted).

## 6. Reproduce

Run the published image (any F2 instance, from the `aws-fpga` repo on branch `f2`):

```bash
source sdk_setup.sh
sudo fpga-load-local-image -S 0 -I agfi-0155529c6b08a03d0
sudo python3 hw/f2/sw/bp_f2_ocl.py bp_circ_vectors.txt --max-mhz 150
```

Rebuild it: see `hw/f2/README.md` (stage → xsim → `run_build.sh` on the FPGA Developer AMI →
`create-fpga-image`).
