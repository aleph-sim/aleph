# hw/f2 — the decoder as an AWS F2 Amazon FPGA Image (appliance v2-large)

Banked 64/192 relay-BP decoder for the gross bivariate-bicycle code `[[144,12,12]]`, noise prior
p = 0.003, wrapped as an F2 Custom Logic. Measured on an f2.6xlarge: **2040/2040 bit-exact, 913 cycles =
6.09 µs worst case at 150 MHz.** Results and caveats are in `docs/perf/p2-f2-afi.md`.

| image | interface | AGFI | decoder clock | status |
|---|---|---|---|---|
| 150 MHz v2 | v2: OCL + PCIS batch | `agfi-073cc3dc45ee25cd8` | 150 MHz, timing met (WNS +0.004 ns) | private |
| 125 MHz v2 | v2: OCL + PCIS batch | `agfi-050f06c5308709a6d` | 125 MHz, timing met (WNS +0.169 ns) | private (margin build) |
| 150 MHz | v1: OCL only | `agfi-0155529c6b08a03d0` | 150 MHz, timing met (WNS +0.031 ns) | **public**, us-east-1 |
| 125 MHz | v1: OCL only | `agfi-0fe0762a0ce11104c` | 125 MHz, timing met (WNS +0.136 ns) | private (margin build) |

On v2, `--pcis` reaches 160,000 decodes/s (6.23 µs per decode, 98 % of the core's own rate), against
88,000/s over OCL (`docs/perf/p2-f2-afi.md` §7).

## Run it

On any F2 instance:

```bash
git clone -b f2 https://github.com/aws/aws-fpga.git && cd aws-fpga && source sdk_setup.sh && cd -
sudo fpga-load-local-image -S 0 -I agfi-0155529c6b08a03d0
# the 40-shot golden: from the appliance-v1/-v2 release, or regenerate it
curl -fsSLO https://github.com/aleph-sim/aleph/releases/download/appliance-v2/bp_circ_vectors.txt
sudo python3 hw/f2/sw/bp_f2_ocl.py bp_circ_vectors.txt --max-mhz 150
```

The driver needs nothing but Python 3 and root. It mmaps the device's `resource0` (and `resource4` /
`resource4_wc` with `--pcis`), checks MAGIC and the geometry, measures the decoder clock against the
shell's 250 MHz, and requires every decode to match the golden. It reads the interface VERSION and
speaks to both the first public image (v1) and the hardened one (v2). Options: `--pcis` (batched path,
v2 only), `--repeat R` (run the vectors R times, for a throughput number), `--window W` (PCIS
experiments in flight, default 64).

**Any other host code must make every OCL access a single aligned 32-bit load/store.** On v1 a split
store corrupts the stream: Python's `struct.pack_into` on an mmap splits them (`docs/perf/p2-f2-afi.md`
§4). v2 ignores a partial-strobe write and flags it in STATUS, but it still can't rebuild the data.

## Register map (OCL BAR0)

Interface v2 (v1 = the first public image: no registers past 0x20, and POP pops on read).

| offset | name | access | meaning |
|---|---|---|---|
| 0x00 | MAGIC | RO | `0xA1E9_B0F2` |
| 0x04 | GEOM | RO | `{BP_BANK_W, BP_BANK_V}` = `{64, 192}` |
| 0x08 | STATUS | RO | [0] in_full [1] in_empty [2] out_full [3] out_empty [4] mmcm_locked [5] core_in_reset [6] in_rst_busy [7] out_rst_busy; sticky: [8] err_strb (partial-strobe write ignored) [9] err_push (PUSH dropped) [10] err_pcis (PCIS write refused); live: [11] ring_clearing [12] pcis_mode |
| 0x0C | CTRL | RW | [0] soft reset (core, FIFOs, counters, PCIS path) [1] early_exit [2] pcis_mode [3] clear_err (write-1 pulse) |
| 0x10 | PUSH | WO | one syndrome word (OCL mode); 5 per experiment, bit c of the syndrome is word c/32 bit c%32 |
| 0x14 | POP | RO | **peek** at the oldest result, or `0xFFFF_FFFF` if none. Result: obs [31:20], valid_flag [19], seq [17:16], latency cycles [15:0] |
| 0x18 | CORE_CYC | RO | free-running decoder-clock counter |
| 0x1C | MAIN_CYC | RO | free-running 250 MHz counter |
| 0x20 | CORE_KHZ | RO | decoder clock this image was built for |
| 0x24 | POP_ACK | WO | [1:0] = the seq just peeked: drops that result. A stale or repeated ack is a no-op |
| 0x28 | PUSH_WORDS | RO | words into the decoder since reset (either path) |
| 0x2C | RESULTS | RO | results out of the decoder since reset (either path); seq = RESULTS[1:0] |
| 0x30 | PCIS_ACC | RO | experiments committed through PCIS since reset |
| 0x34 | PCIS_DROP | RO | PCIS write beats refused since reset |
| 0x38 | VERSION | RO | 2 |
| 0x3C | SLOTS | RO | PCIS experiment slots (128) |

Why peek + ack: every F2 BAR is prefetchable (`hdk/docs/AWS_Fpga_Pcie_Memory_Map.md`), so no read may
have a side effect.

## Batched path (PCIS BAR4, v2)

Set `CTRL.pcis_mode` together with a soft reset (write `0x5`, then `0x4`). Then:

- **Experiment k** goes to slot `s = k mod 128`: a 64-byte store at BAR4 offset `64·s`, bytes 0–19 the
  five syndrome words (same layout as PUSH), bytes 28–31 a 32-bit tag (anything but `0xFFFF_FFFF`),
  the rest ignored. Store it through `resource4_wc` (write-combining) so it usually leaves the CPU as one
  64-byte PCIe write. It doesn't have to: the CL collects each slot's 32 low bytes from any number of
  pieces, in any order, and commits the slot when all of them are in.
- **Its result** lands at BAR4 offset `0x1_0000 + 8·s` as `{tag[63:32], result word[31:0]}` (same result
  word as POP, seq bits 0). One 64-byte read returns eight slots. Reads have no side effects. After a
  reset every entry reads all-ones. Poll until the entry's tag equals k's tag.
- **Flow control** is the host's: reuse slot s only after its result is back, so at most 128 experiments
  are in flight. The CL never stalls PCIS (the shell times a PCIS transaction out after 8 µs).

## Rebuild it

1. **Stage** the CL directory (copies the shared RTL from `hw/` and generates the 64/192 header):
   `hw/f2/stage.sh <dir>/cl_aleph_bp 64 192`
2. **Gate in xsim** before renting anything (any Vivado 2024.x/2025.x). First the PCIS path alone with a
   stand-in decoder (about a minute): `verif/run_ring_xsim.sh [seed]`, must print `PASS: 601 experiments`.
   Then the whole front end with the real core (slow, run detached):
   `cd <dir>/cl_aleph_bp && setsid nohup verif/run_xsim.sh <bp_circ_vectors.txt> [150] > xsim.out 2>&1 &`
   It must print `PASS: 40 golden decodes bit-exact through OCL AXI-Lite and 40 through PCIS`.
3. **Build** on an EC2 instance with the FPGA Developer AMI (`docs/qec/b2-aws-build-runbook.md` covers
   the AMI, key pair and security group). z1d.6xlarge builds both clocks in parallel in ~2.3 h:
   `run_build.sh /scratch/cl_aleph_bp 125 150` → `/scratch/result_<mhz>/*.Developer_CL.tar`, plus
   utilisation / clock / CDC reports of the routed design in `result_<mhz>/reports/` (only the AWS
   build box's Vivado can open a VU47P checkpoint). Check the post-route timing report, not just the
   missing `.VIOLATED` tag.
4. **Create the AFI** from the tarball in S3: `aws ec2 create-fpga-image --input-storage-location
   Bucket=…,Key=…tar --logs-storage-location Bucket=…,Key=logs/` (~30 min). Make it public with
   `aws ec2 modify-fpga-image-attribute --fpga-image-id afi-… --operation-type add --user-groups all`.

Decoder clock: `ALEPH_CORE_MHZ=150` at build time selects the 150 MHz MMCM divide (`design/cl_aleph_bp_defines.vh`);
the default is 125.
