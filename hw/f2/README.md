# hw/f2 — the decoder as an AWS F2 Amazon FPGA Image (appliance v2-large)

Banked 64/192 relay-BP decoder for the gross bivariate-bicycle code `[[144,12,12]]`, noise prior
p = 0.003, wrapped as an F2 Custom Logic. Measured on an f2.6xlarge: **2040/2040 bit-exact, 913 cycles =
6.09 µs worst case at 150 MHz.** Results and caveats are in `docs/perf/p2-f2-afi.md`.

| image | AGFI | decoder clock | status |
|---|---|---|---|
| 150 MHz | `agfi-0155529c6b08a03d0` | 150 MHz, timing met (WNS +0.031 ns) | **public**, us-east-1 |
| 125 MHz | `agfi-0fe0762a0ce11104c` | 125 MHz, timing met (WNS +0.136 ns) | private (margin build) |

## Run it

On any F2 instance:

```bash
git clone -b f2 https://github.com/aws/aws-fpga.git && cd aws-fpga && source sdk_setup.sh && cd -
sudo fpga-load-local-image -S 0 -I agfi-0155529c6b08a03d0
# the 40-shot golden: from the appliance-v1/-v2 release, or regenerate it
curl -fsSLO https://github.com/aleph-sim/aleph/releases/download/appliance-v2/bp_circ_vectors.txt
sudo python3 hw/f2/sw/bp_f2_ocl.py bp_circ_vectors.txt --max-mhz 150
```

The driver needs nothing but Python 3 and root. It mmaps the device's `resource0`, checks MAGIC and the
geometry, measures the decoder clock against the shell's 250 MHz, and requires every decode to match
the golden. **Any other host code must make every OCL access a single aligned 32-bit load/store.**
Python's `struct.pack_into` on an mmap does not, and it corrupts the stream (`docs/perf/p2-f2-afi.md` §4).

## Register map (OCL BAR0)

| offset | name | access | meaning |
|---|---|---|---|
| 0x00 | MAGIC | RO | `0xA1E9_B0F2` |
| 0x04 | GEOM | RO | `{BP_BANK_W, BP_BANK_V}` = `{64, 192}` |
| 0x08 | STATUS | RO | [0] in_full [1] in_empty [2] out_full [3] out_empty [4] mmcm_locked [5] core_in_reset [6] in_rst_busy [7] out_rst_busy |
| 0x0C | CTRL | RW | [0] soft reset (core + both FIFOs) [1] early_exit |
| 0x10 | PUSH | WO | one syndrome word; 5 per experiment, bit c of the syndrome is word c/32 bit c%32 |
| 0x14 | POP | RO | one result word, or `0xFFFF_FFFF` if none is ready. Result: obs [31:20], valid_flag [19], latency cycles [15:0] |
| 0x18 | CORE_CYC | RO | free-running decoder-clock counter |
| 0x1C | MAIN_CYC | RO | free-running 250 MHz counter |
| 0x20 | CORE_KHZ | RO | decoder clock this image was built for |

## Rebuild it

1. **Stage** the CL directory (copies the shared RTL from `hw/` and generates the 64/192 header):
   `hw/f2/stage.sh <dir>/cl_aleph_bp 64 192`
2. **Gate in xsim** before renting anything (any Vivado 2024.x/2025.x; slow, run detached):
   `cd <dir>/cl_aleph_bp && setsid nohup verif/run_xsim.sh <bp_circ_vectors.txt> > xsim.out 2>&1 &`
   It must print `PASS: 40 golden decodes bit-exact`.
3. **Build** on an EC2 instance with the FPGA Developer AMI (`docs/qec/b2-aws-build-runbook.md` covers
   the AMI, key pair and security group). z1d.6xlarge builds both clocks in parallel in ~2.3 h:
   `run_build.sh /scratch/cl_aleph_bp 125 150` → `/scratch/result_<mhz>/*.Developer_CL.tar`.
   Check the post-route timing report, not just the missing `.VIOLATED` tag.
4. **Create the AFI** from the tarball in S3: `aws ec2 create-fpga-image --input-storage-location
   Bucket=…,Key=…tar --logs-storage-location Bucket=…,Key=logs/` (~30 min). Make it public with
   `aws ec2 modify-fpga-image-attribute --fpga-image-id afi-… --operation-type add --user-groups all`.

Decoder clock: `ALEPH_CORE_MHZ=150` at build time selects the 150 MHz MMCM divide (`design/cl_aleph_bp_defines.vh`);
the default is 125.
