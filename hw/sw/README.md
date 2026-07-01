# `hw/sw/` — Q6-08 PS-side host driver

Bare-metal C driver that drives the `uf_axi_wrap` decoder from the Zynq PS. **PS-agnostic** — the
same source runs on the Zynq-7000 (Cortex-A9, Zybo) and Zynq UltraScale+ (Cortex-A53, KV260); the
only per-board difference is the AXI base address (from the Vitis BSP `xparameters.h`).

| file | role |
|------|------|
| `uf_decoder.h` / `uf_decoder.c` | the driver: `uf_init`, `uf_probe` (IDCODE), `uf_decode` (write SYNDROME → START → poll DONE → read CORRECTION/OBS/LATENCY), `uf_latency_ns`. |
| `uf_mmio.h` | MMIO shim the driver goes through (so it links unchanged on board vs host test). |
| `uf_mmio_hw.c` | hardware MMIO (volatile 32-bit) — the on-board image. |
| `main.c` | bare-metal demo: probe, decode example syndromes, print correction/obs/latency vs the ~1 µs budget. |
| `uf_pynq.py` | PYNQ/Python driver — same regmap over `pynq.MMIO`; drives the board over LAN, or runs a board-free software-model self-test. See "Run on the board over LAN" below. |
| `uf_hil.py` | on-board Hardware-in-the-Loop: replays the co-sim Monte-Carlo stream (`hw/cosim_d3.vec`) through the real decoder, checks the on-silicon logical-error rate vs software UF within MC CI, and measures throughput. |
| `test/` | host-side verification (no board) — see below. |

The register map mirrors `../uf_axi_wrap.sv`: `CTRL`(START) / `STATUS`(BUSY,DONE,OBS) / `SYNDROME` /
`CORRECTION` / `LATENCY` / `IDCODE`.

## Verify on the host (no board)

```bash
make -C hw/sw test
```

`test/uf_mmio_model.c` is a software model of the AXI register file backed by the frozen golden
table; `test/test_uf.c` runs the **real driver** over all 256 d=3 syndromes through the modelled
registers and checks every result against the golden, plus the IDCODE probe and the latency-ns
conversion:

```
uf-driver host test: syndromes=256  fails=0  (47 clk @100MHz = 470 ns)
RESULT: PASS (driver protocol verified vs golden over all 256 syndromes; IDCODE ok)
```

This exercises the register protocol end-to-end (offsets, START pulse, DONE poll, OBS bit, reads)
without hardware — the driver object that runs on the board is the one under test.

## Run on the board over LAN (PYNQ — the path actually used)

`uf_pynq.py` is the Python twin of the C driver: same AXI4-Lite regmap and protocol, but it drives
the PL through `pynq.MMIO` after loading the bitstream as a PYNQ overlay — so bring-up is over
SSH/LAN with no JTAG or serial. Board: **Arty Z7-20** on the PYNQ-Z1 v3.1.1 image.

```bash
# on the board (root + XRT env; pynq is in a venv):
sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 uf_pynq.py uf_arty.bit uf_surface_golden.mem
# -> [board] IDCODE ok (0x55460003)
#    uf-pynq driver: syndromes=256  fails=0  worst latency=30 clk = 600 ns @ 50 MHz
#    RESULT: PASS (256/256 syndromes match golden; IDCODE ok)
```

Gotchas (cost real time): pynq lives in `/usr/local/share/pynq-venv`, needs **root**, and needs
`XILINX_XRT=/usr` — a bare `sudo python3` gives `RuntimeError: No Devices Found`. The bitstream is
built by `hw/syn/arty_z7_bd.tcl` (produces `uf_arty.bit` + `uf_arty.hwh`; PYNQ needs both, matching
basenames). Off-board, `python3 uf_pynq.py` (no `.bit`) runs a software-model self-test against the
golden table — same driver logic, no hardware.

## Build for the board (Vitis bare-metal, alternative)

In a Vitis bare-metal application against the PS BSP, compile `main.c`, `uf_decoder.c`, and
`uf_mmio_hw.c`. The base address resolves from the BSP (`XPAR_UF_AXI_WRAP_0_BASEADDR`); override
`UF_BASE_ADDR` / `UF_CLK_HZ` at compile time if the block design differs. Set `UF_CLK_HZ` to the
PL clock (the Q6-05 closed Fmax) so `uf_latency_ns` reports real time against the ~1 µs round budget
(the Q4-03 real-time gate).

**Remaining (needs a board):** flash the bitstream + this app onto Zybo / KV260 and confirm the
host↔PL round-trip + measured per-round latency — closes the board criteria of Q6-01/Q6-02/Q6-08.

## Bring-up is Mac-attached (no Mac-side JTAG)

The boards plug into the **M4 Mac**, but Vivado/Vitis are x86-Linux-only and there is no macOS
Xilinx JTAG/hw_server — so bring-up goes via **SD-boot + serial**, never Mac JTAG:

1. **Build on `openwebgui`** (x86 Linux): bitstream (Vivado) + this bare-metal app, packaged into
   **`BOOT.bin`** (FSBL + bitstream + `app.elf`) for the Zybo; or a PetaLinux/Ubuntu **SD image** for
   the KV260 (the K26 boots Linux on its ARM and self-programs the PL).
2. **`scp`** the boot artifact to the Mac.
3. **Mac writes the microSD** (`dd` / Raspberry Pi Imager / balenaEtcher) and inserts it.
4. The board **self-boots from SD and programs its own PL** — no host JTAG.
5. **Mac ↔ board** over USB-UART (`screen /dev/tty.usbserial-* 115200`); for the KV260 also
   Ethernet/SSH (direct cable or a small switch). `main.c`'s `printf` lands in the serial console.

