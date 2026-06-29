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

## Build for the board (Vitis, pending hardware)

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

