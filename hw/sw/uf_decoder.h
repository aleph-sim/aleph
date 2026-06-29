// Q6-08 — PS-side host driver for the surface-code UF decoder (uf_axi_wrap).
//
// Bare-metal C, PS-agnostic: the same source runs on the Zynq-7000 (Cortex-A9) and Zynq
// UltraScale+ (Cortex-A53) PS — it only does 32-bit MMIO to the AXI4-Lite slave, so the sole
// per-board difference is the base address (from the Vitis BSP `xparameters.h`). The register map
// mirrors `hw/uf_axi_wrap.sv` exactly.

#ifndef UF_DECODER_H
#define UF_DECODER_H

#include <stdint.h>

// ---- AXI4-Lite register map (byte offsets from the slave base) ----
#define UF_REG_CTRL        0x00u   // [W]  bit0 START (self-clearing)
#define UF_REG_STATUS      0x04u   // [R]  BUSY/DONE/OBS_FLIP
#define UF_REG_SYNDROME    0x08u   // [RW] syndrome bits
#define UF_REG_CORRECTION  0x0Cu   // [R]  correction bits
#define UF_REG_LATENCY     0x10u   // [R]  last decode latency in cycles
#define UF_REG_IDCODE      0x14u   // [R]  0x5546_0003

#define UF_CTRL_START      (1u << 0)
#define UF_STATUS_BUSY     (1u << 0)
#define UF_STATUS_DONE     (1u << 1)
#define UF_STATUS_OBS      (1u << 2)

#define UF_IDCODE_EXPECTED 0x55460003u

typedef struct {
    uintptr_t base;       // AXI4-Lite slave base address
    uint32_t  clk_hz;     // PL clock (for latency cycles -> ns); 0 = unknown
    uint32_t  poll_limit; // max STATUS polls before giving up
} uf_decoder_t;

typedef struct {
    uint32_t correction;     // correction bit-mask
    uint8_t  obs_flip;       // predicted logical flip
    uint16_t latency_cycles; // PL-reported decode latency
} uf_result_t;

// Initialise the handle. `base` comes from the BSP (e.g. XPAR_UF_AXI_WRAP_0_BASEADDR).
void uf_init(uf_decoder_t *d, uintptr_t base, uint32_t clk_hz);

// Read IDCODE; returns true iff it matches (a bring-up sanity check).
int uf_probe(const uf_decoder_t *d);

// Decode one syndrome: write SYNDROME, pulse START, poll DONE, read results.
// Returns 0 on success, -1 on poll timeout.
int uf_decode(const uf_decoder_t *d, uint32_t syndrome, uf_result_t *out);

// Convert a PL-reported latency (cycles) to nanoseconds given the handle's clk_hz (0 if unknown).
uint32_t uf_latency_ns(const uf_decoder_t *d, uint16_t cycles);

#endif // UF_DECODER_H
