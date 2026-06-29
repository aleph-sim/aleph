// Q6-08 — MMIO shim for the UF driver.
//
// The driver does all register access through these two functions. On hardware they are simple
// volatile 32-bit accesses (`uf_mmio_hw.c`); the host test swaps in a software model of the AXI
// register file (`test/uf_mmio_model.c`) so the driver's protocol is verified without a board.

#ifndef UF_MMIO_H
#define UF_MMIO_H

#include <stdint.h>

uint32_t uf_mmio_read(uintptr_t addr);
void     uf_mmio_write(uintptr_t addr, uint32_t value);

#endif // UF_MMIO_H
