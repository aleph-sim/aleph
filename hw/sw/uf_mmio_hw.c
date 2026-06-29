// Q6-08 — hardware MMIO implementation (bare-metal on the Zynq PS). Compiled into the on-board image
// only; the host test substitutes the modelled MMIO instead.

#include "uf_mmio.h"

uint32_t uf_mmio_read(uintptr_t addr) {
    return *(volatile uint32_t *)addr;
}

void uf_mmio_write(uintptr_t addr, uint32_t value) {
    *(volatile uint32_t *)addr = value;
}
