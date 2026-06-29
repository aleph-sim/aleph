// Q6-08 — UF decoder host driver implementation. Pure C, no board-specific includes; all register
// access goes through the MMIO shim (`uf_mmio.h`), so this object links unchanged into a bare-metal
// image (real MMIO) or the host test (modelled MMIO).

#include "uf_decoder.h"
#include "uf_mmio.h"

#define UF_DEFAULT_POLL_LIMIT 100000u

void uf_init(uf_decoder_t *d, uintptr_t base, uint32_t clk_hz) {
    d->base = base;
    d->clk_hz = clk_hz;
    d->poll_limit = UF_DEFAULT_POLL_LIMIT;
}

int uf_probe(const uf_decoder_t *d) {
    return uf_mmio_read(d->base + UF_REG_IDCODE) == UF_IDCODE_EXPECTED;
}

int uf_decode(const uf_decoder_t *d, uint32_t syndrome, uf_result_t *out) {
    uf_mmio_write(d->base + UF_REG_SYNDROME, syndrome);
    uf_mmio_write(d->base + UF_REG_CTRL, UF_CTRL_START);

    for (uint32_t i = 0; i < d->poll_limit; ++i) {
        uint32_t status = uf_mmio_read(d->base + UF_REG_STATUS);
        if (status & UF_STATUS_DONE) {
            out->obs_flip       = (status & UF_STATUS_OBS) ? 1u : 0u;
            out->correction     = uf_mmio_read(d->base + UF_REG_CORRECTION);
            out->latency_cycles = (uint16_t)uf_mmio_read(d->base + UF_REG_LATENCY);
            return 0;
        }
    }
    return -1; // timed out waiting for DONE
}

uint32_t uf_latency_ns(const uf_decoder_t *d, uint16_t cycles) {
    if (d->clk_hz == 0) return 0;
    // ns = cycles * 1e9 / clk_hz, computed in 64-bit to avoid overflow.
    return (uint32_t)(((uint64_t)cycles * 1000000000ull) / d->clk_hz);
}
