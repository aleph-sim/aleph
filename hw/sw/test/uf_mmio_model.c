// Q6-08 (host test) — software model of the uf_axi_wrap AXI4-Lite register file, backed by the
// frozen golden table. Linked in place of the hardware MMIO so the driver's register protocol
// (write SYNDROME -> START -> poll DONE -> read CORRECTION/OBS/LATENCY) is exercised end-to-end on
// the host, no board required. Models the register *semantics*, not cycle timing (decode is instant).

#include "../uf_decoder.h"
#include "../uf_mmio.h"
#include "uf_mmio_model.h"

#define UF_MODEL_BASE   0x40000000u
#define UF_MODEL_NSYN   256u           // d=3: 8 detector bits
#define UF_MODEL_LAT    47u            // d=3 worst-case decode latency (cycles), as Q6-09

static uint32_t g_corr[UF_MODEL_NSYN];
static uint8_t  g_obs [UF_MODEL_NSYN];
static uint32_t g_syndrome;
static uint32_t g_cur_corr;
static uint8_t  g_cur_obs;
static int      g_done;

uintptr_t uf_model_base(void) { return UF_MODEL_BASE; }

void uf_model_load_golden(const uint32_t *packed, int n) {
    for (int s = 0; s < n && s < (int)UF_MODEL_NSYN; ++s) {
        g_corr[s] = packed[s] & 0x3FFFFu;   // correction = bits [17:0]
        g_obs[s]  = (packed[s] >> 18) & 1u; // obs_flip   = bit 18
    }
    g_syndrome = 0; g_cur_corr = 0; g_cur_obs = 0; g_done = 0;
}

void uf_mmio_write(uintptr_t addr, uint32_t value) {
    uint32_t off = (uint32_t)(addr - UF_MODEL_BASE);
    switch (off) {
    case UF_REG_SYNDROME:
        g_syndrome = value & (UF_MODEL_NSYN - 1u);
        break;
    case UF_REG_CTRL:
        if (value & UF_CTRL_START) {       // instant decode from the golden table
            g_cur_corr = g_corr[g_syndrome];
            g_cur_obs  = g_obs[g_syndrome];
            g_done     = 1;
        }
        break;
    default:
        break;
    }
}

uint32_t uf_mmio_read(uintptr_t addr) {
    uint32_t off = (uint32_t)(addr - UF_MODEL_BASE);
    switch (off) {
    case UF_REG_STATUS:
        return (g_done ? UF_STATUS_DONE : 0u) | (g_cur_obs ? UF_STATUS_OBS : 0u);
    case UF_REG_CORRECTION: return g_cur_corr;
    case UF_REG_LATENCY:    return UF_MODEL_LAT;
    case UF_REG_IDCODE:     return UF_IDCODE_EXPECTED;
    default:                return 0u;
    }
}
