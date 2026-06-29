// Q6-08 — bare-metal on-board demo for the UF decoder (Zynq PS). Builds in Vitis against the BSP;
// this is the image that runs once a board is flashed. The base address comes from the BSP
// `xparameters.h`; override UF_BASE_ADDR / UF_CLK_HZ at compile time if your block design differs.
//
//   Loop: probe IDCODE -> for a few syndromes: decode -> print {correction, obs, latency(ns)} and
//   check against the ~1 µs round budget (the Q4-03 real-time gate).

#include <stdint.h>
#include <stdio.h>

#include "uf_decoder.h"

#ifdef __has_include
#  if __has_include("xparameters.h")
#    include "xparameters.h"
#  endif
#endif

// Base address: prefer the BSP symbol; fall back to a compile-time define.
#ifndef UF_BASE_ADDR
#  ifdef XPAR_UF_AXI_WRAP_0_BASEADDR
#    define UF_BASE_ADDR XPAR_UF_AXI_WRAP_0_BASEADDR
#  else
#    define UF_BASE_ADDR 0x43C00000u   // typical Zynq-7000 GP0 AXI slave; UltraScale+ often 0xA0000000
#  endif
#endif

#ifndef UF_CLK_HZ
#  define UF_CLK_HZ 100000000u          // 100 MHz PL clock (adjust to your design / Q6-05 Fmax)
#endif

#define ROUND_BUDGET_NS 1000u           // surface-code syndrome round ~1 µs

int main(void) {
    uf_decoder_t dec;
    uf_init(&dec, (uintptr_t)UF_BASE_ADDR, UF_CLK_HZ);

    if (!uf_probe(&dec)) {
        printf("UF: IDCODE mismatch at 0x%08lx — check the block design / base address\n",
               (unsigned long)UF_BASE_ADDR);
        return 1;
    }
    printf("UF: decoder present at 0x%08lx, PL clock %u Hz\n",
           (unsigned long)UF_BASE_ADDR, (unsigned)UF_CLK_HZ);

    // A few example d=3 syndromes (detector bit-masks).
    const uint32_t syndromes[] = {0x00, 0x01, 0x09, 0x100, 0xFF};
    const int n = (int)(sizeof(syndromes) / sizeof(syndromes[0]));

    for (int i = 0; i < n; ++i) {
        uf_result_t r;
        if (uf_decode(&dec, syndromes[i], &r) != 0) {
            printf("  syndrome 0x%02x: TIMEOUT waiting for DONE\n", syndromes[i]);
            continue;
        }
        uint32_t ns = uf_latency_ns(&dec, r.latency_cycles);
        printf("  syndrome 0x%02x -> correction 0x%05x  obs=%u  latency=%u clk (%u ns) %s\n",
               syndromes[i], r.correction, r.obs_flip, r.latency_cycles, ns,
               ns <= ROUND_BUDGET_NS ? "[<=1us OK]" : "[OVER BUDGET]");
    }
    return 0;
}
