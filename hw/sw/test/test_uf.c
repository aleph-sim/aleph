// Q6-08 (host test) — exercise the UF host driver end-to-end against the modelled register file,
// over all 256 d=3 syndromes, checking the driver's results against the golden table. Verifies the
// register protocol (offsets, START pulse, DONE poll, OBS bit, CORRECTION/LATENCY reads) on the host
// with no board. Build+run via `make -C hw/sw test`.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#include "../uf_decoder.h"
#include "uf_mmio_model.h"

#define NSYN 256

int main(int argc, char **argv) {
    const char *golden_path = (argc > 1) ? argv[1] : "../uf_surface_golden.mem";

    // load the golden table: one hex value {obs<<18 | corr} per line (comments start with //).
    uint32_t packed[NSYN];
    int count = 0;
    {
        FILE *f = fopen(golden_path, "r");
        if (!f) { fprintf(stderr, "FAIL: open %s\n", golden_path); return 2; }
        char line[256];
        while (fgets(line, sizeof line, f) && count < NSYN) {
            char *p = line;
            while (*p && (*p == ' ' || *p == '\t')) ++p;
            if (p[0] == '/' || p[0] == '\n' || p[0] == '\0') continue;
            packed[count++] = (uint32_t)strtoul(p, NULL, 16);
        }
        fclose(f);
    }
    if (count != NSYN) { fprintf(stderr, "FAIL: golden has %d entries, expected %d\n", count, NSYN); return 2; }

    uf_model_load_golden(packed, NSYN);

    uf_decoder_t dec;
    uf_init(&dec, uf_model_base(), 100000000u /* 100 MHz */);

    if (!uf_probe(&dec)) { fprintf(stderr, "FAIL: IDCODE probe\n"); return 1; }

    int fails = 0;
    for (int s = 0; s < NSYN; ++s) {
        uf_result_t r;
        if (uf_decode(&dec, (uint32_t)s, &r) != 0) {
            fprintf(stderr, "FAIL s=%d: decode timeout\n", s); ++fails; continue;
        }
        uint32_t want_corr = packed[s] & 0x3FFFFu;
        uint8_t  want_obs  = (packed[s] >> 18) & 1u;
        if (r.correction != want_corr || r.obs_flip != want_obs) {
            fprintf(stderr, "FAIL s=%d: got {obs=%u,corr=0x%05x} want {obs=%u,corr=0x%05x}\n",
                    s, r.obs_flip, r.correction, want_obs, want_corr);
            if (++fails > 10) break;
        }
    }

    // latency conversion sanity: 47 cycles @ 100 MHz = 470 ns.
    uint32_t ns = uf_latency_ns(&dec, 47);
    if (ns != 470) { fprintf(stderr, "FAIL: latency_ns(47@100MHz)=%u, expected 470\n", ns); ++fails; }

    printf("uf-driver host test: syndromes=%d  fails=%d  (47 clk @100MHz = %u ns)\n", NSYN, fails, ns);
    if (fails) { printf("RESULT: FAIL\n"); return 1; }
    printf("RESULT: PASS (driver protocol verified vs golden over all %d syndromes; IDCODE ok)\n", NSYN);
    return 0;
}
