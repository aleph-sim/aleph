// Q6-08 (host test) — control surface for the modelled AXI register file.
#ifndef UF_MMIO_MODEL_H
#define UF_MMIO_MODEL_H

#include <stdint.h>

uintptr_t uf_model_base(void);                       // fake AXI base to init the driver with
void      uf_model_load_golden(const uint32_t *packed, int n); // packed[s] = {obs<<18 | corr}

#endif
