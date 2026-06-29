# Q6-08 — host-side verification of the UF decoder driver (no board).
#   make -C hw/sw test    # build + run the driver against the modelled register file
#
# The on-board image (main.c + uf_mmio_hw.c) is built in Vitis against the PS BSP — see README.md.

CC     ?= cc
CFLAGS ?= -Wall -Wextra -O2

.PHONY: test clean

test:
	$(CC) $(CFLAGS) -I. -Itest uf_decoder.c test/uf_mmio_model.c test/test_uf.c -o test/run_test
	./test/run_test ../uf_surface_golden.mem

clean:
	rm -f test/run_test
