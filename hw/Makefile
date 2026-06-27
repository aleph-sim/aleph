# Q6-01 (sim) — build + run the Verilator simulation of the d=3 surface-code LUT decoder.
#
#   make            # regenerate the oracle LUT, build with Verilator, run the testbench
#   make sim        # build + run (assumes surface_d3_lut.mem exists)
#   make lut        # regenerate surface_d3_lut.mem from the Rust Union-Find decoder
#   make clean
#
# Requires: verilator (brew install verilator) and the Rust workspace (for `make lut`).

TOP      := surface_d3_decoder
OBJ_DIR  := obj_dir
MEM      := surface_d3_lut.mem
VERILATOR ?= verilator

.PHONY: all sim lut clean
all: lut sim

lut:
	cd .. && cargo run --release -q -p aleph-qec --example qec_d3_lut_table > hw/$(MEM)

sim: $(MEM)
	$(VERILATOR) --cc --exe --build -Wall --top-module $(TOP) \
		$(TOP).sv tb_surface_d3.cpp -o sim_$(TOP)
	./$(OBJ_DIR)/sim_$(TOP) $(MEM)

clean:
	rm -rf $(OBJ_DIR)
