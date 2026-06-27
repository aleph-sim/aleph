# Q6 (sim) — build + run the Verilator simulations of the decoder RTL.
#
#   make            # build + run both decoders (regenerating their reference tables)
#   make d3         # Q6-01: d=3 surface-code LUT decoder
#   make rep        # Q6-02: repetition-code Union-Find decoder
#   make clean
#
# Requires: verilator (brew install verilator) and the Rust workspace (for the reference tables).

OBJ_DIR   := obj_dir
VERILATOR ?= verilator
CARGO     := cargo run --release -q -p aleph-qec --example

.PHONY: all d3 rep clean

all: d3 rep

d3:
	cd .. && $(CARGO) qec_d3_lut_table > hw/surface_d3_lut.mem
	$(VERILATOR) --cc --exe --build -Wall --top-module surface_d3_decoder \
		surface_d3_decoder.sv tb_surface_d3.cpp -o sim_d3
	./$(OBJ_DIR)/sim_d3 surface_d3_lut.mem

rep:
	cd .. && $(CARGO) qec_rep_uf_vectors > hw/rep_uf_vectors.mem
	$(VERILATOR) --cc --exe --build -Wall --top-module uf_rep_decoder \
		uf_rep_decoder.sv tb_uf_rep.cpp -o sim_rep
	./$(OBJ_DIR)/sim_rep rep_uf_vectors.mem

clean:
	rm -rf $(OBJ_DIR)
