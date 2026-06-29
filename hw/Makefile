# Q6 (sim) — build + run the Verilator simulations of the decoder RTL.
#
#   make            # build + run all decoders (regenerating their reference tables)
#   make d3         # Q6-01: d=3 surface-code LUT decoder
#   make rep        # Q6-02: repetition-code Union-Find decoder
#   make surf       # Q6-02: d=3 surface-code Union-Find decoder (2-D graph)
#   make clean
#
# Requires: verilator (brew install verilator) and the Rust workspace (for the reference tables).

OBJ_DIR   := obj_dir
VERILATOR ?= verilator
CARGO     := cargo run --release -q -p aleph-qec --example

.PHONY: all d3 rep surf surf-d5 surf-d7 clean

all: d3 rep surf

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

surf:
	cd .. && $(CARGO) qec_surface_uf_graph -- graph  > hw/uf_surface_graph.svh
	cd .. && $(CARGO) qec_surface_uf_graph -- oracle > hw/uf_surface_oracle.mem
	$(VERILATOR) --cc --exe --build -Wall --top-module uf_surface_decoder \
		uf_surface_decoder.sv tb_uf_surface.cpp -o sim_surf
	./$(OBJ_DIR)/sim_surf uf_surface_graph.svh uf_surface_oracle.mem uf_surface_golden.mem

# Q6-09 — distance scaling: build the SAME parametric FSM with the d=5 graph and verify it over all
# weight-≤2 errors (2^24 syndromes is too many to enumerate; a distance-5 code must correct ≤2 errors).
surf-d5:
	cd .. && $(CARGO) qec_surface_uf_graph -- graph 5 > hw/uf_surface_graph_d5.svh
	rm -rf _d5build && mkdir _d5build
	cp uf_surface_decoder.sv uf_surface_graph_d5.svh _d5build/
	mv _d5build/uf_surface_graph_d5.svh _d5build/uf_surface_graph.svh
	cd _d5build && $(VERILATOR) --cc --exe --build -Wall --top-module uf_surface_decoder \
		uf_surface_decoder.sv ../tb_uf_surface_scale.cpp -o sim_scale_d5
	./_d5build/$(OBJ_DIR)/sim_scale_d5 _d5build/uf_surface_graph.svh

# Q6-17 — d=7 scaling: same parametric FSM with the d=7 graph (N=49, M=110). UF_M>64 makes the
# `correction` port a Verilator VlWide<>, which the scale TB handles via the cbit() overloads. A
# distance-7 code corrects <=3 errors, so the weight-<=2 sweep here is a necessary (not exhaustive)
# correctness check; its real purpose is the scaling verdict (fit/Fmax/latency, measured by synth).
surf-d7:
	cd .. && $(CARGO) qec_surface_uf_graph -- graph 7 > hw/uf_surface_graph_d7.svh
	rm -rf _d7build && mkdir _d7build
	cp uf_surface_decoder.sv uf_surface_graph_d7.svh _d7build/
	mv _d7build/uf_surface_graph_d7.svh _d7build/uf_surface_graph.svh
	cd _d7build && $(VERILATOR) --cc --exe --build -Wall --top-module uf_surface_decoder \
		uf_surface_decoder.sv ../tb_uf_surface_scale.cpp -o sim_scale_d7
	./_d7build/$(OBJ_DIR)/sim_scale_d7 _d7build/uf_surface_graph.svh

# Re-baseline the Q6-04 golden table {obs_flip,correction} from the CURRENT RTL. Only run this
# deliberately when the decoder's intended output changes — it overwrites the frozen reference that
# `surf` checks against, so a careless run would mask a regression.
golden-rebaseline:
	$(VERILATOR) --cc --exe --build -Wall --top-module uf_surface_decoder \
		uf_surface_decoder.sv tb_dump_golden.cpp -o dump_golden
	./$(OBJ_DIR)/dump_golden > uf_surface_golden.mem

clean:
	rm -rf $(OBJ_DIR)
