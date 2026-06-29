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

.PHONY: all d3 rep surf surf-d5 surf-d7 surf-3d surf-window stream cosim cosim-3d clean

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

# Q6-19 — multi-round (3D space-time) decoding. The SAME parametric FSM, fed a multi-round
# phenomenological graph (`graph <d> <rounds>`): T measurement rounds with time-like measurement-error
# edges. Proves the decoder handles the time dimension (not just code-capacity) with no RTL change.
# Two cases: d=5x3 (48 det, clean 0/N weight-<=2 since d=5 corrects 2) and d=5x5 (72 det / 186 edges =
# identical size to the Q3 GPU bench graph, exercising the wide-syndrome TB path; for the apples-to-
# apples GPU-vs-FPGA latency comparison in docs/perf/qec-q6-fpga.md §Q6-03).
surf-3d:
	cd .. && $(CARGO) qec_surface_uf_graph -- graph 5 3 > hw/uf_surface_graph_d5r3.svh
	cd .. && $(CARGO) qec_surface_uf_graph -- graph 5 5 > hw/uf_surface_graph_d5r5.svh
	# The graph is compile-time (`include) in the RTL, so each case needs its own Verilator build.
	rm -rf _d5r3build && mkdir _d5r3build && cp uf_surface_decoder.sv _d5r3build/
	cp uf_surface_graph_d5r3.svh _d5r3build/uf_surface_graph.svh
	cd _d5r3build && $(VERILATOR) --cc --exe --build -Wall --top-module uf_surface_decoder \
		uf_surface_decoder.sv ../tb_uf_surface_scale.cpp -o sim_3d
	@echo "== d=5 x 3 rounds (48 detectors, 3 measurement rounds) =="
	./_d5r3build/$(OBJ_DIR)/sim_3d _d5r3build/uf_surface_graph.svh
	rm -rf _d5r5build && mkdir _d5r5build && cp uf_surface_decoder.sv _d5r5build/
	cp uf_surface_graph_d5r5.svh _d5r5build/uf_surface_graph.svh
	cd _d5r5build && $(VERILATOR) --cc --exe --build -Wall --top-module uf_surface_decoder \
		uf_surface_decoder.sv ../tb_uf_surface_scale.cpp -o sim_3d
	@echo "== d=5 x 5 rounds (72 detectors, = GPU bench graph size) =="
	./_d5r5build/$(OBJ_DIR)/sim_3d _d5r5build/uf_surface_graph.svh

# Q6-20 (part 1) — streaming-window graph: verify the existing per-window core decodes a steady-state
# sliding-window graph (a W-round volume whose time cuts route to temporal-sink nodes), generated from
# the same SlidingWindowDecoder::window_dem the software uses. This validates the window-graph
# structure the Q6-20 streaming wrapper will drive; the wrapper itself (residual carry, commit, shift)
# is parts 2-3. d=3 W=9 C=3 mirrors the software Q4-03 params (W=3d, C=d).
surf-window:
	cd .. && $(CARGO) qec_surface_uf_graph -- window 3 9 3 > hw/uf_surface_graph_win.svh
	rm -rf _winbuild && mkdir _winbuild && cp uf_surface_decoder.sv _winbuild/
	cp uf_surface_graph_win.svh _winbuild/uf_surface_graph.svh
	cd _winbuild && $(VERILATOR) --cc --exe --build -Wall --top-module uf_surface_decoder \
		uf_surface_decoder.sv ../tb_uf_surface_scale.cpp -o sim_win
	./_winbuild/$(OBJ_DIR)/sim_win _winbuild/uf_surface_graph.svh

# Q6-20 (part 2) — sliding-window streaming wrapper smoke test: build the wrapper + the per-window
# core against the d=3 W=9 C=3 window graph and run the streaming handshake (zero stream -> zero
# logical, windows fire, no stall). Both modules `include the window graph (include-guarded).
stream:
	cd .. && $(CARGO) qec_surface_uf_graph -- window 3 9 3 > hw/uf_surface_graph_win.svh
	rm -rf _streambuild && mkdir _streambuild
	cp uf_streaming_decoder.sv uf_surface_decoder.sv uf_surface_graph_win.svh _streambuild/
	mv _streambuild/uf_surface_graph_win.svh _streambuild/uf_surface_graph.svh
	cd _streambuild && $(VERILATOR) --cc --exe --build -Wall --top-module uf_streaming_decoder \
		uf_streaming_decoder.sv uf_surface_decoder.sv ../tb_uf_streaming.cpp -o sim_stream
	./_streambuild/$(OBJ_DIR)/sim_stream

# Q6-21 — board-free sim<->RTL co-simulation. The simulator dumps a Monte-Carlo syndrome stream
# (qec_q6_cosim) drawn from the SAME DEM the RTL graph (graph d rounds) is generated from; the TB
# feeds it through the Verilated decoder and checks the RTL logical-error rate vs the software
# UnionFind baseline (carried in the .vec headers) within Monte-Carlo CI. d=3, code-capacity (1
# round): the full 2D matching graph, swept over p=0.01..0.05.
COSIM_SHOTS ?= 20000
COSIM_SEED  ?= 2024
cosim:
	cd .. && $(CARGO) qec_surface_uf_graph -- graph > hw/uf_surface_graph.svh
	cd .. && $(CARGO) qec_q6_cosim -- 3 1 $(COSIM_SHOTS) $(COSIM_SEED) > hw/cosim_d3.vec
	$(VERILATOR) --cc --exe --build -Wall --top-module uf_surface_decoder \
		uf_surface_decoder.sv tb_uf_cosim.cpp -o sim_cosim
	./$(OBJ_DIR)/sim_cosim uf_surface_graph.svh cosim_d3.vec

# Q6-21 (3D) — same harness on the multi-round phenomenological graph (d=5 x 3 rounds, 48 detectors,
# Q6-19): exercises the time dimension end-to-end (noise -> 3D syndromes -> RTL decode -> LER). The
# graph is compile-time `include, so this needs its own build dir (cf. surf-3d).
cosim-3d:
	cd .. && $(CARGO) qec_surface_uf_graph -- graph 5 3 > hw/uf_surface_graph_d5r3.svh
	cd .. && $(CARGO) qec_q6_cosim -- 5 3 $(COSIM_SHOTS) $(COSIM_SEED) > hw/cosim_d5r3.vec
	rm -rf _cosim3dbuild && mkdir _cosim3dbuild && cp uf_surface_decoder.sv _cosim3dbuild/
	cp uf_surface_graph_d5r3.svh _cosim3dbuild/uf_surface_graph.svh
	cd _cosim3dbuild && $(VERILATOR) --cc --exe --build -Wall --top-module uf_surface_decoder \
		uf_surface_decoder.sv ../tb_uf_cosim.cpp -o sim_cosim3d
	# gate_p=0.02: gate the sub-threshold (operating) regime; supra-threshold rows are reported as
	# info (unweighted RTL UF tie-breaks degenerate cosets more crudely above threshold — expected).
	./_cosim3dbuild/$(OBJ_DIR)/sim_cosim3d _cosim3dbuild/uf_surface_graph.svh cosim_d5r3.vec 0.02

# Re-baseline the Q6-04 golden table {obs_flip,correction} from the CURRENT RTL. Only run this
# deliberately when the decoder's intended output changes — it overwrites the frozen reference that
# `surf` checks against, so a careless run would mask a regression.
golden-rebaseline:
	$(VERILATOR) --cc --exe --build -Wall --top-module uf_surface_decoder \
		uf_surface_decoder.sv tb_dump_golden.cpp -o dump_golden
	./$(OBJ_DIR)/dump_golden > uf_surface_golden.mem

clean:
	rm -rf $(OBJ_DIR)
