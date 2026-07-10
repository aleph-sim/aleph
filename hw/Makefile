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

.PHONY: all d3 rep surf surf-d5 surf-d7 surf-3d surf-window stream stream-axi stream-axi-circuit cosim-stream cosim-stream-circuit cosim-stream-ler cosim-stream-ler-circuit cosim cosim-3d cosim-circuit cosim-circuit-3d bpcheck bprelay bpunroll bppartial bppartialfast bppipe bpfast bpaxi bpcirc bpaxiwide bpbram bpbramfast bpbramdp bpbramdpearly checkminsum varupdate unrollskel bpunrollpipe bpbanked-lint clean

all: d3 rep surf

# Q7-02 M1 — fixed-point relay-BP check-update datapath vs the Rust golden (gross [[144,12,12]]).
# Emits the baked-in Tanner graph + Q5.3 params, then replays 256 random check-update vectors through
# the combinational RTL and checks all 432 output messages bit-for-bit against FixedRelayBp.
bpcheck:
	cd .. && $(CARGO) qec_q7_bp_graph -- graph   > hw/bb_gross_tanner.svh
	cd .. && $(CARGO) qec_q7_bp_graph -- vectors > hw/bp_check_vectors.txt
	$(VERILATOR) --cc --exe --build -Wall --top-module bp_check_update \
		bp_check_update.sv tb_bp_check.cpp -o sim_bpcheck
	./$(OBJ_DIR)/sim_bpcheck bp_check_vectors.txt

# Q7-02 M2 — full sequential relay-BP decoder vs the Rust golden (gross [[144,12,12]]).
# Emits the graph + full-decode golden vectors, then drives each syndrome through the FSM and checks
# the chosen error, observable flips, and validity bit-for-bit against FixedRelayBp::decode_fixed_ehat.
bprelay:
	cd .. && $(CARGO) qec_q7_bp_graph -- graph      > hw/bb_gross_tanner.svh
	cd .. && $(CARGO) qec_q7_bp_graph -- decvectors > hw/bp_dec_vectors.txt
	$(VERILATOR) --cc --exe --build -Wall --top-module bp_relay_decoder \
		bp_relay_decoder.sv tb_bp_relay.cpp -o sim_bprelay
	./$(OBJ_DIR)/sim_bprelay bp_dec_vectors.txt

# Q7-02 M4 — spatially-unrolled relay-BP decoder (all checks/vars per cycle) vs the SAME golden.
# Same TB/vectors as M2 (built with -DUNROLL); a pass proves the unroll is bit-for-bit identical to
# the sequential M2 (hence FixedRelayBp) while collapsing 28 944 cycles → ~302.
bpunroll:
	cd .. && $(CARGO) qec_q7_bp_graph -- graph      > hw/bb_gross_tanner.svh
	cd .. && $(CARGO) qec_q7_bp_graph -- decvectors > hw/bp_dec_vectors.txt
	$(VERILATOR) --cc --exe --build -Wall --Mdir obj_unroll --top-module bp_relay_unrolled -CFLAGS -DUNROLL \
		bp_relay_unrolled.sv tb_bp_relay.cpp -o sim_bpunroll
	./obj_unroll/sim_bpunroll bp_dec_vectors.txt

# Q7-02 M5-followup — partially-unrolled relay-BP decoder (CHK_UNROLL checks / VAR_UNROLL vars per
# cycle) vs the SAME golden. Middle point between M2 (1 node/cyc, fits anything) and M4 (all/cyc, KV260-
# only): fits a xc7z020 (Arty/Zybo) while far faster than M2. -DPARTIAL selects it in the shared TB.
bppartial:
	cd .. && $(CARGO) qec_q7_bp_graph -- graph      > hw/bb_gross_tanner.svh
	cd .. && $(CARGO) qec_q7_bp_graph -- decvectors > hw/bp_dec_vectors.txt
	$(VERILATOR) --cc --exe --build -Wall --Mdir obj_partial --top-module bp_relay_partial -CFLAGS -DPARTIAL \
		bp_relay_partial.sv tb_bp_relay.cpp -o sim_bppartial
	./obj_partial/sim_bppartial bp_dec_vectors.txt

# Q7-02 M5-followup — SAT-OVERLAPPED partial decoder (the Arty-fit fast variant): folds S_SAT into the
# next S_CHECK's group cursor → G_CHK+G_VAR cyc/iter not 2·G_CHK+G_VAR (12/24: 12 not 18 → ~720 vs 1086,
# ~1.5×) at the same Arty part. -DPARTIALFAST selects it in the shared TB.
bppartialfast:
	cd .. && $(CARGO) qec_q7_bp_graph -- graph      > hw/bb_gross_tanner.svh
	cd .. && $(CARGO) qec_q7_bp_graph -- decvectors > hw/bp_dec_vectors.txt
	$(VERILATOR) --cc --exe --build -Wall --Mdir obj_partialfast --top-module bp_relay_partial_fast -CFLAGS -DPARTIALFAST \
		bp_relay_partial_fast.sv tb_bp_relay.cpp -o sim_bppartialfast
	./obj_partialfast/sim_bppartialfast bp_dec_vectors.txt

# Q7-02 M5-followup — min-sum-PIPELINED unrolled decoder: S_CHECK split into S_CHK1 (reduction, deep)
# + S_CHK2 (output, shallow) to raise Fmax past the ~96 MHz min-sum wall. +1 cyc/iter (181->241);
# net win only if Fmax rises >1.33×. -DPIPE selects it in the shared TB.
bppipe:
	cd .. && $(CARGO) qec_q7_bp_graph -- graph      > hw/bb_gross_tanner.svh
	cd .. && $(CARGO) qec_q7_bp_graph -- decvectors > hw/bp_dec_vectors.txt
	$(VERILATOR) --cc --exe --build -Wall --Mdir obj_pipe --top-module bp_relay_pipe -CFLAGS -DPIPE \
		bp_relay_pipe.sv tb_bp_relay.cpp -o sim_bppipe
	./obj_pipe/sim_bppipe bp_dec_vectors.txt

# Q7-02 M5-followup — SAT-OVERLAPPED unrolled decoder: fold S_SAT to run in parallel with the next
# S_CHECK (independent reg->reg paths) → 2 cyc/iter (181->~123) at the unchanged min-sum Fmax wall.
# Pure cycle-count lever, no Fmax gamble. -DFAST selects it in the shared TB.
bpfast:
	cd .. && $(CARGO) qec_q7_bp_graph -- graph      > hw/bb_gross_tanner.svh
	cd .. && $(CARGO) qec_q7_bp_graph -- decvectors > hw/bp_dec_vectors.txt
	$(VERILATOR) --cc --exe --build -Wall --Mdir obj_fast --top-module bp_relay_fast -CFLAGS -DFAST \
		bp_relay_fast.sv tb_bp_relay.cpp -o sim_bpfast
	./obj_fast/sim_bpfast bp_dec_vectors.txt

# Q7-02 board bring-up — AXI4-Lite wrapper (bp_axi_wrap) around the partial decoder, driven by REAL
# AXI-Lite transactions from tb_bp_axi.cpp (the sim twin of hw/sw/bp_pynq.py). Validates the PS<->PL
# shell + multi-word syndrome/correction register map before the Vivado board build.
bpaxi:
	cd .. && $(CARGO) qec_q7_bp_graph -- graph      > hw/bb_gross_tanner.svh
	cd .. && $(CARGO) qec_q7_bp_graph -- decvectors > hw/bp_dec_vectors.txt
	$(VERILATOR) --cc --exe --build -Wall -Wno-fatal --Mdir obj_bpaxi --top-module bp_axi_wrap \
		bp_axi_wrap.sv bp_relay_partial_fast.sv tb_bp_axi.cpp -o sim_bpaxi
	./obj_bpaxi/sim_bpaxi bp_dec_vectors.txt

# Q7-02 M5-followup — sim↔RTL CO-SIM at CIRCUIT LEVEL. Emits the depth-7 circuit-level gross-code graph
# (irregular, 864 vars / 144 checks / 2952 edges, max check-deg 25 vs code capacity's uniform 6) + real
# DEM-shot golden vectors, then decodes them through the graph-generic M2 sequential decoder in a build
# dir (the circuit header cp'd over bb_gross_tanner.svh). A pass proves the parametric RTL generalises
# past the baked-in code-capacity graph. (Uses M2, not the unrolled variants: the circuit graph is far
# too large to spatially unroll; M2's runtime cursor handles any size.)
bpcirc:
	cd .. && $(CARGO) qec_q7_bp_graph -- circgraph   1 0.003        > hw/bb_circuit_tanner.svh
	cd .. && $(CARGO) qec_q7_bp_graph -- circvectors 1 0.003 40 2024 > hw/bp_circ_vectors.txt
	rm -rf _bpcircbuild && mkdir _bpcircbuild
	cp bp_relay_decoder.sv tb_bp_relay.cpp _bpcircbuild/
	cp bb_circuit_tanner.svh _bpcircbuild/bb_gross_tanner.svh
	cd _bpcircbuild && $(VERILATOR) --cc --exe --build -Wall --top-module bp_relay_decoder \
		bp_relay_decoder.sv tb_bp_relay.cpp -o sim_bpcirc
	./_bpcircbuild/obj_dir/sim_bpcirc bp_circ_vectors.txt

# Q7-02 M5-followup — WIDE AXI4-Lite wrapper (bp_axi_wrap_wide) around the M2 circuit-level decoder,
# driven by REAL AXI-Lite transactions (tb_bp_axi_wide.cpp = the on-board driver's sim twin) over the
# generic NS-syndrome-word / NC-correction-word register map. Validates the wide PS<->PL shell before
# the Vivado circuit-level board build. Same circuit graph/vectors as bpcirc.
bpaxiwide:
	cd .. && $(CARGO) qec_q7_bp_graph -- circgraph   1 0.003        > hw/bb_circuit_tanner.svh
	cd .. && $(CARGO) qec_q7_bp_graph -- circvectors 1 0.003 40 2024 > hw/bp_circ_vectors.txt
	rm -rf _bpaxiwidebuild && mkdir _bpaxiwidebuild
	cp bp_axi_wrap_wide.sv bp_relay_bram_dp.sv tb_bp_axi_wide.cpp _bpaxiwidebuild/
	cp bb_circuit_tanner.svh _bpaxiwidebuild/bb_gross_tanner.svh
	cd _bpaxiwidebuild && $(VERILATOR) --cc --exe --build -Wall -Wno-fatal --top-module bp_axi_wrap_wide \
		bp_axi_wrap_wide.sv bp_relay_bram_dp.sv tb_bp_axi_wide.cpp -o sim_bpaxiwide
	./_bpaxiwidebuild/obj_dir/sim_bpaxiwide bp_circ_vectors.txt

# Q7-02 M2-BRAM — the SYNTHESIZABLE-at-circuit-scale relay-BP core (bp_relay_bram): m_vc/e_cv in block
# RAM + edge-serial check/var updates, so the runtime-cursor register-file mux that OOMs Vivado on the
# flop-array M2 (#447) disappears. Bit-for-bit verified against the SAME golden as bpcirc (tb_bp_relay
# with -DBRAM), proving the BRAM core is the exact silicon twin of the M2 reference at circuit level.
bpbram:
	cd .. && $(CARGO) qec_q7_bp_graph -- circgraph   1 0.003        > hw/bb_circuit_tanner.svh
	cd .. && $(CARGO) qec_q7_bp_graph -- circvectors 1 0.003 40 2024 > hw/bp_circ_vectors.txt
	rm -rf _bpbrambuild && mkdir _bpbrambuild
	cp bp_relay_bram.sv tb_bp_relay.cpp _bpbrambuild/
	cp bb_circuit_tanner.svh _bpbrambuild/bb_gross_tanner.svh
	cd _bpbrambuild && $(VERILATOR) --cc --exe --build -Wall -Wno-fatal -CFLAGS -DBRAM \
		--top-module bp_relay_bram bp_relay_bram.sv tb_bp_relay.cpp -o sim_bpbram
	./_bpbrambuild/obj_dir/sim_bpbram bp_circ_vectors.txt

# Q7-02 M2-BRAM-FAST — pipelined-read variant of bp_relay_bram: the three edge-serial BRAM read passes
# drop from 2 cyc/edge to 1 cyc/edge (lead/lag address/consume cursor), and the S_SAT keep-best commit is
# registered for Fmax. ~1.39× fewer cycles on the circuit graph. Bit-exact against the SAME golden.
bpbramfast:
	cd .. && $(CARGO) qec_q7_bp_graph -- circgraph   1 0.003        > hw/bb_circuit_tanner.svh
	cd .. && $(CARGO) qec_q7_bp_graph -- circvectors 1 0.003 40 2024 > hw/bp_circ_vectors.txt
	rm -rf _bpbramfastbuild && mkdir _bpbramfastbuild
	cp bp_relay_bram_fast.sv tb_bp_relay.cpp _bpbramfastbuild/
	cp bb_circuit_tanner.svh _bpbramfastbuild/bb_gross_tanner.svh
	cd _bpbramfastbuild && $(VERILATOR) --cc --exe --build -Wall -Wno-fatal -CFLAGS -DBRAMFAST \
		--top-module bp_relay_bram_fast bp_relay_bram_fast.sv tb_bp_relay.cpp -o sim_bpbramfast
	./_bpbramfastbuild/obj_dir/sim_bpbramfast bp_circ_vectors.txt

# Q7-02 M2-BRAM-DP — dual-port, 2-edges/cycle variant: m_vc/e_cv each two TRUE dual-port banks (edge&1),
# every pass runs 2 edges/cycle (contiguous + scattered). ~2× fewer cycles than the fast core. Bit-exact
# against the SAME golden.
bpbramdp:
	cd .. && $(CARGO) qec_q7_bp_graph -- circgraph   1 0.003        > hw/bb_circuit_tanner.svh
	cd .. && $(CARGO) qec_q7_bp_graph -- circvectors 1 0.003 40 2024 > hw/bp_circ_vectors.txt
	rm -rf _bpbramdpbuild && mkdir _bpbramdpbuild
	cp bp_relay_bram_dp.sv tb_bp_relay.cpp _bpbramdpbuild/
	cp bb_circuit_tanner.svh _bpbramdpbuild/bb_gross_tanner.svh
	cd _bpbramdpbuild && $(VERILATOR) --cc --exe --build -Wall -Wno-fatal -CFLAGS -DBRAMDP \
		--top-module bp_relay_bram_dp bp_relay_bram_dp.sv tb_bp_relay.cpp -o sim_bpbramdp
	./_bpbramdpbuild/obj_dir/sim_bpbramdp bp_circ_vectors.txt

# Q7-02 M2-BRAM-DP EARLY-EXIT — same dual-port core, early_exit=1 (stop at first syndrome-valid), checked
# bit-for-bit against an early-exit golden (circvectorsearly). Proves the runtime early-exit mode matches
# the software early-stop; the per-decode latency now varies per shot (average-case lever).
bpbramdpearly:
	cd .. && $(CARGO) qec_q7_bp_graph -- circgraph        1 0.003        > hw/bb_circuit_tanner.svh
	cd .. && $(CARGO) qec_q7_bp_graph -- circvectorsearly 1 0.003 40 2024 > hw/bp_circ_vectors_early.txt
	rm -rf _bpbramdpearlybuild && mkdir _bpbramdpearlybuild
	cp bp_relay_bram_dp.sv tb_bp_relay.cpp _bpbramdpearlybuild/
	cp bb_circuit_tanner.svh _bpbramdpearlybuild/bb_gross_tanner.svh
	cd _bpbramdpearlybuild && $(VERILATOR) --cc --exe --build -Wall -Wno-fatal -CFLAGS -DBRAMDP \
		--top-module bp_relay_bram_dp bp_relay_bram_dp.sv tb_bp_relay.cpp -o sim_bpbramdpearly
	./_bpbramdpearlybuild/obj_dir/sim_bpbramdpearly bp_circ_vectors_early.txt early

# Q7-02 M7 — standalone `check_minsum` submodule: the FIRST of the two small per-node submodules the
# hierarchically-modular relay-BP core (M7) will stamp out many times (one per check node). Generic in
# MW/DEG (no baked-in graph `include), so no golden-vector regeneration step is needed here — the C++
# TB reimplements the bp_relay_fast.sv S_CHECK arithmetic directly and drives >=10000 random cases.
checkminsum:
	$(VERILATOR) --cc --exe --build -Wall --top-module check_minsum \
		check_minsum.sv tb_check_minsum.cpp -o sim_checkminsum
	./$(OBJ_DIR)/sim_checkminsum

# Q7-02 M7 — standalone `var_update` submodule: the SECOND of the two small per-node submodules the
# hierarchically-modular relay-BP core (M7) will stamp out many times (one per variable node, 864x on
# the circuit graph). Generic in MW/WACC/FRAC/DEG/MAXMAG (no baked-in graph `include), so no
# golden-vector regeneration step is needed here — the C++ TB reimplements the bp_relay_fast.sv S_VAR
# arithmetic directly and drives >=10000 random cases.
varupdate:
	$(VERILATOR) --cc --exe --build -Wall --top-module var_update \
		var_update.sv tb_var_update.cpp -o sim_varupdate
	./$(OBJ_DIR)/sim_varupdate

# Q7-02 M7 — Step-0 OOC fit-gate SKELETON: Verilator LINT-ONLY elaboration of `bp_unroll_skeleton`
# (all 144 `check_minsum` + all 864 `var_update` instances, CSR-constant-indexed wiring at circuit-graph
# scale) against `check_minsum`/`var_update` (already unit-verified in isolation by `checkminsum`/
# `varupdate` above). Lint-only, no C++ TB: the skeleton has no intended functional behavior to check —
# it exists to catch width/UNDRIVEN/index mistakes in the generate-based stamping CHEAPLY, before the
# expensive step of staging to Vivado for an out-of-context synth "fit" run (hw/syn/ooc_fit_gate.tcl).
unrollskel:
	cd .. && $(CARGO) qec_q7_bp_graph -- circgraph 1 0.003 > hw/bb_gross_tanner.svh
	$(VERILATOR) --lint-only -Wall --top-module bp_unroll_skeleton \
		check_minsum.sv var_update.sv bp_unroll_skeleton.sv

# Q7-02 M7 — MODULAR PARTIAL-UNROLL relay-BP core (`bp_relay_unroll_pipe`): stamps CHK_UNROLL
# `check_minsum` + VAR_UNROLL `var_update` submodule slots (Tasks 1-2), time-multiplexed across NGROUP
# groups per BP phase (pipelined 2-cycle submodules). Regenerates the circuit-level graph + 40-shot golden
# (the SAME golden `bpcirc`/`bpfast` use), then co-sims the core at BOTH NGROUP=2 and NGROUP=4, asserting
# 40/40 bit-identical at each — the decode is INDEPENDENT of NGROUP (grouping is a pure scheduling change),
# so identical decodes at two group counts is the G-invariance check that catches gather/scatter bugs.
bpunrollpipe:
	cd .. && $(CARGO) qec_q7_bp_graph -- circgraph   1 0.003        > hw/bb_gross_tanner.svh
	cd .. && $(CARGO) qec_q7_bp_graph -- circvectors 1 0.003 40 2024 > hw/bp_circ_vectors.txt
	@echo "== NGROUP=4 =="
	$(VERILATOR) --cc --exe --build -Wall --Mdir obj_unrollpipe4 --top-module bp_relay_unroll_pipe \
		-GNGROUP=4 check_minsum.sv var_update.sv bp_relay_unroll_pipe.sv tb_bp_unroll_pipe.cpp -o sim_bpunrollpipe4
	./obj_unrollpipe4/sim_bpunrollpipe4 bp_circ_vectors.txt
	@echo "== NGROUP=2 (G-invariance) =="
	$(VERILATOR) --cc --exe --build -Wall --Mdir obj_unrollpipe2 --top-module bp_relay_unroll_pipe \
		-GNGROUP=2 check_minsum.sv var_update.sv bp_relay_unroll_pipe.sv tb_bp_unroll_pipe.cpp -o sim_bpunrollpipe2
	./obj_unrollpipe2/sim_bpunrollpipe2 bp_circ_vectors.txt

# Q7-02 M7 — K-BANKED relay-BP core (`bp_relay_banked`): Verilator LINT-ONLY elaboration against the
# beta-split check-major LUTRAM-bank layout (Task 10). Regenerates the circuit-level graph WITH the banked
# header tables (BP_BANK_W/V, BP_GC/GV, BP_CHK_AT/BP_VAR_AT, BP_EDGE_CHK/POS/BETA — the `8 24` bank/group
# sizes the Task-9 emitter solves offline), then elaborates the core + its two unit-verified submodules.
# Lint-only, no C++ TB: full co-sim (G-invariance vs the bp_relay_fast golden) is the NEXT task; this gate
# catches width / UNDRIVEN / bank-index mistakes cheaply before that.
bpbanked-lint:
	cd .. && $(CARGO) qec_q7_bp_graph -- circgraph 1 0.003 8 24 > hw/bb_gross_tanner.svh
	$(VERILATOR) --lint-only -Wall --top-module bp_relay_banked \
		check_minsum.sv var_update.sv bp_relay_banked.sv

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

# Q6-20 (on silicon) — AXI4-Stream front-end smoke test: wrap the streaming decoder behind the DMA
# stream interface (one round/beat in, one word/window out) and check that the framing preserves
# validity, tags tlast on the last window, and is invariant under S2MM back-pressure. Window params
# (UF_ACTIVE/UF_DPR/UF_LOAD_LO) are passed to the TB as -D defines so it drives exact window boundaries.
stream-axi:
	cd .. && $(CARGO) qec_surface_uf_graph -- window 3 9 3 > hw/uf_surface_graph_win.svh
	rm -rf _streamaxibuild && mkdir _streamaxibuild
	cp uf_stream_win_core.sv uf_streaming_decoder.sv uf_surface_decoder.sv uf_surface_graph_win.svh _streamaxibuild/
	mv _streamaxibuild/uf_surface_graph_win.svh _streamaxibuild/uf_surface_graph.svh
	cd _streamaxibuild && \
	  A=$$(grep -oE 'UF_ACTIVE *= *[0-9]+' uf_surface_graph.svh | grep -oE '[0-9]+$$'); \
	  D=$$(grep -oE 'UF_DPR *= *[0-9]+' uf_surface_graph.svh | grep -oE '[0-9]+$$'); \
	  L=$$(grep -oE 'UF_LOAD_LO *= *[0-9]+' uf_surface_graph.svh | grep -oE '[0-9]+$$'); \
	  $(VERILATOR) --cc --exe --build -Wall --top-module uf_stream_win_core \
	    -CFLAGS "-DUF_ACTIVE=$$A -DUF_DPR=$$D -DUF_LOAD_LO=$$L" \
	    uf_stream_win_core.sv uf_streaming_decoder.sv uf_surface_decoder.sv ../tb_uf_stream_win.cpp \
	    -o sim_stream_axi
	./_streamaxibuild/$(OBJ_DIR)/sim_stream_axi

# Q6-23 (circuit-level) — same AXI framing smoke test, but on the CIRCUIT-LEVEL window graph
# (window-circuit: same detectors/streaming metadata, +hook-error edges). The graph-agnostic decoder
# handles it with no RTL change; this checks validity/tlast/back-pressure/frame-independence hold.
stream-axi-circuit:
	cd .. && $(CARGO) qec_surface_uf_graph -- window-circuit 3 9 3 > hw/uf_surface_graph_win_circuit.svh
	rm -rf _streamaxicirc && mkdir _streamaxicirc
	cp uf_stream_win_core.sv uf_streaming_decoder.sv uf_surface_decoder.sv uf_surface_graph_win_circuit.svh _streamaxicirc/
	mv _streamaxicirc/uf_surface_graph_win_circuit.svh _streamaxicirc/uf_surface_graph.svh
	cd _streamaxicirc && \
	  A=$$(grep -oE 'UF_ACTIVE *= *[0-9]+' uf_surface_graph.svh | grep -oE '[0-9]+$$'); \
	  D=$$(grep -oE 'UF_DPR *= *[0-9]+' uf_surface_graph.svh | grep -oE '[0-9]+$$'); \
	  L=$$(grep -oE 'UF_LOAD_LO *= *[0-9]+' uf_surface_graph.svh | grep -oE '[0-9]+$$'); \
	  $(VERILATOR) --cc --exe --build -Wall --top-module uf_stream_win_core \
	    -CFLAGS "-DUF_ACTIVE=$$A -DUF_DPR=$$D -DUF_LOAD_LO=$$L" \
	    uf_stream_win_core.sv uf_streaming_decoder.sv uf_surface_decoder.sv ../tb_uf_stream_win.cpp \
	    -o sim_stream_axi_circ
	./_streamaxicirc/$(OBJ_DIR)/sim_stream_axi_circ

# Q6-20 (on silicon) — regenerate the per-round streaming co-sim vector for the on-board DMA driver
# (hw/sw/uf_dma_stream.py). Realistic Monte-Carlo round streams from the phenomenological DEM at each p.
STREAM_ROUNDS ?= 800
STREAM_SEED   ?= 2024
cosim-stream:
	cd .. && $(CARGO) qec_q6_stream_cosim -- 3 9 3 $(STREAM_ROUNDS) $(STREAM_SEED) > hw/cosim_stream_d3.vec

# Q6-23 — circuit-level variants of the streaming vectors (pair with the window-circuit bitstream).
cosim-stream-circuit:
	cd .. && $(CARGO) qec_q6_stream_cosim -- 3 9 3 $(STREAM_ROUNDS) $(STREAM_SEED) circuit > hw/cosim_stream_circuit_d3.vec

# Q6-22 (on silicon) — regenerate the finite-experiment streaming LER vector: E memory-Z experiments
# (17 rounds = 18 slices = W+3C) with true logical + the software SlidingWindowDecoder baseline, for
# the on-board streaming LER-within-CI check (hw/sw/uf_dma_stream_ler.py).
LER_SHOTS ?= 4000
cosim-stream-ler:
	cd .. && $(CARGO) qec_q6_stream_ler -- 3 9 3 17 $(LER_SHOTS) $(STREAM_SEED) > hw/cosim_stream_ler_d3.vec

cosim-stream-ler-circuit:
	cd .. && $(CARGO) qec_q6_stream_ler -- 3 9 3 17 $(LER_SHOTS) $(STREAM_SEED) circuit > hw/cosim_stream_ler_circuit_d3.vec

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

# Q6-21 (circuit-level) — board-free co-sim on the full CIRCUIT-LEVEL graph (Q-surface): every CNOT /
# idle / prep / measurement is a fault site, so the matching graph carries hook-error edges the
# phenomenological graph lacks. The RTL decoder thus sees the realistic graph. d=3 x 3 rounds; the
# circuit-level threshold is ~0.9%, so the prob grid is lower (0.002..0.010) and we gate the
# sub-threshold operating regime (gate_p=0.006). Own build dir (compile-time `include graph).
cosim-circuit:
	cd .. && $(CARGO) qec_surface_uf_graph -- graph-circuit 3 3 > hw/uf_surface_graph_d3r3c.svh
	cd .. && $(CARGO) qec_q6_cosim -- 3 3 $(COSIM_SHOTS) $(COSIM_SEED) circuit > hw/cosim_d3r3c.vec
	rm -rf _cosimcircuitbuild && mkdir _cosimcircuitbuild && cp uf_surface_decoder.sv _cosimcircuitbuild/
	cp uf_surface_graph_d3r3c.svh _cosimcircuitbuild/uf_surface_graph.svh
	cd _cosimcircuitbuild && $(VERILATOR) --cc --exe --build -Wall --top-module uf_surface_decoder \
		uf_surface_decoder.sv ../tb_uf_cosim.cpp -o sim_cosimc
	./_cosimcircuitbuild/$(OBJ_DIR)/sim_cosimc _cosimcircuitbuild/uf_surface_graph.svh cosim_d3r3c.vec 0.006

# Q6-21 (circuit-level, d=5) — circuit-level co-sim on the larger d=5 x 3 graph: more detectors and a
# denser hook-error graph than d=3 (a sterner test of the RTL decoder on the realistic graph). d=5
# corrects 2 errors, so it stays within CI deeper into the grid; gate the sub-threshold regime.
cosim-circuit-3d:
	cd .. && $(CARGO) qec_surface_uf_graph -- graph-circuit 5 3 > hw/uf_surface_graph_d5r3c.svh
	cd .. && $(CARGO) qec_q6_cosim -- 5 3 $(COSIM_SHOTS) $(COSIM_SEED) circuit > hw/cosim_d5r3c.vec
	rm -rf _cosimc3dbuild && mkdir _cosimc3dbuild && cp uf_surface_decoder.sv _cosimc3dbuild/
	cp uf_surface_graph_d5r3c.svh _cosimc3dbuild/uf_surface_graph.svh
	cd _cosimc3dbuild && $(VERILATOR) --cc --exe --build -Wall --top-module uf_surface_decoder \
		uf_surface_decoder.sv ../tb_uf_cosim.cpp -o sim_cosimc3d
	./_cosimc3dbuild/$(OBJ_DIR)/sim_cosimc3d _cosimc3dbuild/uf_surface_graph.svh cosim_d5r3c.vec 0.006

# Re-baseline the Q6-04 golden table {obs_flip,correction} from the CURRENT RTL. Only run this
# deliberately when the decoder's intended output changes — it overwrites the frozen reference that
# `surf` checks against, so a careless run would mask a regression.
golden-rebaseline:
	$(VERILATOR) --cc --exe --build -Wall --top-module uf_surface_decoder \
		uf_surface_decoder.sv tb_dump_golden.cpp -o dump_golden
	./$(OBJ_DIR)/dump_golden > uf_surface_golden.mem

clean:
	rm -rf $(OBJ_DIR)
