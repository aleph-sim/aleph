# `hw/` — real-time QEC decoders in RTL

The North Star of the QEC track is a real-time decoder on hardware (FPGA → ASIC). This directory
holds two decoder families as synthesisable SystemVerilog, each verified bit-for-bit against its
Rust reference in `crates/aleph-qec`, then on real boards:

| track | code | decoder | flagship result | where |
|-------|------|---------|-----------------|-------|
| **Q7** (the current program) | gross bivariate-bicycle qLDPC code `[[144,12,12]]`, circuit-level noise | **relay-BP**, fixed-point Q5.3, K-banked | **15.64 µs worst / 0.85 µs median on a $300 Kria KV260**, bit-exact to software on 3×10⁶ shots; 6.07 µs on a VU47P; 0.88 µs at ASAP7 7 nm (predictive) | [§ Phase Q7](#phase-q7--relay-bp-decoder-for-the-gross-code-1441212) |
| **Q6** (the first program) | rotated surface code, d=3…7 | **Union-Find**, 2-D + space-time | 600 ns/decode on an Arty Z7-20; 16.5 M decodes/s through AXI-DMA | [§ Phase Q6](#phase-q6--union-find-surface-code-decoder-sim-first-fpga-track) |

The shippable artefact is the Q7 decoder: a pre-built KV260 bitstream, driver and self-test live in
[`hw/product/`](product/README.md) and are published as the
[`appliance-v1`](https://github.com/aleph-sim/aleph/releases/tag/appliance-v1) release. The paper
covering the whole Q7 ladder is in [`paper/`](../paper/) (preprint).

Everything here is licensed **Apache-2.0** (patent grant), unlike the MIT Rust crates — see
[§ Licence](#licence--apache-20-not-mit).

Simulation is Mac-native (Verilator ≥ 5.050); synthesis needs Vivado on an x86-Linux host; the
ASIC probes use OpenROAD-flow-scripts. All three flows are board-independent up to the final
bitstream load.

-----

## Phase Q7 — relay-BP decoder for the gross code [[144,12,12]]

### What it decodes, and how

The target is the **gross code** of Bravyi et al. (2024): a bivariate-bicycle CSS code with
n=144, k=12, d=12, 72 X-checks and 72 Z-checks, every check of weight 6 and every qubit of degree 3.
Each error lights three checks, so the syndrome graph is a hypergraph — matching decoders do not
apply and belief propagation is required.

The decoder is **relay-BP** (Müller et al., 2025): several *legs* of normalised min-sum BP, each
reseeding a per-variable disordered memory coefficient γ_v and relaying the previous leg's messages,
keeping the lowest-weight syndrome-consistent solution seen. It stays inside message passing — no
data-dependent Gaussian elimination — so the worst-case latency is a constant. In software it beats
BP-OSD at low p on this code (`docs/perf/qec-q5-gross.md`).

Two detector error models feed the same RTL through a Tanner graph header:

| DEM | detectors / mechanisms / edges | used by |
|-----|-------------------------------|---------|
| **code capacity** (one perfect round, independent Z error per qubit) | 72 / 144 / 432 | M2–M5 (the fixed-point and schedule studies; the fully unrolled cores) |
| **circuit level** (depth-7 syndrome extraction, CNOT/idle/prep/meas faults at rate p, Z sector; verified edge-for-edge against Stim) | 144 / 864 / 2952, max check degree 25 | M6 onward — everything that went to the KV260 |

The **fixed-point word** is **Q5.3**: 8-bit signed messages with 3 fractional bits, chosen by a
(W,F) sweep against the f64 decoder (`docs/perf/qec-q7-fixed-bp.md` § M0). With α = 7/8 the check
update is compare/min/sign with no multiplier (`mag − (mag ≫ 3)`); the only multiply is the relay
blend `(1−γ_v)·computed + γ_v·m_old` with γ_v a ROM constant, truncated by arithmetic shift. The
schedule is **6 legs × 10 iterations**, the LER-optimal split at the operating point. All of this is
frozen in the bit-accurate golden model `FixedRelayBp` (`crates/aleph-qec/src/fixed_bp.rs`), which
*is* the RTL specification: a variant is accepted only if its chosen error, observable flips and
`valid_flag` are bit-identical to the golden on every vector.

**`valid_flag` is a herald, and that is load-bearing.** On 10⁶ circuit-level shots per point at
p = 0.003/0.005/0.007, the share of logical errors that carry `valid_flag = 0` is 1.000 / 0.997 /
0.996, and a converged decode is wrong with probability ≤ 3×10⁻⁶ / 1.9×10⁻⁵ / 1.2×10⁻⁴. The
decoder's LER is essentially its non-convergence rate; a consumer that can discard or escalate
flagged shots gets a post-selected LER two to three orders better for free
(`docs/qec/q7-07-nonconvergence-policy.md`). The real-time policy is *do-nothing-but-flag*: an OSD-0
tail on flagged shots rescues 76–96 % of the LER but costs ~1.5 ms/shot on a CPU, so it lives off
the real-time path.

### Golden model, generators and vectors

Every header and every vector file under `hw/` is **generated** from the Rust model and committed;
CI regenerates them and fails on any diff (see [CI](#ci--what-is-gated)). Never hand-edit them.

| generator | emits |
|-----------|-------|
| `cargo run -p aleph-qec --example qec_q7_bp_graph -- graph` | `bb_gross_tanner.svh` — code-capacity Tanner graph + fixed-point params (`BP_C=72, BP_N=144, BP_E=432`). |
| `… -- circgraph <rounds> <p> [W V]` | the same header at circuit level (`BP_C=144, BP_N=864, BP_E=2952`) with the per-variable priors λ(p) baked in and, with `W V`, the banked geometry solved offline. **The committed header is `circgraph 1 0.003 16 48`.** The M2-BRAM family writes its copy to the git-ignored `bb_circuit_tanner.svh`. |
| `… -- vectors` / `decvectors` | `bp_check_vectors.txt` (432 check→variable messages for the combinational M1 block), `bp_dec_vectors.txt` (65 code-capacity decodes: empty syndrome, every single-variable error, 40 random low-weight syndromes). |
| `… -- circvectors <p>` / `circvectorsearly` | 40 circuit-level shots at p, full schedule / early exit — the vectors every KV260 core is gated on. |
| `… -- silvectors <p>` | the on-silicon campaign vectors; takes the decoder's p explicitly and stamps it into the file header (see "matched priors" below). |
| `… -- streamgraph <rounds> <p> <W> <C> <bankW> <bankV>` / `streamvectors` | `bb_stream_tanner.svh` + `bp_stream_vectors*.txt` for the M9b sliding-window core (committed at W=6, C=2). |
| `crates/aleph-qec/src/benes.rs` | the Beneš / AS-Waksman control words the M9c permutation fabrics load (mirrored exactly by `bp_benes.sv` / `bp_asw.sv`). |

The other Q7 examples (`qec_q7_fixed_bp`, `qec_q7_budget`, `qec_q7_circuit_budget`, `qec_q7_early`,
`qec_q7_nonconv`, `qec_q7_osd`, `qec_q7_stream_sweep`) are the software studies behind each
architectural decision; each is cited from the perf record it produced.

**Matched priors.** A bitstream bakes λ(p) into its header. A bit-exactness claim is only meaningful
when the golden decodes with the *same* p as the bitstream: the first 10⁶-shot campaign reported
7 067 / 30 703 divergences at p = 0.005 / 0.007 against a p = 0.003 overlay, and it was a prior
mismatch, not an RTL bug (0/3×10⁶ once matched). `make -C hw bpbanked-highweight` (2 000 shots at
p = 0.007) is the regression gate that reproduces that off-board.

### The architecture ladder (what each file is)

Six synthesised variants, each bit-exact against the same golden, each measured out-of-context on
the target part before the next step was chosen. Cycles are the full-schedule worst case; LUTs are
post-synthesis on the Kria KV260 (XCK26, 117 120 LUTs). Full detail per milestone:
`docs/perf/qec-q7-fixed-bp.md` (the 1 800-line master record).

| milestone | file(s) | graph | cycles | KV260 LUT | clock → latency | what it taught |
|-----------|---------|-------|-------:|----------:|-----------------|----------------|
| **M1** combinational check update | `bp_check_update.sv` | code cap. | — | — | — | the min-sum block, 432/432 messages bit-exact (`make bpcheck`). |
| **M2** sequential FSM | `bp_relay_decoder.sv` | code cap. | 28 944 | 24 119 (21 %) | 67.8 MHz → 427 µs | one check or one variable per cycle; **the runtime cursor is the wall** — a runtime index into a message array synthesises to a mux tree the size of the array (M3 verdict). |
| **M4** spatial unroll | `bp_relay_unrolled.sv` | code cap. | 301 | 94 194 (80 %) | 95.2 MHz → 3.16 µs | every edge a compile-time constant; 96× fewer cycles. Vivado constant-folded the whole datapath away once — flops are anchored with `dont_touch` and flop counts are checked against elaboration on every variant. |
| **M5** SAT-overlap 6×10 | `bp_relay_fast.sv` | code cap. | 122 | 93 487 (80 %) | 95.9 MHz → 1.27 µs | fold the syndrome check into the next check pass: 1.48× at unchanged area. `bp_relay_pipe.sv` is the measured *negative* result (min-sum pipelining: +16 % clock, +33 % cycles). |
| M5 partial unroll | `bp_relay_partial.sv`, `bp_relay_partial_fast.sv` | code cap. | — | fits xc7z020 | 29.3 µs on Arty Z7-20 | the small-part (Arty/Zybo) vehicle: `CHK_UNROLL`/`VAR_UNROLL` groups per cycle; mux the inputs, not the addresses. |
| **M6** edge-serial BRAM | `bp_relay_bram.sv`, `_fast`, `_dp` | circuit | 672 000 | 8 509 (7 %) | 100 MHz → 6.72 ms | the circuit-level graph (degree 25) **does not unroll** on the KV260; only BRAM-resident edge-serial cores fit. First circuit-level qLDPC decode on silicon (Arty, then KV260). Early termination ≈ 18× average-case. |
| **M7** K-banked 12/36 | `bp_relay_banked.sv` | circuit | 2 460 | 88.0 k (75 %) | 75 MHz → 32.8 µs | messages in hundreds of small distributed-RAM banks in **check-major, β-split** order so check-phase access is hardwired and only the variable phase scatters; an offline solve in the emitter guarantees ≤ 1 write per `m_cm` half-bank and ≤ 2 reads per `e_cm` bank per cycle. Zero BRAM. 205× M6. |
| **M8** banked 16/48 (**shipped**) | same file, header at 16/48 | circuit | 2 085 | 102.8 k (88 %) | **133.3 MHz → 15.64 µs** | register plane on the gather outputs + 3-stage `check_minsum`; 16/48 times better than 12/36. Median 0.85 µs with early exit. This is `appliance-v1`. |
| M8 register-file twins | `bp_relay_banked_bram.sv`, `bp_relay_banked_bram_m.sv` | circuit | 2 085 | — | — | decision-equal siblings whose constant decode fabrics are BRAM ROMs (the storage style the ASIC flow uses; `bpbankedbram`/`bpbankedbramm` gates). |
| **M9a/b** sliding-window streaming | `bp_streaming_decoder.sv`, `bp_stream_win_core.sv`, `bb_stream_tanner.svh` | circuit, rounds=12, W=6 C=2 | 3 871 (8/24 window; the rounds=1 register-file core is 2 206 at 16/48, +121 over M8) | — | — | bounded-state decoding of an unbounded round stream; (W,C) chosen by a 100 000-shot sweep (master record § M9a). Bit-exact co-sim in both exit modes. |
| **M9c** full-parallel gather | `bp_benes.sv`, `bp_asw.sv` | circuit, streaming | 2 810 / 4 475 (8/24 / 16/48; the fabrics sit inside the non-overlapped iteration loop) | 189 740 (162 %) | OOC 177.7 MHz | the mux wall in its purest form: a ROM-indexed register-array gather placed at 2.23 M LUTs (1906 %); ROM-configured **Beneš** networks cut it 9.3×, then **AS-Waksman** right-sizing and an `e_cm` address ROM a further −50 k. Terminal verdict: **no single-KV260 fit** (floor is the two runtime-data fabrics); the same fabrics are what the ASIC carries. |
| superseded | `bp_relay_unroll_pipe.sv`, `bp_unroll_skeleton.sv` | circuit | — | 453 k (386 %) | — | the modular full unroll (fit-gate evidence only) and the NGROUP partial unroll: `cycles = 122·NGROUP + 240`, but LUTs = 616 k + 18.4 M/NGROUP — every NGROUP is over budget *and* slower than banked (`docs/perf/q7-02-ngroup-sweep.md`). Kept as the G-invariance reference. |

Leaf datapath modules shared by the banked cores, each with its own randomised golden gate:

| file | role |
|------|------|
| `check_minsum.sv` | one check's min-sum, 2- or 3-stage pipelined (`STAGES`), α = 7/8 multiply-free, exclusive minimum via (min1, min2, argmin). `make checkminsum` proves STAGES-invariance on ≥ 10 000 random cases. |
| `var_update.sv` | one variable's update, 2-cycle pipelined: `total = λ + Σ e_cv`, sign → `ehat`, per-edge relay blend. `make varupdate`. |
| `bp_benes.sv`, `bp_asw.sv` | rearrangeable permutation networks (power-of-two Beneš; arbitrary-size Waksman, switch-optimal at ⌈N log₂ N⌉ − N + 1), control pipelined in lockstep with data; the routing mirrors `benes.rs` exactly. `make bpbenes`, `make bpasw` drive every site-specific top against a reference permutation. |

### Wrappers, board tops and interfaces

| file | role |
|------|------|
| `bp_axi_wrap.sv` / `bp_axi_top.v` | AXI4-Lite wrapper for the code-capacity partial decoder — the Arty Z7-20 bring-up (`syn/arty_z7_bp_bd.tcl`, driver `sw/bp_pynq.py`). |
| `bp_axi_wrap_wide.sv` / `bp_axi_top_wide.v` | graph-generic AXI4-Lite wrapper (multi-word syndrome/correction derived from `BP_C`/`BP_N`) around the M2-BRAM circuit-level cores — Arty (`syn/arty_z7_bp_circ_bd.tcl`) and KV260 M6 (`syn/kv260_bp_circ_bd.tcl`), drivers `sw/bp_circ_pynq.py`, `sw/bp_circ_kv260.py`. |
| `bp_axi_wrap_banked.sv` / `bp_axi_top_banked.v` | the same register map around the **banked core** — the M7/M8 KV260 overlay (`syn/kv260_bp_circ_banked_bd.tcl`). IDCODE `0x4250_0003` ('BP', v3). |
| `bp_stream_banked_core.sv` / `bp_stream_banked.v` | **AXI4-Stream batch front-end** for the banked block decoder, fed by AXI-DMA from PS DDR (`syn/kv260_bp_stream_banked_bd.tcl`): one DMA transfer streams a whole batch of independent experiments. Result word per experiment: `[31:20] obs_flip[11:0]`, `[19] valid_flag`, `[15:0] latency_cycles`. Drivers `sw/bp_stream_banked_kv260.py` (throughput + 40/40 self-test, what `product/deploy.sh` runs) and `sw/bp_stream_banked_ler_kv260.py` (the 10⁶-shot campaign). |
| `bp_stream_win_core.sv` / `bp_stream_win.v` | AXI4-Stream front-end for the M9b sliding-window streaming decoder (per-frame re-arm, `early_exit` as a board-top port). |
| `tb_bp_gate_asap7.sv`, `asap7_latch.v`, `sw/gate_vectors.py` | event-driven gate-level testbench for the ASAP7 routed netlist (Icarus; vendor UDP models), and the behavioural latch model Verilator uses in its place. |

The user-facing contract (bit order, sequences, latency promise, what is *not* implemented) is
`product/interface-spec.md`; `product/BRINGUP.md` is the from-scratch KV260 recipe.

### Results on silicon (Kria KV260, banked 16/48)

All numbers below are measured on one $300-class board, none projected from a larger device
(`docs/perf/qec-q7-fixed-bp.md` § M8, `docs/qec/q7-06-ac1-batched-dma.md`, § Q7-05).

| quantity | value |
|----------|-------|
| timing | `TIMING_MET` at 133.332 MHz (PS clock grid); 102.8 k LUTs (88 %) OOC |
| worst-case decode, full schedule | **2 085 cycles = 15.64 µs** (min = p50 = mean = p99 = max) |
| early exit, 40 shots at p = 0.003 | min 0.59 / **median 0.85** / mean 1.15 / max 4.4 µs |
| bit-exactness | 40/40 both modes over AXI4-Lite; 40/40 at batch sizes 1…20 000 over DMA |
| batched throughput (DMA overlay, 100 MHz) | 43 380 exp/s full schedule (decode-bound); **553 000 exp/s** early exit — 163× the per-word PS-polled path |
| on-silicon LER campaign, 10⁶ shots × {0.003, 0.005, 0.007}, matched-prior bitstreams | LER identical to software to the digit; **0 divergences in 3×10⁶ shots** |
| `valid_flag` on silicon (p = 0.005, 10⁵ fresh shots) | 712 errors on both sides, 857 non-converged on both sides, `valid_mismatch = 0` |
| power (INA260, SOM-total, delta method) | idle 3.25 W → 3.50 W under decode; PL-core dynamic ≲ 17 µJ (full) / ≲ 3 µJ (early exit) per decode, harness-bound upper bounds |

The ladder on the same board: 6.72 ms (M6) → 32.8 µs (M7) → **15.64 µs (M8)**, 430× end to end.

### Scaling past the KV260 (measured, not modelled)

Within the banked family, cycles follow `60·(G_C + G_V) + 420 + (2·G_V + G_C)` exactly, where the
420 is a per-iteration pipeline tail that banking cannot touch — so 4× banking buys 2.28× cycles and
the floor is the full-parallel 144/864 geometry at 543 cycles (`docs/perf/q7-02-fullparallel-fpga.md`).

| geometry | cycles | device | LUTs (util.) | Fmax | latency | source |
|----------|-------:|--------|-------------:|-----:|--------:|--------|
| 16/48 | 2 085 | KV260 (silicon) | 102.8 k (88 %) | 133.3 MHz | 15.64 µs | § above |
| 64/192 | 913 | VU47P, post-route (`syn/impl_vu47p.tcl`, AWS F2 build host, ~$5.50) | 247 k (19 %) | 150.4 MHz | **6.07 µs** | `docs/perf/q7-02-fullparallel-fpga.md` |
| 144/864 | 543 | VU47P, post-route | 995 k (76 %) | 97.3 MHz | 5.58 µs | same — 1.09× faster for 4× the area; the clock collapses on a fan-out-7 941 control net across three SLRs |
| 16/48 | 2 085 | ASAP7 7 nm predictive, OpenROAD, latch register file | 0.163 mm² die | 686 MHz setup-only | ~3.0 µs | `docs/perf/q7-02-asap7-timing.md`; 0.149 W / 0.31 µJ per window from a gate-level VCD |
| 144/864 | 543 | ASAP7 7 nm predictive | 0.869 mm² die, zero DRC | 615 MHz setup-only | **0.88 µs** | `docs/perf/q7-02-b3-asap7-fullparallel.md` |

Two honest caveats travel with the ASIC rows: ASAP7 is a predictive academic PDK, not a node this
project can tape out on (the sky130 probe of the same core is met2-congested and does not route,
`docs/perf/qec-q7-asic-sky130-probe.md`); and the ASAP7 netlists carry unrepaired hold violations on
the latch register-file clock (43 802 at 16/48), so gate-level co-sim fails and `Fmax` is
setup-only. Sub-microsecond needs ~543 MHz on 543 cycles; nothing in this family on any FPGA is
within 3× of that. The ASIC architecture and the open-silicon program that would fund it are
`docs/qec/asic-architecture.md` and `docs/qec/open-silicon-program.md`.

### Run it

```bash
brew install verilator          # ≥ 5.050 — 5.020 cannot compile the design, 5.032 rejects the banked core

# leaf datapaths
make -C hw checkminsum          # check_minsum vs C++ golden, STAGES-invariance
make -C hw varupdate            # var_update vs golden
make -C hw bpbenes bpasw        # permutation fabrics vs their reference permutation

# the ladder (each regenerates its header + vectors from the Rust model first)
make -C hw bpcheck              # M1 combinational check update, 432/432
make -C hw bprelay              # M2 sequential FSM, 65/65 code-capacity decodes
make -C hw bpunroll bpfast      # M4 unroll / M5 SAT-overlap (bppipe, bppartial, bppartialfast likewise)
make -C hw bpcirc               # M2 on the circuit-level graph
make -C hw bpbram bpbramfast bpbramdp bpbramdpearly   # M6 edge-serial BRAM family
make -C hw bpbanked             # M7/M8 banked core: 40/40 at 8/24, 12/36 AND 16/48 (the (W,V)-invariance gate)
make -C hw bpbanked-highweight  # 2 000 shots at p = 0.007 against the p = 0.007 header (matched-prior gate)
make -C hw bpbankedbram bpbankedbramm                 # register-file twins, same golden
make -C hw bpstream bpstreamaxi # M9b sliding-window core + AXI front-end, both exit modes
make -C hw bpstreambanked       # Q7-06 batched AXI4-Stream front-end
make -C hw bpaxibanked          # AXI4-Lite wrapper + register map, pre-silicon gate for the board build
make -C hw bpunrollcirc         # M4 unroll on the circuit-level golden
make -C hw bpgate-asap7         # gate-level sim of the ASAP7 netlist (needs the routed netlist on disk)
```

Every target regenerates `bb_gross_tanner.svh` before building. The banked targets end on the
committed at-rest header (`circgraph 1 0.003 16 48`) so `git status` stays clean; the code-capacity
targets (`bpcheck`, `bprelay`, `bpunroll`, `bpfast`, …) leave the small `graph` header behind —
run a circuit-level target last or `git checkout hw/bb_gross_tanner.svh` before committing.

Synthesis and boards (Vivado on an x86-Linux host; see `syn/README.md` for the shared OOC flow):

```bash
vivado -mode batch -source hw/syn/ooc_banked.tcl -tclargs 5.0 m8 bp_relay_banked     # OOC fit + Fmax probe
vivado -mode batch -source hw/syn/kv260_bp_stream_banked_bd.tcl -tclargs <proj> <out> # the shipped overlay
vivado -mode batch -source ../impl_vu47p.tcl -tclargs 5.0 b2                          # 64/192 on the AWS F2 part (run from the staged-source dir)
hw/syn/asic_probe.sh                                                                  # sky130 synth probe
```

On the board, the one-command path is `product/deploy.sh` (fetches `appliance-v1`, checks SHA-256,
runs the 40/40 self-test). The AWS build recipe is `docs/qec/b2-aws-build-runbook.md`.

### CI — what is gated

`.github/workflows/hw.yml` (Verilator 5.050, pinned) runs on every push to `main` and every PR that
touches `hw/**` or `crates/aleph-qec/**`:

1. `checkminsum`, `varupdate` — leaf datapaths vs golden.
2. `bpcheck` (M1), `bprelay` (M2) — code-capacity cores.
3. `bpbanked` (M8) — **the gate that matters**: the design on silicon and the ASIC target, at all
   three geometries.
4. `bpunrollcirc` (M4, circuit level).
5. `git diff --exit-code -- hw/` — the committed generated artefacts still match the generator.
   Order is load-bearing: a `circgraph`-emitting target must run last.

Slow gates (`bpbankedscale`, ~1 h; anything needing Vivado or a board) do not run in CI.

### Where the detail lives

| document | what |
|----------|------|
| `docs/perf/qec-q7-fixed-bp.md` | the master record: M0 word sweep → M1…M8 → M9a/b/c → Q7-05 power, every number with its command. |
| `docs/perf/q7-02-ngroup-sweep.md`, `q7-02-fullparallel-fpga.md`, `q7-02-asap7-timing.md`, `q7-02-b3-asap7-fullparallel.md`, `qec-q7-asic-sky130-probe.md` | scaling: NGROUP, VU47P, ASAP7 16/48 and 144/864, sky130. |
| `docs/qec/q7-06-ac1-batched-dma.md` | batched DMA path + the 3×10⁶-shot campaign and the matched-prior finding. |
| `docs/qec/q7-07-nonconvergence-policy.md` | `valid_flag` heralding, OSD tail candidates, the pre-registered decision. |
| `docs/qec/asic-architecture.md`, `regfile-plan.md`, `open-silicon-program.md` | the ASIC plan and the open-hardware program around it. |
| `product/` | the appliance: README, interface spec, bring-up, releasing, support policy. |
| `paper/` | the preprint that ties the ladder together (sections cite the records above). |

-----

## Phase Q6 — Union-Find surface-code decoder (sim-first FPGA track)

The first hardware program. Two boards — a **Digilent Zybo Z7-20** (Zynq-7020) and a **Xilinx Kria
KV260** (Zynq UltraScale+) — so the RTL targets both. The work was done **in simulation** first so
the RTL, testbench, and host↔hardware data flow were ready and verified before either board arrived,
then brought up on an Arty Z7-20. Everything below is still built and still passes; the Q7 program
reused its AXI wrappers, DMA streaming front-ends and co-simulation harness.

### What's here

Two decoders as synthesisable RTL, each verified in simulation against its Rust reference:

| file | role |
|------|------|
| `surface_d3_decoder.sv` | **Q6-01:** d=3 surface-code memory-Z decoder — an 8-bit syndrome indexes a 256-entry ROM → 1-bit correction, 1-cycle valid/valid handshake (the "syndrome-in / correction-out skeleton"). |
| `surface_d3_lut.mem` | its ROM — **generated by the Rust Union-Find decoder** (`crates/aleph-qec/examples/qec_d3_lut_table.rs`). |
| `tb_surface_d3.cpp` | Verilator TB: all 2^8 syndromes vs the Rust oracle + latency. |
| `uf_rep_decoder.sv` | **Q6-02 (stepping stone):** repetition-code **Union-Find / minimum-weight** decoder — a *real datapath* (prefix-XOR network + popcount + min-coset select), not a ROM. Full correction + logical flip, 1-cycle latency. |
| `tb_uf_rep.cpp` | Verilator TB: all 2^(D-1) syndromes — correction **reproduces the syndrome** and logical flip **matches the Rust `UnionFindDecoder`** (`qec_rep_uf_vectors.rs`). |
| `uf_surface_decoder.sv` | **Q6-04:** d=3 surface-code **Union-Find** decoder on the 2-D matching graph — cluster **growth** → spanning **forest** → **peeling** (`uf_surface_graph.svh`, generated by `qec_surface_uf_graph.rs`). **Synthesizable sequential FSM**: one bounded pass per cycle (Q6-02 was a single combinational `always_comb` cloud that can't close timing). Multi-cycle `in_valid → out_valid` handshake (`busy`, `latency_cycles`); 33 clk for d=3. |
| `tb_uf_surface.cpp` | Verilator TB: validity on all syndromes, **bit-for-bit equality vs the Q6-02 golden** (`uf_surface_golden.mem`), **distance-3 correctness** (all weight-1 errors corrected), and a weight-≤2 logical-error-rate comparison vs the CPU UF. |
| `uf_surface_golden.mem` | **Q6-04:** frozen `{obs_flip,correction}` table snapshotted from the Q6-02 combinational RTL; the regression lock for the sequential rewrite. Re-baseline via `make golden-rebaseline` + `tb_dump_golden.cpp`. |
| `tb_uf_surface_xsim.sv` | **Q6-06:** self-checking SV testbench for Vivado `xsim` — replays all 256 syndromes against behavioral RTL, the post-route functional netlist, and the post-route timing (SDF) netlist; checks golden bit-match + validity + no-X. Driven by `syn/gatesim.sh`. |
| `syn/gatesim.{tcl,sh}` | **Q6-06:** write funcsim/timesim netlists + SDF from a routed checkpoint and run the three `xsim` gate-level elaborations. |
| `tb_uf_cosim.cpp` | **Q6-21:** board-free sim↔RTL **co-simulation** — drives the decoder from the simulator's Monte-Carlo syndrome stream (`qec_q6_cosim.rs`) and checks the **RTL logical-error rate** vs the software Union-Find baseline within MC CI. Closes noise→syndrome→**RTL**→LER without a board (`docs/perf/qec-q6-cosim.md`). |
| `Makefile` | regenerate the reference tables, build with Verilator, run. |

### Run it

```bash
brew install verilator        # one-time (macOS)
make -C hw                    # build + run all decoders
make -C hw rep                # repetition-code UF
make -C hw surf               # surface-code UF (2-D)
make -C hw cosim              # Q6-21: board-free sim<->RTL co-sim (d=3, LER vs software UF)
make -C hw cosim-3d           # Q6-21: same, on the d=5x3 3-D space-time graph
make -C hw cosim-circuit      # Q6-21: same, on the d=3x3 CIRCUIT-LEVEL graph (hook errors)
```

Expected:
```
PASS: 256/256 syndromes match oracle; decode latency = 1 clock(s)
PASS: 64/64 syndromes valid + match Rust UF oracle; decode latency = 1 clock(s)
validity: PASS ...; golden bit-match: PASS ...; weight-1 correctness: PASS ...; latency = 33 clk
quality (wt<=2): RTL 40, CPU UF 50
```

### Why a lookup table first, then a datapath

A full syndrome→correction table is exponential in the detector count, so the d=3 LUT (Q6-01) is only
a *flow* proof. **Q6-02** is the real **Union-Find** datapath. The repetition-code core is the 1-D
specialisation (on a line, UF growth = minimum-weight matching = prefix-XOR / min-coset). The
**surface-code** decoder (`uf_surface_decoder.sv`) is the genuine 2-D engine — cluster growth, a
spanning forest, and peeling on the matching graph.

#### Verification note (surface UF)

The RTL UF is **valid** on all syndromes and **distance-3 correct** (corrects every weight-1 error).
It agrees with the CPU `UnionFindDecoder` on 171/256 syndromes; the rest are **logically degenerate**
(multiple equal-weight cosets) where UF tie-breaks legitimately differ — so we verify *decoder
quality* (validity + distance + a weight-≤2 logical-error-rate that matches/beats the CPU UF: 40 vs
50) rather than bit-identical agreement.

### Roadmap

- **Q6-01:** sim-first toolchain + d=3 LUT decoder skeleton, Verilator-verified. ✅ (sim)
- **Q6-02:** Union-Find datapath — repetition-code core ✅ + d=3 surface-code growth/peel ✅ (sim).
- **Q6-04:** synthesizable **sequential FSM** rewrite of the surface decoder ✅ (sim) — bounded
  per-cycle combinational depth, multi-cycle handshake, bit-identical to the Q6-02 golden.
- **Q6-05:** Vivado dual-target synth (XC7Z020 + XCK26) ✅ — fits with huge headroom; Fmax 58.7 /
  170 MHz; 33-clk decode = 562 / 194 ns, both within the 1 µs budget (`docs/perf/qec-q6-fpga.md`).
- **Q6-06:** gate-level sign-off ✅ — post-route functional + timing(SDF) xsim, 256/256 bit-match
  golden on both parts, no X.
- **Q6-07…Q6-09:** AXI PS↔PL wrapper, host software, d=5 scaling. **Q6-03:** GPU-vs-FPGA report
  (last; needs on-board numbers).
- **Q6-21:** board-free sim↔RTL **co-simulation** ✅ (sim) — the simulator's Monte-Carlo syndrome
  stream drives the Verilated decoder; the RTL logical-error rate matches the software UF within CI
  at d=3 (every p) and sub-threshold at d=5×3, with the supra-threshold unweighted-UF quality gap
  surfaced honestly (`docs/perf/qec-q6-cosim.md`). The board-free stand-in for full HiL; swaps the
  Verilated model for the real board over the Q6-07 AXI link at Q6-08.

Targeted at **two boards** — Digilent **Zybo Z7-20** (`xc7z020clg400-1`) and Xilinx **Kria KV260**
(`xck26-sfvc784-2LV-c`). Synthesis is board-independent; only final bring-up needs hardware. Boards
self-program / load over JTAG; Vivado (x86-Linux only) builds the bitstream. See `docs/qec/BACKLOG.md`
Phase Q6 and the project memory for hardware specifics.

### Q6-07 — AXI PS↔PL wrapper

`uf_axi_wrap.sv` wraps the decoder for the Zynq PS over two standard interfaces (identical on
Zynq-7020 and Zynq UltraScale+): an **AXI4-Lite** slave (control plane) and an **AXI4-Stream** pair
(syndrome in / correction out — the data plane the Q4 streaming maps onto). A single decode-owner
FSM serves whichever interface triggered the decode; the decoder core is unchanged.

`tb_uf_axi_xsim.sv` (run via `syn/axisim.sh` on a Vivado host) drives **all 256 syndromes through
both planes** and checks them bit-for-bit against `uf_surface_golden.mem` → `RESULT: PASS`.

**AXI4-Lite register map** (32-bit data, byte addresses):

| addr | name | access | meaning |
|------|------|--------|---------|
| 0x00 | `CTRL`       | W  | bit0 `START` (self-clearing): latch `SYNDROME`, run one decode |
| 0x04 | `STATUS`     | R  | bit0 `BUSY`, bit1 `DONE` (sticky, cleared on next `START`), bit2 `OBS_FLIP` |
| 0x08 | `SYNDROME`   | RW | `syndrome[SYN_W-1:0]` |
| 0x0C | `CORRECTION` | R  | `correction[M-1:0]` |
| 0x10 | `LATENCY`    | R  | last decode latency in cycles |
| 0x14 | `IDCODE`     | R  | `0x5546_0003` constant ('UF', d=3) for bring-up sanity |

AXI4-Stream: write a syndrome word on `s_axis` (one beat per frame); the wrapper emits
`{obs_flip, correction}` on `m_axis` with `tlast` per frame.

### Q6-08 — board bring-up (Arty Z7-20, PYNQ, measured silicon)

The decoder now runs on real hardware. Board: **Digilent Arty Z7-20** (`xc7z020clg400-1`, the same PL
part as the Zybo Z7-20 target), booted from the PYNQ-Z1 v3.1.1 image over LAN.

| file | role |
|------|------|
| `uf_axi_top.v` | Verilog-2001 board top: instantiates `uf_axi_wrap` exposing only the AXI4-Lite control plane, ties off the AXI4-Stream data plane. (Vivado forbids a SystemVerilog file as the top of a block-design module reference, so this thin structural top is Verilog; the submodules stay SV.) |
| `syn/arty_z7_bd.tcl` | Vivado **block design + bitstream** (not the OOC study): Zynq-7 PS + `uf_axi_top` on the PS GP0 AXI master, FCLK 50 MHz → `<name>.bit` + `<name>.hwh` for PYNQ. No Digilent board files needed (generic PS7; DDR/MIO from the image FSBL, PL clock applied by PYNQ from the `.hwh`). |
| `sw/uf_pynq.py` | PYNQ/Python host driver (twin of the bare-metal C `sw/uf_decoder.c`): loads the overlay, drives the AXI4-Lite regmap, checks all 256 syndromes vs golden, reports latency. Also runs a board-free software-model self-test. |

Build the bitstream (Vivado on an x86 Linux host):
```bash
vivado -mode batch -source syn/arty_z7_bd.tcl -tclargs <proj_dir> <out_dir>
```
Run on the board (root + XRT env; PYNQ lives in a venv):
```bash
sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 uf_pynq.py uf_arty.bit uf_surface_golden.mem
```
Measured d=3: `WNS +7.29 ns (TIMING_MET)`, **256/256 bit-identical to golden**, IDCODE ok, worst
decode **600 ns @ 50 MHz** — under the 1 µs round budget (real-time on silicon). Closes the on-board
ACs of Q6-01/Q6-02/Q6-08. Details in `docs/perf/qec-q6-fpga.md`.

#### Decoder-bound throughput (AXI DMA)

| file | role |
|------|------|
| `uf_stream_core.sv` | pure AXI4-Stream engine over the `uf_surface_decoder` core (syndrome in → decode → `{obs,corr}` out), tlast propagated input→output so one DMA transfer streams a whole batch. |
| `uf_stream.v` | Verilog board top for the BD module reference (fixed 32-bit AXIS, distance-independent). |
| `syn/arty_z7_dma_bd.tcl` | block design with **AXI DMA** (MM2S+S2MM) feeding the engine from PS DDR and back — the PS is out of the per-decode loop. |
| `sw/uf_dma.py` | PYNQ driver: DMA a batch through the decoder, measure throughput, re-check LER. |
| `uf_stream_array_core.sv` | K decoder engines behind one AXI4-Stream (round-robin dispatch + in-order collect) — replicates the engine across the free fabric. |
| `uf_stream_array.v` | Verilog board top for the array (parameter `K`); `arty_z7_dma_bd.tcl <proj> <out> <fclk> <K> [ooo]` picks single-engine (K≤1) or the K-way array. |
| `uf_stream_array_ooo_core.sv` | out-of-order variant: dispatch to any free engine + a **reorder buffer** (in-order output), removing round-robin head-of-line stalls. `uf_stream_array_ooo.v` is its Verilog top (5th tcl arg `1`). |

Measured d=3: single engine **1.39 M decodes/s (0.72 µs/decode) — ~191× the PS-polled AXI4-Lite path,
decoder-bound**. Replicated: **K=8 → 9.55 M/s (6.56×); K=16 → 16.5 M/s (11.3×, 32 % LUT)**, all
timing-met, LER unchanged. Out-of-order lifts per-engine efficiency (K=8: 82 % → **95 %**, 11.0 M/s) but
its reorder buffer is O(K²) area — OOO K=8 ≈ in-order K=16 in LUTs, and OOO K=16 fails to route; so on
this small part more engines beat reordering. This is the Q6-03 FPGA throughput figure; see
`docs/perf/qec-q6-fpga.md` for the full table and verdict.

## Licence — Apache-2.0, not MIT

**Everything under `hw/` is licensed under Apache-2.0** ([`hw/LICENSE`](LICENSE)), unlike the Rust
crates in this repository, which stay MIT ([`../LICENSE`](../LICENSE)).

The difference is deliberate and it is about patents, not about openness. MIT grants copyright
permission but says nothing about patents, which is a routine blocker for anyone whose legal team has
to approve pulling RTL into a chip they will fabricate — the exact use we want to enable. Apache-2.0
carries an explicit patent grant (§3) and a defensive termination clause, which is why it is the
standard choice for open silicon (OpenTitan and most RISC-V cores use it).

Copyright © 2026 Ruslan Malymon and the aleph contributors. `hw/LICENSE` is the canonical Apache-2.0
text, unmodified (SHA-256 `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`); the
copyright statement lives here rather than being pasted into the licence template.

If you are integrating this RTL and the licence is still an obstacle, open an issue — the point of the
project is that the design gets used.
