#!/usr/bin/env bash
# Q6-06 — Vivado xsim gate-level sign-off for the surface UF decoder.
# Runs the SAME self-checking SV TB (hw/tb_uf_surface_xsim.sv) over three elaborations:
#   0. behavioral RTL          (TB sanity)
#   1. post-route FUNCTIONAL   (synth/sim mismatch, latches)
#   2. post-route TIMING (SDF) (X-prop / setup at the closed clock)
# Requires Vivado on PATH (settings64.sh sourced). Run from hw/syn/.
#   ./gatesim.sh <report-dir-with-post_route.dcp>   e.g. ./gatesim.sh reports/zybo
set -euo pipefail
cd "$(dirname "$0")"
HW="$(cd .. && pwd)"
PARTDIR="${1:-reports/zybo}"
OUT="gatesim/$(basename "$PARTDIR")"
mkdir -p "$OUT"
GLBL="$XILINX_VIVADO/data/verilog/src/glbl.v"

pass() { if grep -qa "RESULT: PASS" "$1"; then echo "  -> PASS"; else echo "  -> FAIL"; tail -12 "$1"; exit 1; fi; }

echo "==================== 0. behavioral RTL ===================="
rm -rf "$OUT/beh" && mkdir -p "$OUT/beh" && pushd "$OUT/beh" >/dev/null
cp "$HW/uf_surface_golden.mem" .
# Compile RTL and TB in SEPARATE xvlog calls: both `include the graph svh, so a single compilation
# unit double-declares its localparams. Separate units keep each file's $unit scope independent.
xvlog -sv -i "$HW" "$HW/uf_surface_decoder.sv" >xvlog_rtl.log 2>&1
xvlog -sv -i "$HW" "$HW/tb_uf_surface_xsim.sv" >xvlog_tb.log 2>&1
xelab tb_uf_surface_xsim -s beh >xelab.log 2>&1
xsim beh -runall >run.log 2>&1; cat run.log | grep -aE "xsim:|RESULT"; pass run.log
popd >/dev/null

echo "==================== 1. write netlists from $PARTDIR/post_route.dcp ===================="
vivado -mode batch -nojournal -log "$OUT/write.log" -source gatesim.tcl \
       -tclargs "$PARTDIR/post_route.dcp" "$OUT"
ls -lh "$OUT"/{funcsim.v,timesim.v,timesim.sdf}

echo "==================== 2. post-route FUNCTIONAL sim ===================="
rm -rf "$OUT/func" && mkdir -p "$OUT/func" && pushd "$OUT/func" >/dev/null
cp "$HW/uf_surface_golden.mem" .
xvlog -sv -i "$HW" "$HW/tb_uf_surface_xsim.sv" >xvlog_tb.log 2>&1
xvlog ../funcsim.v >xvlog_net.log 2>&1
xvlog "$GLBL" >xvlog_glbl.log 2>&1
xelab -L unisims_ver -L secureip tb_uf_surface_xsim glbl -s func >xelab.log 2>&1
xsim func -runall >run.log 2>&1; cat run.log | grep -aE "xsim:|RESULT"; pass run.log
popd >/dev/null

echo "==================== 3. post-route TIMING sim (SDF, 50 MHz) ===================="
rm -rf "$OUT/timing" && mkdir -p "$OUT/timing" && pushd "$OUT/timing" >/dev/null
cp "$HW/uf_surface_golden.mem" .
xvlog -sv -i "$HW" "$HW/tb_uf_surface_xsim.sv" >xvlog_tb.log 2>&1
xvlog ../timesim.v >xvlog_net.log 2>&1
xvlog "$GLBL" >xvlog_glbl.log 2>&1
xelab -L simprims_ver -L secureip -generic_top "HALF_NS=10" \
      -sdfmax tb_uf_surface_xsim/dut=../timesim.sdf -transport_int_delays -pulse_r 0 -pulse_int_r 0 \
      tb_uf_surface_xsim glbl -s timing >xelab.log 2>&1
xsim timing -runall >run.log 2>&1; cat run.log | grep -aE "xsim:|RESULT"; pass run.log
popd >/dev/null

echo "==================== ALL GATE-LEVEL SIGN-OFF PASSED ($PARTDIR) ===================="
