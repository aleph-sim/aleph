#!/usr/bin/env bash
# Q6-07 — Vivado xsim behavioral sign-off for the AXI PS<->PL wrapper (uf_axi_wrap).
# Drives all 256 syndromes through AXI4-Lite and AXI4-Stream and checks them against the golden table.
# Requires Vivado on PATH (settings64.sh sourced). Run from hw/syn/.
set -euo pipefail
cd "$(dirname "$0")"
HW="$(cd .. && pwd)"
rm -rf xsim_axi && mkdir xsim_axi && cd xsim_axi
cp "$HW/uf_surface_golden.mem" .
# RTL files each `include the graph header → compile in separate units to avoid double-declaration.
xvlog -sv -i "$HW" "$HW/uf_surface_decoder.sv" >x1.log 2>&1
xvlog -sv -i "$HW" "$HW/uf_axi_wrap.sv"         >x2.log 2>&1
xvlog -sv -i "$HW" "$HW/tb_uf_axi_xsim.sv"      >x3.log 2>&1
xelab tb_uf_axi_xsim -s axi >xe.log 2>&1
xsim axi -runall >run.log 2>&1
grep -aE "axi:|RESULT" run.log
grep -qa "RESULT: PASS" run.log
