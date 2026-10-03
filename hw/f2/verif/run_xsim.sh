#!/usr/bin/env bash
# P2 appliance v2-large — xsim gate for aleph_bp_ocl. Run from a staged CL dir (see ../stage.sh):
#   verif/run_xsim.sh <bp_circ_vectors.txt> [150]
# Needs Vivado on PATH (any 2024.x/2025.x: XPM and unisim models are part-independent).
# Slow: the 64/192 core is a large combinational design and xsim evaluates it every cycle (elaboration
# ~30 min, then tens of minutes per 40 decodes). Run it detached and logged, never piped into ssh:
#   setsid nohup verif/run_xsim.sh bp_circ_vectors.txt > xsim.out 2>&1 < /dev/null &
set -euo pipefail
vec="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
if [ "${2:-125}" = "150" ]; then DIV=10.0; KHZ=150000; else DIV=12.0; KHZ=125000; fi
V="$(dirname "$(dirname "$(command -v vivado)")")"
rm -rf xsim_run && mkdir xsim_run && cd xsim_run
xvlog -sv -d TB_DIV_F=$DIV -d TB_KHZ=$KHZ -i ../design \
  "$V/data/ip/xpm/xpm_cdc/hdl/xpm_cdc.sv" "$V/data/ip/xpm/xpm_memory/hdl/xpm_memory.sv" \
  "$V/data/ip/xpm/xpm_fifo/hdl/xpm_fifo.sv" \
  ../design/check_minsum.sv ../design/var_update.sv ../design/bp_relay_banked.sv \
  ../design/bp_stream_banked_core.sv ../design/aleph_pcis_ring.sv ../design/aleph_bp_ocl.sv ../verif/tb_aleph_bp_ocl.sv > xvlog.log
xvlog "$V/data/verilog/src/glbl.v" >> xvlog.log
xelab -L unisims_ver -debug off -timescale 1ns/1ps tb_aleph_bp_ocl glbl -s tb > xelab.log
xsim tb -R --testplusarg "VEC=$vec" | tee xsim.log
grep -q '^PASS' xsim.log
