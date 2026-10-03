#!/bin/bash
# P2 appliance v2-large — DCP build for the aleph F2 CL, run ON the AWS build instance (FPGA Developer AMI).
#
#   run_build.sh <staged_cl_dir> <mhz> [<mhz> ...]
#
# For each requested decoder clock (125 and/or 150) this copies the staged CL into its own aws-fpga
# example dir and runs the HDK flow in parallel, so two clocks cost one wall-clock build. Results land in
# /scratch/result_<mhz>/ (DCP tarball, timing reports, utilisation/clock/CDC reports from the routed
# checkpoint via build/scripts/aleph_reports.tcl, log tail) and /scratch/summary.txt.
# It does NOT shut the instance down: copy the results off first, then terminate from outside.
#
# Source the tools before `set -u` (Vivado's settings chain dereferences unset variables).
source /opt/Xilinx/2025.2/Vivado/settings64.sh 2>/dev/null || true
set -u

staged="${1:?usage: run_build.sh <staged_cl_dir> <mhz>...}"; shift
clocks=("$@")
set --   # a sourced script sees the caller's positional args; hdk_setup.sh rejects ours as options
cd /scratch
[ -d aws-fpga ] || git clone -q --depth 1 -b f2 https://github.com/aws/aws-fpga.git
cd aws-fpga
# hdk_setup.sh downloads the shell DCPs and checks the Vivado version; it must be sourced, in bash.
source hdk_setup.sh > /scratch/hdk_setup.log 2>&1 || { echo "hdk_setup FAILED" >> /scratch/summary.txt; exit 1; }

for mhz in "${clocks[@]}"; do
  (
    name=cl_aleph_bp
    dir="/scratch/cl_$mhz/$name"        # CL name must match the top module; separate parents per clock
    rm -rf "/scratch/cl_$mhz" && mkdir -p "/scratch/cl_$mhz" && cp -r "$staged" "$dir"
    export CL_DIR="$dir" ALEPH_CORE_MHZ="$mhz"
    cd "$dir/build/scripts"
    # The flow runs from here and sources these by relative name. common/'s encrypt.tcl is the generic one
    # that takes every file in design/, which is what this CL wants.
    common="$HDK_DIR/common/shell_stable/build/scripts"
    ln -sf "$common/aws_build_dcp_from_cl.py" "$common/build_all.tcl" "$common/build_level_1_cl.tcl" .
    cp "$common/encrypt.tcl" .
    start=$(date +%s)
    ./aws_build_dcp_from_cl.py --cl "$name" --mode small_shell > "/scratch/build_$mhz.out" 2>&1
    rc=$?
    out="/scratch/result_$mhz"; mkdir -p "$out"
    cp "$dir"/build/checkpoints/*.Developer_CL.tar "$out/" 2>/dev/null
    cp -r "$dir"/build/reports "$out/" 2>/dev/null
    tail -40 "$dir"/build/scripts/*.vivado.log > "$out/vivado_tail.log" 2>/dev/null
    violated=$(ls "$dir"/build/checkpoints/ 2>/dev/null | grep -c VIOLATED)
    # Utilisation, clock and CDC reports from the routed checkpoint (the flow itself writes timing only).
    dcp=$(ls "$dir"/build/checkpoints/*.post_route*.dcp 2>/dev/null | head -1)
    if [ -n "$dcp" ]; then
      ( cd "$out" && vivado -mode batch -nojournal -log reports_vivado.log \
          -source "$dir/build/scripts/aleph_reports.tcl" -tclargs "$dcp" "$out/reports" > /dev/null 2>&1 )
    fi
    echo "$mhz rc=$rc minutes=$(( ($(date +%s) - start) / 60 )) violated=$violated tar=$(ls "$out"/*.tar 2>/dev/null)" \
      "util=$(ls "$out"/reports/util_cl.rpt 2>/dev/null)" >> /scratch/summary.txt
  ) &
done
wait
echo DONE >> /scratch/summary.txt
