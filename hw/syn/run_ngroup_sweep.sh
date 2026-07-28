#!/bin/bash
# Q7 Task B2 — Fmax/area half of the NGROUP sweep for `bp_relay_unroll_pipe`.
#
# Runs the OOC probe (ooc_core.tcl) once per NGROUP, SERIALLY: the fully-unrolled end of this design
# peaked at 47.9 GB / 17 processes in the Task-B0 probe, so two concurrent runs would take the box down.
# Big NGROUP first — those are the cheap, small designs, so the area/Fmax trend lands early even if the
# small-NGROUP tail has to be abandoned.
#
# Run it detached or it dies with the ssh session:
#   nohup setsid ./run_ngroup_sweep.sh >/dev/null 2>&1 </dev/null &
set -u
cd "$(dirname "$0")"
export PATH=/tools/Xilinx/Vivado/2024.2/bin:$PATH

PERIOD=5.0                     # same 5.0 ns target the Task-B0 M4 probe used, so Fmax is comparable
NG_LIST=(144 72)                # floor probe: at these the stamped arithmetic is negligible, so what
                                # is left IS the NGROUP-invariant crossbar. Downward tail (12..4) was
                                # cut after 48/24/16 confirmed area grows as NGROUP shrinks.

echo "SWEEP period=$PERIOD groups=${NG_LIST[*]}" >>sweep_summary.txt   # append: never drop earlier points
for g in "${NG_LIST[@]}"; do
  echo "== NGROUP=$g started $(date -u +%FT%TZ)" >>sweep_summary.txt
  vivado -mode batch -source ooc_core.tcl -tclargs "$PERIOD" "$g" >"ooc_ng$g.log" 2>&1
  rc=$?
  mv -f util_core.rpt "util_ng$g.rpt" 2>/dev/null
  mv -f timing_core.rpt "timing_ng$g.rpt" 2>/dev/null
  line=$(grep -m1 '^RESULT ' "ooc_ng$g.log")
  echo "NGROUP=$g rc=$rc ${line:-NO-RESULT} finished $(date -u +%FT%TZ)" >>sweep_summary.txt
done
echo "SWEEP_DONE" >>sweep_summary.txt
