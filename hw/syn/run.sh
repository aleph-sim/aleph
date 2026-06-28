#!/usr/bin/env bash
# Q6-05 — run the dual-target OOC synth/impl Fmax+utilization study on BOTH boards' parts.
# Requires Vivado on PATH (x86 Linux; we run this on openwebgui). Run from anywhere:
#   hw/syn/run.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p reports

echo "=== Zybo Z7-20  (xc7z020clg400-1) ==="
vivado -mode batch -nojournal -log reports/zybo.vivado.log \
       -source synth.tcl -tclargs xc7z020clg400-1 zybo_z7_20.xdc reports/zybo

echo "=== Kria KV260  (xck26-sfvc784-2LV-c) ==="
vivado -mode batch -nojournal -log reports/kv260.vivado.log \
       -source synth.tcl -tclargs xck26-sfvc784-2LV-c kv260.xdc reports/kv260

echo
echo "=== Fmax ==="
cat reports/zybo/fmax.txt reports/kv260/fmax.txt
echo
echo "=== Utilization (impl) ==="
grep -hE "Slice LUTs|CLB LUTs|Slice Registers|CLB Registers|Block RAM Tile|DSPs" \
     reports/zybo/util_impl.rpt reports/kv260/util_impl.rpt || true
