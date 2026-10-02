#!/usr/bin/env bash
# P2 appliance v2-large — assemble the F2 CL directory (what the AWS HDK calls $CL_DIR).
#
#   hw/f2/stage.sh <out_dir> [W V]       default geometry 64/192
#
# Copies this directory's design/, build/ and verif/ plus the shared decoder RTL from hw/, and generates
# the core header for the requested bank geometry. The header and the copied RTL are not checked in under
# hw/f2: they are derived, and one source of truth lives in hw/ (the RTL) and aleph-qec (the header).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
out="${1:?usage: stage.sh <out_dir> [W V]}"
W="${2:-64}"; V="${3:-192}"

mkdir -p "$out"
cp -r "$here/design" "$here/build" "$here/verif" "$out/"
for f in bp_stream_banked_core.sv bp_relay_banked.sv check_minsum.sv var_update.sv; do
  cp "$repo/hw/$f" "$out/design/"
done
( cd "$repo" && cargo run --release -q -p aleph-qec --example qec_q7_bp_graph -- circgraph 1 0.003 "$W" "$V" ) \
  > "$out/design/bb_gross_tanner.svh"
grep -q "BP_BANK_W = $W;" "$out/design/bb_gross_tanner.svh" || { echo "header is not $W/$V" >&2; exit 1; }
echo "staged CL for $W/$V in $out"
