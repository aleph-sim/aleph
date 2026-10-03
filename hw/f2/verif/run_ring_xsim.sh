#!/usr/bin/env bash
# P2 appliance v2-large — fast unit gate for the PCIS batch path (aleph_pcis_ring + stand-in decoder).
#   verif/run_ring_xsim.sh [seed]     from a staged CL dir or hw/f2; needs Vivado on PATH; ~1 min
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
rm -rf xsim_ring && mkdir xsim_ring && cd xsim_ring
xvlog -sv "$here/design/aleph_pcis_ring.sv" "$here/verif/tb_aleph_pcis_ring.sv" > xvlog.log
xelab -debug off tb_aleph_pcis_ring -s tb > xelab.log
xsim tb -R -sv_seed "${1:-1}" | tee xsim.log
grep -q '^PASS' xsim.log
