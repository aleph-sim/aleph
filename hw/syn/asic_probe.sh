#!/bin/sh
# ASIC synth probe — SKY130 HD standard-cell mapping of a relay-BP core (Q7-01 input).
#
# Flow: slang SV frontend -> generic synth -> memories collected as $mem_v2 and KEPT as
# blackbox boundaries (on an ASIC they become SRAM/ROM macros; "Number of memory bits" in the
# stat block is the macro budget; deleting them instead would let opt_clean sweep the whole
# in-loop datapath) -> techmap -> dfflibmap/abc against sky130_fd_sc_hd tt_025C_1v80 -> stat.
# The final `stat -liberty` "Chip area" is standard-cell logic area in um^2 (no routing, no
# macros); abc's reported delay at -D <ps> is the first-order critical path.
#
# usage: probe.sh <workdir> <top> <period_ps> <src...>
set -e
WD=$1; TOP=$2; PER=$3; shift 3
LIB=/data/asicprobe/sky130_fd_sc_hd__tt_025C_1v80.lib
export PATH=/data/asicprobe/oss-cad-suite/bin:$PATH
cd "$WD"

cat > ${TOP}_probe.ys <<EOF
plugin -i slang
read_slang -I . --top ${TOP} --unroll-limit=1000000 $*
hierarchy -top ${TOP} -check
proc
flatten
opt
memory -nomap
opt_clean
stat
tee -q -o ${TOP}_mems.txt dump t:\$mem_v2
techmap
opt -fast
dfflibmap -liberty ${LIB}
abc -liberty ${LIB} -D ${PER} -script "+strash;&get,-n;&fraig,-x;&put;scorr;dc2;dretime;strash;&get,-n;&dch,-f;&nf,-D,${PER};&put;topo;stime"
setundef -zero
clean -purge
stat -liberty ${LIB}
write_blif -gates ${TOP}_mapped.blif
EOF

exec yosys -l ${TOP}_sky130.log ${TOP}_probe.ys
