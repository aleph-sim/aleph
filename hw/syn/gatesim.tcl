# Q6-06 — write functional + timing simulation netlists (+SDF) from an implemented checkpoint.
# Usage: vivado -mode batch -source gatesim.tcl -tclargs <post_route.dcp> <outdir>
set dcp    [lindex $argv 0]
set outdir [lindex $argv 1]
file mkdir $outdir
open_checkpoint $dcp
write_verilog -force -mode funcsim [file join $outdir funcsim.v]
write_verilog -force -mode timesim [file join $outdir timesim.v]
write_sdf     -force               [file join $outdir timesim.sdf]
close_design
