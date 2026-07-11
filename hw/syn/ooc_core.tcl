# Q7-02 M7 — OOC fit+Fmax probe for bp_relay_unroll_pipe (the partial-unroll core).
# Usage: vivado -mode batch -source ooc_core.tcl -tclargs [period_ns] [NGROUP]
# Run from a dir holding check_minsum.sv var_update.sv bp_relay_unroll_pipe.sv bb_gross_tanner.svh.
set period [expr {$argc >= 1 ? double([lindex $argv 0]) : 5.0}]
set part   xck26-sfvc784-2LV-c

read_verilog -sv {check_minsum.sv var_update.sv bp_relay_unroll_pipe.sv}
if {$argc >= 2} {
  synth_design -top bp_relay_unroll_pipe -part $part -mode out_of_context \
    -flatten_hierarchy none -directive RuntimeOptimized -include_dirs [pwd] -generic NGROUP=[lindex $argv 1]
} else {
  synth_design -top bp_relay_unroll_pipe -part $part -mode out_of_context \
    -flatten_hierarchy none -directive RuntimeOptimized -include_dirs [pwd]
}

create_clock -name clk -period $period [get_ports clk]
report_utilization -file util_core.rpt
report_timing_summary -delay_type max -file timing_core.rpt

set lut    [llength [get_cells -hier -filter {REF_NAME =~ LUT*}]]
set ff     [llength [get_cells -hier -filter {IS_SEQUENTIAL}]]
set carry  [llength [get_cells -hier -filter {REF_NAME =~ CARRY*}]]
set dsp    [llength [get_cells -hier -filter {REF_NAME =~ DSP*}]]
set wns    [get_property SLACK [lindex [get_timing_paths -max_paths 1 -nworst 1 -setup] 0]]
set fmax   [expr {1000.0/($period - $wns)}]
# authoritative CLB LUT count from the util report
set fh [open util_core.rpt r]; set t [read $fh]; close $fh
set clut -1
regexp {(?:CLB|Slice) LUTs\s*\|\s*([0-9]+)} $t -> clut

puts [format "RESULT CLBLUT=%s cellLUT=%d FF=%d CARRY8=%d DSP=%d period=%.2f WNS=%.3f Fmax=%.1fMHz" \
  $clut $lut $ff $carry $dsp $period $wns $fmax]
