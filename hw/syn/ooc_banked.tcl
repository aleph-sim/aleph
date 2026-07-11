# Q7-02 M7 — OOC fit+Fmax probe for bp_relay_banked (the beta-split banked core).
# Usage: vivado -mode batch -source ooc_banked.tcl -tclargs [period_ns] [label]
# Run from a dir holding check_minsum.sv var_update.sv bp_relay_banked.sv bb_gross_tanner.svh
# (the header bakes the (W,V) config — stage one dir per probed config).
set period [expr {$argc >= 1 ? double([lindex $argv 0]) : 5.0}]
set label  [expr {$argc >= 2 ? [lindex $argv 1] : "wv"}]
set part   xck26-sfvc784-2LV-c

read_verilog -sv {check_minsum.sv var_update.sv bp_relay_banked.sv}
synth_design -top bp_relay_banked -part $part -mode out_of_context \
  -flatten_hierarchy none -include_dirs [pwd]

create_clock -name clk -period $period [get_ports clk]
report_utilization -file util_banked.rpt
report_utilization -hierarchical -hierarchical_depth 2 -file util_hier.rpt
report_timing_summary -delay_type max -file timing_banked.rpt

set lut    [llength [get_cells -hier -filter {REF_NAME =~ LUT*}]]
set ff     [llength [get_cells -hier -filter {IS_SEQUENTIAL}]]
set carry  [llength [get_cells -hier -filter {REF_NAME =~ CARRY*}]]
set dsp    [llength [get_cells -hier -filter {REF_NAME =~ DSP*}]]
set bram   [llength [get_cells -hier -filter {REF_NAME =~ RAMB*}]]
set wns    [get_property SLACK [lindex [get_timing_paths -max_paths 1 -nworst 1 -setup] 0]]
set fmax   [expr {1000.0/($period - $wns)}]
# authoritative counts from the util report: CLB LUTs + LUT-as-memory (the LUTRAM banks)
set fh [open util_banked.rpt r]; set t [read $fh]; close $fh
set clut -1; set lutram -1
regexp {(?:CLB|Slice) LUTs\s*\|\s*([0-9]+)} $t -> clut
regexp {LUT as Memory\s*\|\s*([0-9]+)} $t -> lutram

puts [format "RESULT %s CLBLUT=%s LUTRAM=%s cellLUT=%d FF=%d CARRY8=%d DSP=%d RAMB=%d period=%.2f WNS=%.3f Fmax=%.1fMHz" \
  $label $clut $lutram $lut $ff $carry $dsp $bram $period $wns $fmax]
