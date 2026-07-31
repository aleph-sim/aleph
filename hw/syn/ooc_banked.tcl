# Q7-02 M7 — OOC fit+Fmax probe for the banked relay-BP cores.
# Usage: vivado -mode batch -source ooc_banked.tcl -tclargs [period_ns] [label] [top]
# Run from a dir holding ALL .sv sources for the probed top plus bb_gross_tanner.svh
# (the header bakes the (W,V) config — stage one dir per probed config; every *.sv in
# the dir is read, so the staging dir defines the file set).
set period [expr {$argc >= 1 ? double([lindex $argv 0]) : 5.0}]
set label  [expr {$argc >= 2 ? [lindex $argv 1] : "wv"}]
set top    [expr {$argc >= 3 ? [lindex $argv 2] : "bp_relay_banked"}]
set part   xck26-sfvc784-2LV-c

read_verilog -sv [lsort [glob *.sv]]
synth_design -top $top -part $part -mode out_of_context \
  -flatten_hierarchy none -include_dirs [pwd]

create_clock -name clk -period $period [get_ports clk]
report_utilization -file util_banked.rpt
report_utilization -hierarchical -hierarchical_depth 2 -file util_hier.rpt
report_timing_summary -delay_type max -file timing_banked.rpt

set lut    [llength [get_cells -hier -filter {REF_NAME =~ LUT*}]]
set ff     [llength [get_cells -hier -filter {IS_SEQUENTIAL}]]
set carry  [llength [get_cells -hier -filter {REF_NAME =~ CARRY*}]]
# DSP48E2 exactly: `REF_NAME =~ DSP*` also matches the ~9 sub-primitives each DSP48E2 macro expands
# into (DSP_ALU, DSP_A_B_DATA, ...), inflating the count 9x — the M7 report shipped with that error.
set dsp    [llength [get_cells -hier -filter {REF_NAME == DSP48E2}]]
set bram   [llength [get_cells -hier -filter {REF_NAME =~ RAMB*}]]
set uram   [llength [get_cells -hier -filter {REF_NAME == URAM288}]]
set wns    [get_property SLACK [lindex [get_timing_paths -max_paths 1 -nworst 1 -setup] 0]]
set fmax   [expr {1000.0/($period - $wns)}]
# authoritative counts from the util report: CLB LUTs + LUT-as-memory (the LUTRAM banks)
# The row label carries a trailing '*' (footnote marker) — `CLB LUTs*` — and the older
# `LUTs\s*\|` pattern silently failed on it and reported -1 for the whole B2 Step 0 sweep.
# Match anything up to the column separator instead. Same trap as ooc_core.tcl.
set fh [open util_banked.rpt r]; set t [read $fh]; close $fh
set clut -1; set lutram -1
regexp {(?:CLB|Slice) LUTs[^|]*\|\s*([0-9]+)} $t -> clut
regexp {LUT as Memory\s*\|\s*([0-9]+)} $t -> lutram

puts [format "RESULT %s CLBLUT=%s LUTRAM=%s cellLUT=%d FF=%d CARRY8=%d DSP=%d RAMB=%d URAM=%d period=%.2f WNS=%.3f Fmax=%.1fMHz" \
  $label $clut $lutram $lut $ff $carry $dsp $bram $uram $period $wns $fmax]
