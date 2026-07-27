# Q7-02 B0 — OOC fit+Fmax probe for the M4 SPATIALLY-UNROLLED core (`bp_relay_unrolled`).
#
# Modelled on ooc_core.tcl (the M7 probe for bp_relay_unroll_pipe); same part, same OOC flow, same
# RESULT line format, so the numbers drop straight into the same comparison table.
#
# This is the decisive measurement for the whole silicon programme, not a routine utilisation check.
# B0/Option B established that the unrolled core decodes the circuit-level DEM golden bit-exactly in
# 181 cycles, against 913 for the banked core at 64/192 and 544 for the 144/864 banked configuration
# that cannot be generated at all. At 181 cycles, sub-microsecond needs only ~200 MHz. So:
#
#   fits an affordable FPGA at ~200 MHz  -> the ASIC is no longer needed for LATENCY; the silicon
#                                           case shrinks to power and embeddability, and the EUR 45-93k
#                                           is better spent elsewhere.
#   does not fit                         -> the ASIC case is stronger than it has ever been, and now
#                                           with an exact number attached to what it buys.
#
# Expect this to be expensive or to fail outright: arty_z7_bp_circ_bd.tcl records that the flop-array
# M2 core (bp_relay_decoder) OOMed Vivado at this same BP_E=2952 scale (PR #447). M4 deletes M2's
# runtime cursor mux -- the thing M3 diagnosed as the wall -- so it is genuinely a different netlist,
# but a spatial unroll of 2952 edges is still the largest thing this project has asked Vivado to do.
# A crash or an OOM is a RESULT here, not a failure of the experiment: it bounds the FPGA option.
#
# Usage: vivado -mode batch -source ooc_unrolled.tcl -tclargs [period_ns]
# Run from a directory holding bp_relay_unrolled.sv + bb_gross_tanner.svh (the CIRCUIT-level header,
# emitted by `qec_q7_bp_graph -- circgraph`, not the code-capacity `graph` one).
#
# 5.0 ns default = 200 MHz, i.e. exactly the clock at which 181 cycles lands under 1 us. That is the
# number we actually care about, so it is the default rather than an arbitrary round figure.

set period [expr {$argc >= 1 ? double([lindex $argv 0]) : 5.0}]
set part   xck26-sfvc784-2LV-c

read_verilog -sv {bp_relay_unrolled.sv}

# -flatten_hierarchy none for the same reason ooc_fit_gate.tcl uses it: we want the cost of the real
# instances, not of a deduplicated one. RuntimeOptimized because this is a volume/Fmax probe, not a
# QoR-tuned build -- if it does not fit under RuntimeOptimized it will not fit by directive-tweaking.
synth_design -top bp_relay_unrolled -part $part -mode out_of_context \
  -flatten_hierarchy none -directive RuntimeOptimized -include_dirs [pwd]

create_clock -name clk -period $period [get_ports clk]
report_utilization -file util_unrolled.rpt
report_timing_summary -delay_type max -file timing_unrolled.rpt

set lut    [llength [get_cells -hier -filter {REF_NAME =~ LUT*}]]
set ff     [llength [get_cells -hier -filter {IS_SEQUENTIAL}]]
set carry  [llength [get_cells -hier -filter {REF_NAME =~ CARRY*}]]
set dsp    [llength [get_cells -hier -filter {REF_NAME =~ DSP*}]]
set wns    [get_property SLACK [lindex [get_timing_paths -max_paths 1 -nworst 1 -setup] 0]]
set fmax   [expr {1000.0/($period - $wns)}]
# authoritative CLB LUT count from the util report
set fh [open util_unrolled.rpt r]; set t [read $fh]; close $fh
set clut -1
regexp {(?:CLB|Slice) LUTs\s*\|\s*([0-9]+)} $t -> clut

puts [format "RESULT CLBLUT=%s cellLUT=%d FF=%d CARRY8=%d DSP=%d period=%.2f WNS=%.3f Fmax=%.1fMHz" \
  $clut $lut $ff $carry $dsp $period $wns $fmax]
