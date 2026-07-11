# Q7-02 M7 — Step-0 OOC "fit gate" (Task 3, Step 2): SYNTH-ONLY out-of-context run of the full modular
# unroll skeleton (`bp_unroll_skeleton.sv`) to answer one question cheaply before committing to the
# hierarchically-modular M7 design: does instantiating ALL 144 `check_minsum` + ALL 864 `var_update`
# submodules (Tasks 1-2), wired at circuit-graph scale, fit the KV260 (xck26-sfvc784-2LV-c) at all?
#
# Modeled on hw/syn/synth.tcl's OOC flow (Q6-05: `read_verilog -sv`, `synth_design ... -mode
# out_of_context`, `report_utilization`) but narrower on purpose:
#   - SYNTH ONLY, no `opt_design`/`place_design`/`route_design`/timing — a Step-0 fit gate only cares
#     about instantiated-logic VOLUME (LUT/FF/DSP/BRAM), not Fmax. Full place & route is a later task
#     (Task 4+), gated on THIS gate being GO.
#   - `-flatten_hierarchy none` keeps the 144+864 submodule instances visible in the netlist rather than
#     letting Vivado's synth-time flattening merge/dedupe them — we want the reported utilization to
#     reflect what 1008 real instances cost, not what a single deduplicated instance would.
#   - part/top are fixed (not `-tclargs` parameterised like synth.tcl/synth_bp.tcl) since this gate only
#     ever targets one device for one module; the M6 dual-target pattern doesn't apply here.
#
# Usage (Vivado on PATH; run from a directory holding the 3 RTL files + bb_gross_tanner.svh, e.g. the
# staged hw/ directory on the remote synth box — see Task 3 Step 4):
#   vivado -mode batch -source ooc_fit_gate.tcl
# Then grep `^RESULT` out of the log (the controller polls this after staging + launching per Step 4;
# this script is NOT run as part of Task 3 Steps 1-3 — Verilator elaboration is the local gate here).

set part xck26-sfvc784-2LV-c
set top  bp_unroll_skeleton

read_verilog -sv {check_minsum.sv var_update.sv bp_unroll_skeleton.sv}

synth_design -top $top -part $part -mode out_of_context -flatten_hierarchy none -include_dirs [pwd]

report_utilization -file util_skeleton.rpt

# ---------------------------------------------------------------------- structural cross-check
# Raw post-synth cell counts by mapped primitive family. `IS_SEQUENTIAL` catches every flop/latch
# regardless of family name (FDRE/FDSE/FDCE/... on UltraScale+); the REF_NAME globs mirror M6's ooc.tcl
# pattern for LUT/DSP/RAMB primitive counting. These are a sanity cross-check against the utilization
# report below, not the primary source — `report_utilization`'s "Used" column is the authoritative,
# Vivado-reconciled number (it accounts for LUT-as-SRL/carry/etc. that a naive REF_NAME glob can miss).
set cc_ff   [llength [get_cells -hier -filter {IS_SEQUENTIAL}]]
set cc_lut  [llength [get_cells -hier -filter {REF_NAME =~ LUT*}]]
set cc_dsp  [llength [get_cells -hier -filter {REF_NAME =~ DSP*}]]
set cc_ramb [llength [get_cells -hier -filter {REF_NAME =~ RAMB*}]]
puts [format "INFO cell-count cross-check: LUT=%d FF=%d DSP=%d RAMB=%d" $cc_lut $cc_ff $cc_dsp $cc_ramb]

# ---------------------------------------------------------------------- parse report_utilization
# UltraScale+ (this part) labels rows "CLB LUTs"/"CLB Registers"; older 7-series parts (M6) used
# "Slice LUTs"/"Slice Registers" — match either so this script survives a part change. Block RAM is
# reported in 36Kb-tile-equivalent units ("Block RAM Tile"); DSPs are just "DSPs".
proc parse_util_row {text pattern} {
  if {[regexp $pattern $text -> val]} {
    return $val
  }
  return -1
}

set fh   [open util_skeleton.rpt r]
set text [read $fh]
close $fh

set rpt_lut  [parse_util_row $text {(?:CLB|Slice) LUTs\s*\|\s*([0-9]+)}]
set rpt_ff   [parse_util_row $text {(?:CLB|Slice) Registers\s*\|\s*([0-9]+)}]
set rpt_dsp  [parse_util_row $text {\|\s*DSPs\s*\|\s*([0-9]+)}]
set rpt_ramb [parse_util_row $text {Block RAM Tile\s*\|\s*([0-9]+(?:\.[0-9]+)?)}]

# Prefer the reconciled report numbers; fall back to the structural cross-check only if a row failed
# to parse (e.g. a report-format change) so the gate still emits a usable RESULT line.
set final_lut  [expr {$rpt_lut  >= 0 ? $rpt_lut  : $cc_lut}]
set final_ff   [expr {$rpt_ff   >= 0 ? $rpt_ff   : $cc_ff}]
set final_dsp  [expr {$rpt_dsp  >= 0 ? $rpt_dsp  : $cc_dsp}]
set final_ramb [expr {$rpt_ramb >= 0 ? $rpt_ramb : $cc_ramb}]

puts [format "RESULT LUT=%s FF=%s DSP=%s RAMB=%s" $final_lut $final_ff $final_dsp $final_ramb]
