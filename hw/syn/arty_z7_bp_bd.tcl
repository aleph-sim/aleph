# Q7-02 board build — Arty Z7-20 block design + bitstream for the partial relay-BP decoder.
#
# Builds a real board design (unlike the OOC fit/Fmax study in synth_bp.tcl): Zynq-7 PS + our
# `bp_axi_top` (bp_axi_wrap exposing the AXI4-Lite control plane) on the PS GP0 AXI master, then
# synth -> impl -> bitstream, and copies out `<name>.bit` + `<name>.hwh` for PYNQ overlay loading.
# Structurally identical to arty_z7_bd.tcl (the UF board build); only the RTL sources and the module
# reference change.
#
# The gross-BB partial decoder's OOC Fmax (12/24 unroll) was 35.5 MHz, so the default PL clock is 25 MHz
# to give in-context timing margin on the first build (override with the 3rd arg once WNS is known).
# No Digilent board files required: generic PS7 (M_AXI_GP0 + one FCLK); DDR/MIO set by the PYNQ-Z1 FSBL.
#
# Usage (Vivado on PATH, run from hw/):
#   vivado -mode batch -source syn/arty_z7_bp_bd.tcl -tclargs <proj_dir> <out_dir> [fclk_mhz]
#   e.g. vivado -mode batch -source syn/arty_z7_bp_bd.tcl -tclargs /root/q7synth/artybp /root/q7synth/out

set part   xc7z020clg400-1
set bdname bp_bd
set outname bp_arty

set proj_dir [expr {$argc >= 1 ? [lindex $argv 0] : "artybp"}]
set out_dir  [expr {$argc >= 2 ? [lindex $argv 1] : "out"}]
set fclk_mhz [expr {$argc >= 3 ? [lindex $argv 2] : 25}]
set hw [file normalize [file join [file dirname [info script]] ..]]
file mkdir $out_dir

create_project -force bp_arty $proj_dir -part $part

# RTL: decoder core + AXI wrapper + board top. The generated graph header is found via include dir.
add_files -norecurse [list \
  [file join $hw bp_relay_partial_fast.sv] \
  [file join $hw bp_axi_wrap.sv] \
  [file join $hw bp_axi_top.v] \
  [file join $hw bb_gross_tanner.svh]]
set_property include_dirs $hw [current_fileset]
set_property file_type SystemVerilog [get_files *.sv]
# The generated graph header must be a project file (for the BD module reference to elaborate) and
# resolvable via include_dirs, so the explicit `include "bb_gross_tanner.svh"` in the RTL picks it up.
# The header carries its own `ifndef/`define guard, so even if Vivado global-includes it the localparams
# are declared once.
set_property file_type {Verilog Header} [get_files bb_gross_tanner.svh]
update_compile_order -fileset sources_1

# ---- Block design ----
create_bd_design $bdname

# Zynq-7 PS, generic config (no board preset): enable GP0 master + one FCLK at $fclk_mhz.
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
  -config {make_external "FIXED_IO, DDR" apply_board_preset "0" Master "Disable" Slave "Disable"} \
  [get_bd_cells ps7]
set_property -dict [list \
  CONFIG.PCW_USE_M_AXI_GP0 {1} \
  CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ [format %d $fclk_mhz]] [get_bd_cells ps7]

# Our decoder as a module reference; interfaces inferred from the AXI-standard port names.
create_bd_cell -type module -reference bp_axi_top bp_0

# Connect PS GP0 master -> bp_0 AXI4-Lite, auto clock + reset (inserts interconnect + proc reset).
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
  -config [list Master "/ps7/M_AXI_GP0" Clk "Auto"] \
  [get_bd_intf_pins bp_0/s_axil]

assign_bd_address
regenerate_bd_layout
validate_bd_design
save_bd_design

# ---- Wrapper + implementation ----
make_wrapper -files [get_files ${bdname}.bd] -top
add_files -norecurse [file join $proj_dir bp_arty.gen sources_1 bd $bdname hdl ${bdname}_wrapper.v]
set_property top ${bdname}_wrapper [current_fileset]
update_compile_order -fileset sources_1

launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1

if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
  error "impl_1 did not finish: [get_property STATUS [get_runs impl_1]]"
}

# ---- Collect artifacts for PYNQ: <outname>.bit + <outname>.hwh (matching basenames) ----
set bit [glob -nocomplain [file join $proj_dir bp_arty.runs impl_1 ${bdname}_wrapper.bit]]
set hwh [glob -nocomplain [file join $proj_dir bp_arty.gen sources_1 bd $bdname hw_handoff ${bdname}.hwh]]
if {$bit eq ""} { error "no bitstream produced" }
if {$hwh eq ""} { error "no .hwh produced (needed by PYNQ)" }
file copy -force $bit [file join $out_dir ${outname}.bit]
file copy -force $hwh [file join $out_dir ${outname}.hwh]

# Post-route timing sanity (WNS must be >= 0 at $fclk_mhz).
open_run impl_1
set wns [get_property SLACK [get_timing_paths -setup -max_paths 1 -nworst 1]]
puts [format "RESULT bitstream=%s hwh=%s FCLK=%dMHz WNS=%.3fns %s" \
  [file join $out_dir ${outname}.bit] [file join $out_dir ${outname}.hwh] \
  $fclk_mhz $wns [expr {$wns >= 0 ? "TIMING_MET" : "TIMING_VIOLATED"}]]
