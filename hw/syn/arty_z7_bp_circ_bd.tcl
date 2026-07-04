# Q7-02 M5-followup — Arty Z7-20 block design + bitstream for the CIRCUIT-LEVEL M2 relay-BP decoder.
#
# Same structure as arty_z7_bp_bd.tcl, but builds the graph-generic M2-BRAM decoder
# (bp_relay_bram — block-RAM message tables + edge-serial updates) behind the WIDE AXI4-Lite wrapper
# (bp_axi_wrap_wide), against the depth-7 circuit-level gross-code graph. The BRAM core is the one that
# actually FITS the xc7z020: the flop-array M2 (bp_relay_decoder) OOMs Vivado at BP_E=2952 (PR #447).
# The circuit header must be staged as `bb_gross_tanner.svh` in $hw (the wide wrapper + core both
# `include that name). First circuit-level qLDPC decode on the Arty.
#
# Usage (from a dir whose parent `hw` holds the circuit-level bb_gross_tanner.svh):
#   vivado -mode batch -source syn/arty_z7_bp_circ_bd.tcl -tclargs <proj_dir> <out_dir> [fclk_mhz]

set part   xc7z020clg400-1
set bdname bp_circ_bd
set outname bp_arty_circ

set proj_dir [expr {$argc >= 1 ? [lindex $argv 0] : "artybpcirc"}]
set out_dir  [expr {$argc >= 2 ? [lindex $argv 1] : "out"}]
set fclk_mhz [expr {$argc >= 3 ? [lindex $argv 2] : 50}]
set hw [file normalize [file join [file dirname [info script]] ..]]
file mkdir $out_dir

create_project -force bp_arty_circ $proj_dir -part $part

add_files -norecurse [list \
  [file join $hw bp_relay_bram.sv] \
  [file join $hw bp_axi_wrap_wide.sv] \
  [file join $hw bp_axi_top_wide.v] \
  [file join $hw bb_gross_tanner.svh]]
set_property include_dirs $hw [current_fileset]
set_property file_type SystemVerilog [get_files *.sv]
set_property file_type {Verilog Header} [get_files bb_gross_tanner.svh]
update_compile_order -fileset sources_1

# ---- Block design ----
create_bd_design $bdname
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
  -config {make_external "FIXED_IO, DDR" apply_board_preset "0" Master "Disable" Slave "Disable"} \
  [get_bd_cells ps7]
set_property -dict [list \
  CONFIG.PCW_USE_M_AXI_GP0 {1} \
  CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ [format %d $fclk_mhz]] [get_bd_cells ps7]

create_bd_cell -type module -reference bp_axi_top_wide bp_0
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
  -config [list Master "/ps7/M_AXI_GP0" Clk "Auto"] \
  [get_bd_intf_pins bp_0/s_axil]

assign_bd_address
regenerate_bd_layout
validate_bd_design
save_bd_design

make_wrapper -files [get_files ${bdname}.bd] -top
add_files -norecurse [file join $proj_dir bp_arty_circ.gen sources_1 bd $bdname hdl ${bdname}_wrapper.v]
set_property top ${bdname}_wrapper [current_fileset]
update_compile_order -fileset sources_1

launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
  error "impl_1 did not finish: [get_property STATUS [get_runs impl_1]]"
}

set bit [glob -nocomplain [file join $proj_dir bp_arty_circ.runs impl_1 ${bdname}_wrapper.bit]]
set hwh [glob -nocomplain [file join $proj_dir bp_arty_circ.gen sources_1 bd $bdname hw_handoff ${bdname}.hwh]]
if {$bit eq ""} { error "no bitstream produced" }
if {$hwh eq ""} { error "no .hwh produced (needed by PYNQ)" }
file copy -force $bit [file join $out_dir ${outname}.bit]
file copy -force $hwh [file join $out_dir ${outname}.hwh]

open_run impl_1
set wns [get_property SLACK [get_timing_paths -setup -max_paths 1 -nworst 1]]
puts [format "RESULT bitstream=%s hwh=%s FCLK=%dMHz WNS=%.3fns %s" \
  [file join $out_dir ${outname}.bit] [file join $out_dir ${outname}.hwh] \
  $fclk_mhz $wns [expr {$wns >= 0 ? "TIMING_MET" : "TIMING_VIOLATED"}]]
