# Q7-02 M7 — KV260 (Zynq UltraScale+ XCK26) block design + bitstream for the K-BANKED CIRCUIT-LEVEL
# relay-BP decoder: bp_relay_banked (beta-split check-major LUTRAM banks, W/V per cycle from the header)
# behind the WIDE AXI4-Lite wrapper (bp_axi_wrap_banked / bp_axi_top_banked, IDCODE 0x4250_0003),
# against the depth-7 circuit-level gross-code graph (BP_N=864 / BP_C=144 / BP_E=2952).
# The header staged as bb_gross_tanner.svh MUST be generated at the chosen (W,V) (M7 pick: 12 36).
# NOTE: FLATTEN_HIERARCHY none on synth_1 is load-bearing (M7 post-mortem: flat area-opt stall).
#
# Mirrors arty_z7_bp_circ_bd.tcl but on the UltraScale+ PS. Chosen after an OOC sweep (M6): the
# fully-unrolled / partial-unroll cores are pathological for Vivado synthesis at circuit scale
# (deg-25 min-sum comparator trees OOM/stall), while bram_dp fits KV260 at ~7% LUT, Fmax ~121 MHz.
# The KV260's larger fabric therefore buys clock (~2x over the Arty's 50 MHz), not unrolling.
#
# The circuit header must be staged as bb_gross_tanner.svh in $hw (wrapper + core both `include it).
#
# Usage (from a dir whose parent `hw` holds the circuit-level bb_gross_tanner.svh):
#   vivado -mode batch -source syn/kv260_bp_circ_bd.tcl -tclargs <proj_dir> <out_dir> [fclk_mhz] [bdonly]
# `bdonly` as the 4th arg assembles + validates the block design then stops (fast pre-flight).

set part    xck26-sfvc784-2LV-c
set bdname  bp_bank_bd
set outname bp_kv260_banked

set proj_dir [expr {$argc >= 1 ? [lindex $argv 0] : "kv260bpcirc"}]
set out_dir  [expr {$argc >= 2 ? [lindex $argv 1] : "out"}]
set fclk_mhz [expr {$argc >= 3 ? [lindex $argv 2] : 100}]
set bdonly   [expr {$argc >= 4 && [lindex $argv 3] eq "bdonly"}]
set hw [file normalize [file join [file dirname [info script]] ..]]
file mkdir $out_dir

create_project -force bp_kv260_banked $proj_dir -part $part

add_files -norecurse [list \
  [file join $hw check_minsum.sv] \
  [file join $hw var_update.sv] \
  [file join $hw bp_relay_banked.sv] \
  [file join $hw bp_axi_wrap_banked.sv] \
  [file join $hw bp_axi_top_banked.v] \
  [file join $hw bb_gross_tanner.svh]]
set_property include_dirs $hw [current_fileset]
set_property file_type SystemVerilog [get_files *.sv]
set_property file_type {Verilog Header} [get_files bb_gross_tanner.svh]
update_compile_order -fileset sources_1

# ---- Block design ----
create_bd_design $bdname

set ps_vlnv [lindex [lsort [get_ipdefs -all *:ip:zynq_ultra_ps_e:*]] end]
create_bd_cell -type ip -vlnv $ps_vlnv ps
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e \
  -config {make_external "FIXED_IO, DDR" apply_board_preset "0" Master "Disable" Slave "Disable"} \
  [get_bd_cells ps]
# Enable one AXI-Lite master (M_AXI_HPM0_FPD) + one PL clock at the requested FCLK. The PS DDR/MIO
# config is irrelevant for a PL-partial overlay loaded on an already-booted Linux (PYNQ reprograms
# only the PL); it exists here just to generate the AXI address map + the .hwh PYNQ needs.
set_property -dict [list \
  CONFIG.PSU__USE__M_AXI_GP0 {1} \
  CONFIG.PSU__USE__M_AXI_GP2 {0} \
  CONFIG.PSU__FPGA_PL0_ENABLE {1} \
  CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ [format %d $fclk_mhz]] [get_bd_cells ps]

create_bd_cell -type module -reference bp_axi_top_banked bp_0
apply_bd_automation -rule xilinx.com:bd_rule:axi4 \
  -config [list Master "/ps/M_AXI_HPM0_FPD" Clk "Auto"] \
  [get_bd_intf_pins bp_0/s_axil]

assign_bd_address
regenerate_bd_layout
validate_bd_design
save_bd_design

if {$bdonly} { puts "BD_OK vlnv=$ps_vlnv"; return }

make_wrapper -files [get_files ${bdname}.bd] -top
add_files -norecurse [file join $proj_dir bp_kv260_banked.gen sources_1 bd $bdname hdl ${bdname}_wrapper.v]
set_property top ${bdname}_wrapper [current_fileset]
update_compile_order -fileset sources_1

set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY none [get_runs synth_1]
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
  error "impl_1 did not finish: [get_property STATUS [get_runs impl_1]]"
}

set bit [glob -nocomplain [file join $proj_dir bp_kv260_banked.runs impl_1 ${bdname}_wrapper.bit]]
set hwh [glob -nocomplain [file join $proj_dir bp_kv260_banked.gen sources_1 bd $bdname hw_handoff ${bdname}.hwh]]
if {$bit eq ""} { error "no bitstream produced" }
if {$hwh eq ""} { error "no .hwh produced (needed by PYNQ)" }
file copy -force $bit [file join $out_dir ${outname}.bit]
file copy -force $hwh [file join $out_dir ${outname}.hwh]

open_run impl_1
set wns [get_property SLACK [get_timing_paths -setup -max_paths 1 -nworst 1]]
puts [format "RESULT bitstream=%s hwh=%s FCLK=%dMHz WNS=%.3fns %s" \
  [file join $out_dir ${outname}.bit] [file join $out_dir ${outname}.hwh] \
  $fclk_mhz $wns [expr {$wns >= 0 ? "TIMING_MET" : "TIMING_VIOLATED"}]]
