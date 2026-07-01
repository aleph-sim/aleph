# Q6-03 (throughput) — Arty Z7-20 block design with AXI DMA for decoder-bound throughput.
#
# PS DDR --(AXI DMA MM2S)--> uf_stream.s_axis --> decode --> uf_stream.m_axis --(AXI DMA S2MM)--> DDR.
# The PS only sets up the transfer, so measured throughput is the decoder's own rate (one decode per
# its multi-cycle latency), not the AXI4-Lite-poll rate of the Q6-08 bring-up build.
#
# Explicit clock/reset/interconnect wiring (no apply_bd_automation for the AXI links) so the headless
# build is deterministic. Generic PS7 (no Digilent board files); FCLK 50 MHz.
#
# Usage (Vivado on PATH, run from hw/):
#   vivado -mode batch -source syn/arty_z7_dma_bd.tcl -tclargs <proj_dir> <out_dir> [fclk_mhz]

set part    xc7z020clg400-1
set bdname  uf_dma
set outname uf_arty_dma

set proj_dir [expr {$argc >= 1 ? [lindex $argv 0] : "artybd_dma"}]
set out_dir  [expr {$argc >= 2 ? [lindex $argv 1] : "out_dma"}]
set fclk_mhz [expr {$argc >= 3 ? [lindex $argv 2] : 50}]
# 4th arg: number of parallel decoder engines. K<=1 -> single uf_stream; K>=2 -> K-way uf_stream_array.
set nk       [expr {$argc >= 4 ? [lindex $argv 3] : 1}]
set hw [file normalize [file join [file dirname [info script]] ..]]
file mkdir $out_dir

create_project -force uf_arty_dma $proj_dir -part $part

add_files -norecurse [list \
  [file join $hw uf_surface_decoder.sv] \
  [file join $hw uf_stream_core.sv] \
  [file join $hw uf_stream.v] \
  [file join $hw uf_stream_array_core.sv] \
  [file join $hw uf_stream_array.v] \
  [file join $hw uf_surface_graph.svh]]
set_property include_dirs $hw [current_fileset]
set_property file_type SystemVerilog        [get_files uf_surface_decoder.sv]
set_property file_type SystemVerilog        [get_files uf_stream_core.sv]
set_property file_type SystemVerilog        [get_files uf_stream_array_core.sv]
set_property file_type {Verilog Header}      [get_files uf_surface_graph.svh]
update_compile_order -fileset sources_1

create_bd_design $bdname

# ---- Zynq-7 PS: GP0 master (DMA control) + HP0 slave (DMA <-> DDR) + one 50 MHz FCLK ----
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
  -config {make_external "FIXED_IO, DDR" apply_board_preset "0" Master "Disable" Slave "Disable"} \
  [get_bd_cells ps7]
set_property -dict [list \
  CONFIG.PCW_USE_M_AXI_GP0 {1} \
  CONFIG.PCW_USE_S_AXI_HP0 {1} \
  CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ [format %d $fclk_mhz]] [get_bd_cells ps7]

# ---- AXI DMA: simple (register) mode, MM2S + S2MM, 32-bit streams, 26-bit length (64 MB) ----
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma axi_dma_0
set_property -dict [list \
  CONFIG.c_include_sg {0} \
  CONFIG.c_sg_include_stscntrl_strm {0} \
  CONFIG.c_include_mm2s {1} \
  CONFIG.c_include_s2mm {1} \
  CONFIG.c_m_axis_mm2s_tdata_width {32} \
  CONFIG.c_s_axis_s2mm_tdata_width {32} \
  CONFIG.c_sg_length_width {26} \
  CONFIG.c_micro_dma {0}] [get_bd_cells axi_dma_0]

# ---- decoder streaming datapath (single engine or K-way array) ----
if {$nk <= 1} {
  create_bd_cell -type module -reference uf_stream uf_0
} else {
  create_bd_cell -type module -reference uf_stream_array uf_0
  if {[catch {set_property CONFIG.K [format %d $nk] [get_bd_cells uf_0]} e]} {
    puts "WARN: could not set CONFIG.K=$nk on module ref ($e); using the module's default K"
  }
}

# ---- reset + smartconnects ----
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset rst0   ;# FCLK_RESET0_N (active-low) -> ext_reset_in

create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect sc_ctrl
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells sc_ctrl]
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect sc_hp
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] [get_bd_cells sc_hp]

# ---- clock + reset nets ----
set clk [get_bd_pins ps7/FCLK_CLK0]
connect_bd_net $clk [get_bd_pins rst0/slowest_sync_clk]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins rst0/ext_reset_in]
set arstn [get_bd_pins rst0/peripheral_aresetn]

connect_bd_net $clk \
  [get_bd_pins ps7/M_AXI_GP0_ACLK] [get_bd_pins ps7/S_AXI_HP0_ACLK] \
  [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] \
  [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] \
  [get_bd_pins uf_0/aclk] \
  [get_bd_pins sc_ctrl/aclk] [get_bd_pins sc_hp/aclk]

connect_bd_net $arstn \
  [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins uf_0/aresetn] \
  [get_bd_pins sc_ctrl/aresetn] [get_bd_pins sc_hp/aresetn]

# ---- AXI: control plane (GP0 -> DMA S_AXI_LITE) ----
connect_bd_intf_net [get_bd_intf_pins ps7/M_AXI_GP0]   [get_bd_intf_pins sc_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/M00_AXI] [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

# ---- AXI: data plane (DMA MM2S + S2MM masters -> HP0) ----
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins sc_hp/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins sc_hp/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_hp/M00_AXI]        [get_bd_intf_pins ps7/S_AXI_HP0]

# ---- AXI4-Stream: DMA MM2S -> decoder -> DMA S2MM ----
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] [get_bd_intf_pins uf_0/s_axis]
connect_bd_intf_net [get_bd_intf_pins uf_0/m_axis]           [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

assign_bd_address
regenerate_bd_layout
validate_bd_design
save_bd_design

# ---- wrapper + implementation ----
make_wrapper -files [get_files ${bdname}.bd] -top
add_files -norecurse [file join $proj_dir uf_arty_dma.gen sources_1 bd $bdname hdl ${bdname}_wrapper.v]
set_property top ${bdname}_wrapper [current_fileset]
update_compile_order -fileset sources_1

launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
  error "impl_1 did not finish: [get_property STATUS [get_runs impl_1]]"
}

set bit [glob -nocomplain [file join $proj_dir uf_arty_dma.runs impl_1 ${bdname}_wrapper.bit]]
set hwh [glob -nocomplain [file join $proj_dir uf_arty_dma.gen sources_1 bd $bdname hw_handoff ${bdname}.hwh]]
if {$bit eq ""} { error "no bitstream produced" }
if {$hwh eq ""} { error "no .hwh produced (needed by PYNQ)" }
file copy -force $bit [file join $out_dir ${outname}.bit]
file copy -force $hwh [file join $out_dir ${outname}.hwh]

open_run impl_1
set wns [get_property SLACK [get_timing_paths -setup -max_paths 1 -nworst 1]]
puts [format "RESULT bitstream=%s hwh=%s FCLK=%dMHz WNS=%.3fns %s" \
  [file join $out_dir ${outname}.bit] [file join $out_dir ${outname}.hwh] \
  $fclk_mhz $wns [expr {$wns >= 0 ? "TIMING_MET" : "TIMING_VIOLATED"}]]
