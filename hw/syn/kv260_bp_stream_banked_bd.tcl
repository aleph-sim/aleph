# Q7-06 (AC-1) — KV260 (Zynq UltraScale+ XCK26) block design + bitstream for the BATCHED banked
# relay-BP decoder over AXI-DMA:
#
#   PS DDR --(AXI DMA MM2S)--> bp_stream_banked.s_axis --> banked decode --> bp_stream_banked.m_axis
#          --(AXI DMA S2MM)--> DDR
#
# A whole batch of independent syndrome->result experiments streams through one DMA transfer: NS=5
# syndrome beats per experiment in, one status word {obs,vflag,latency} per experiment out. This is the
# Q7-06 AC-1 throughput path -- it removes the per-experiment Python+MMIO round-trips of the M8 AXI-Lite
# overlay (bp_axi_wrap_banked / kv260_bp_circ_banked_bd.tcl), which is what caps its harness throughput.
#
# Structural fusion of kv260_bp_circ_banked_bd.tcl (the UltraScale+ PS + banked core + circuit header)
# and arty_z7_dma_win_bd.tcl (the AXI-DMA + AXIS streaming plumbing), retargeted to the MPSoC PS pins.
# The header staged as bb_gross_tanner.svh MUST be generated at the chosen (W,V) (default 16 48, shipped).
# NOTE: FLATTEN_HIERARCHY none on synth_1 is load-bearing (M7 post-mortem: flat area-opt stall).
#
# Usage (Vivado on PATH, from a dir whose parent `hw` holds the circuit-level bb_gross_tanner.svh):
#   vivado -mode batch -source syn/kv260_bp_stream_banked_bd.tcl -tclargs <proj_dir> <out_dir> [fclk_mhz] [bdonly]
# `bdonly` as the 4th arg assembles + validates the block design then stops (fast pre-flight, no impl).

set part    xck26-sfvc784-2LV-c
set bdname  bp_stream_bank_bd
set outname bp_kv260_stream_banked

set proj_dir [expr {$argc >= 1 ? [lindex $argv 0] : "kv260bpstream"}]
set out_dir  [expr {$argc >= 2 ? [lindex $argv 1] : "out_stream"}]
set fclk_mhz [expr {$argc >= 3 ? [lindex $argv 2] : 100}]
set bdonly   [expr {$argc >= 4 && [lindex $argv 3] eq "bdonly"}]
# optional 5th arg: impl_1 strategy (e.g. Performance_Explore); empty = defaults
set strategy [expr {$argc >= 5 ? [lindex $argv 4] : ""}]
set hw [file normalize [file join [file dirname [info script]] ..]]
file mkdir $out_dir

create_project -force bp_kv260_stream_banked $proj_dir -part $part

add_files -norecurse [list \
  [file join $hw check_minsum.sv] \
  [file join $hw var_update.sv] \
  [file join $hw bp_relay_banked.sv] \
  [file join $hw bp_stream_banked_core.sv] \
  [file join $hw bp_stream_banked.v] \
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
# One AXI-Lite master (M_AXI_HPM0_FPD, DMA control), one HP slave (S_AXI_HP0_FPD, DMA <-> DDR), one PL
# clock at the requested FCLK. PS DDR/MIO is irrelevant for a PL overlay on a booted Linux (PYNQ programs
# only the PL); it exists to generate the AXI address map + the .hwh PYNQ needs.
# On zynq_ultra_ps_e the HP slave ports are named S_AXI_GP* (SAXIGP2 == the S_AXI_HP0_FPD interface),
# NOT S_AXI_HP0 — there is no PSU__USE__S_AXI_HP0 flag in Vivado 2024.2. Enabling PSU__USE__S_AXI_GP2
# creates the S_AXI_HP0_FPD port (+ saxihp0_fpd_aclk); PSU__USE__M_AXI_GP0 creates M_AXI_HPM0_FPD
# (+ maxihpm0_fpd_aclk). 64-bit HP is ample for one 32-bit DMA stream pair; sc_hp adapts the width.
set_property -dict [list \
  CONFIG.PSU__USE__M_AXI_GP0 {1} \
  CONFIG.PSU__USE__M_AXI_GP2 {0} \
  CONFIG.PSU__USE__S_AXI_GP2 {1} \
  CONFIG.PSU__SAXIGP2__DATA_WIDTH {64} \
  CONFIG.PSU__FPGA_PL0_ENABLE {1} \
  CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ [format %d $fclk_mhz]] [get_bd_cells ps]

# ---- AXI DMA: simple (register) mode, MM2S + S2MM, 32-bit streams, 26-bit length (64 MB/transfer) ----
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

# ---- batched decoder datapath ----
create_bd_cell -type module -reference bp_stream_banked bp_0
# early_exit tied off to 0 (full schedule) for the AC-1/AC-2 build; rebuild with 1 for early-exit mode.
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant ee0
set_property -dict [list CONFIG.CONST_WIDTH {1} CONFIG.CONST_VAL {0}] [get_bd_cells ee0]

# ---- reset + smartconnects ----
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset rst0
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect sc_ctrl
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells sc_ctrl]
create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect sc_hp
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] [get_bd_cells sc_hp]

# ---- clock + reset nets (UltraScale+ MPSoC pins) ----
set clk [get_bd_pins ps/pl_clk0]
connect_bd_net $clk [get_bd_pins rst0/slowest_sync_clk]
connect_bd_net [get_bd_pins ps/pl_resetn0] [get_bd_pins rst0/ext_reset_in]
set arstn [get_bd_pins rst0/peripheral_aresetn]

connect_bd_net $clk \
  [get_bd_pins ps/maxihpm0_fpd_aclk] [get_bd_pins ps/saxihp0_fpd_aclk] \
  [get_bd_pins axi_dma_0/s_axi_lite_aclk] [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] \
  [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] \
  [get_bd_pins bp_0/aclk] \
  [get_bd_pins sc_ctrl/aclk] [get_bd_pins sc_hp/aclk]

connect_bd_net $arstn \
  [get_bd_pins axi_dma_0/axi_resetn] [get_bd_pins bp_0/aresetn] \
  [get_bd_pins sc_ctrl/aresetn] [get_bd_pins sc_hp/aresetn]

connect_bd_net [get_bd_pins ee0/dout] [get_bd_pins bp_0/early_exit]

# ---- AXI: control plane (HPM0_FPD -> DMA S_AXI_LITE) ----
connect_bd_intf_net [get_bd_intf_pins ps/M_AXI_HPM0_FPD] [get_bd_intf_pins sc_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_ctrl/M00_AXI]   [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

# ---- AXI: data plane (DMA MM2S + S2MM masters -> HP0) ----
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] [get_bd_intf_pins sc_hp/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] [get_bd_intf_pins sc_hp/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins sc_hp/M00_AXI]        [get_bd_intf_pins ps/S_AXI_HP0_FPD]

# ---- AXI4-Stream: DMA MM2S -> batched decoder -> DMA S2MM ----
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] [get_bd_intf_pins bp_0/s_axis]
connect_bd_intf_net [get_bd_intf_pins bp_0/m_axis]           [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

assign_bd_address
regenerate_bd_layout
validate_bd_design
save_bd_design

if {$bdonly} { puts "BD_OK vlnv=$ps_vlnv"; return }

make_wrapper -files [get_files ${bdname}.bd] -top
add_files -norecurse [file join $proj_dir bp_kv260_stream_banked.gen sources_1 bd $bdname hdl ${bdname}_wrapper.v]
set_property top ${bdname}_wrapper [current_fileset]
update_compile_order -fileset sources_1

set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY none [get_runs synth_1]
if {$strategy ne ""} { set_property strategy $strategy [get_runs impl_1] }
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
  error "impl_1 did not finish: [get_property STATUS [get_runs impl_1]]"
}

set bit [glob -nocomplain [file join $proj_dir bp_kv260_stream_banked.runs impl_1 ${bdname}_wrapper.bit]]
set hwh [glob -nocomplain [file join $proj_dir bp_kv260_stream_banked.gen sources_1 bd $bdname hw_handoff ${bdname}.hwh]]
if {$bit eq ""} { error "no bitstream produced" }
if {$hwh eq ""} { error "no .hwh produced (needed by PYNQ)" }
file copy -force $bit [file join $out_dir ${outname}.bit]
file copy -force $hwh [file join $out_dir ${outname}.hwh]

open_run impl_1
set wns [get_property SLACK [get_timing_paths -setup -max_paths 1 -nworst 1]]
puts [format "RESULT bitstream=%s hwh=%s FCLK=%dMHz WNS=%.3fns %s" \
  [file join $out_dir ${outname}.bit] [file join $out_dir ${outname}.hwh] \
  $fclk_mhz $wns [expr {$wns >= 0 ? "TIMING_MET" : "TIMING_VIOLATED"}]]
