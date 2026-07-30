# Q7-02 Task B2 Step 1 — full implementation (synth -> opt -> place -> phys_opt -> route) of the
# banked relay-BP core on the AWS F2 part, for POST-ROUTE utilisation and Fmax.
# Step 0 already has the out-of-context synthesis numbers; a second synth-only figure would not
# justify renting anything, so this script must route.
# Usage: vivado -mode batch -source ../impl_vu47p.tcl -tclargs [period_ns] [label]
# Run from a dir holding check_minsum.sv var_update.sv bp_relay_banked.sv bb_gross_tanner.svh.
set period [expr {$argc >= 1 ? double([lindex $argv 0]) : 5.0}]
set label  [expr {$argc >= 2 ? [lindex $argv 1] : "wv"}]
set part   xcvu47p-fsvh2892-2-e

proc stamp {msg} { puts "STAGE $msg [clock format [clock seconds] -format %H:%M:%S]" ; flush stdout }

read_verilog -sv [lsort [glob *.sv]]

stamp synth_begin
synth_design -top bp_relay_banked -part $part -mode out_of_context \
  -flatten_hierarchy none -include_dirs [pwd]
create_clock -name clk -period $period [get_ports clk]
report_utilization -file util_synth.rpt
stamp synth_done

stamp opt_begin
opt_design
stamp place_begin
place_design
stamp physopt_begin
phys_opt_design
stamp route_begin
route_design
stamp route_done

report_utilization -file util_route.rpt
report_utilization -hierarchical -hierarchical_depth 2 -file util_route_hier.rpt
report_timing_summary -delay_type max -file timing_route.rpt
report_design_analysis -congestion -file congestion.rpt
write_checkpoint -force post_route.dcp

# Post-route WNS is the number this whole rental exists to produce.
set wns  [get_property SLACK [lindex [get_timing_paths -max_paths 1 -nworst 1 -setup] 0]]
set fmax [expr {1000.0/($period - $wns)}]

# Authoritative counts from the util report. The row label carries a trailing '*' footnote marker
# ("CLB LUTs*") and a pattern anchored on '\s*|' silently fails on it -- that bug produced -1 for the
# whole Step 0 sweep. Match up to the column separator instead.
set fh [open util_route.rpt r]; set t [read $fh]; close $fh
set clut -1; set creg -1; set carry -1; set dsp -1; set bram -1
regexp {(?:CLB|Slice) LUTs[^|]*\|\s*([0-9]+)}   $t -> clut
regexp {CLB Registers[^|]*\|\s*([0-9]+)}        $t -> creg
regexp {CARRY8[^|]*\|\s*([0-9]+)}               $t -> carry
regexp {DSPs?[^|]*\|\s*([0-9]+)}                $t -> dsp
regexp {Block RAM Tile[^|]*\|\s*([0-9.]+)}      $t -> bram

# SLR crossings: VU47P is a multi-die part and an 800k-LUT monolithic design gets split across super
# logic regions. If Fmax collapses against the Step 0 estimate this is the first suspect, so count it.
set slrx "NA"
if {![catch {llength [get_nets -hier -filter {ROUTE_STATUS != INTRASITE}]}]} {
  catch { set slrx [llength [get_slrs]] }
}

puts [format "RESULT %s part=%s CLBLUT=%s CLBREG=%s CARRY8=%s DSP=%s BRAM=%s SLRs=%s period=%.2f WNS=%.3f Fmax=%.1fMHz" \
  $label $part $clut $creg $carry $dsp $bram $slrx $period $wns $fmax]
stamp all_done
