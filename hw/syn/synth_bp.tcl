# Q7-02 M3 — non-project, out-of-context Vivado synth + impl for the fixed-point relay-BP decoder.
#
# Same OOC fit + Fmax study as synth.tcl (the UF flow), retargeted to bp_relay_decoder. Out-of-context
# (no I/O buffers / pin placement): in the real design clk comes from the PS and the wide message/
# syndrome ports go to AXI, not chip pins. Emits utilization + timing reports and a one-line Fmax.
#
# Usage (Vivado on PATH):
#   vivado -mode batch -source synth_bp.tcl -tclargs <part> <xdc> <outdir> <srcdir>
#   e.g. vivado -mode batch -source synth_bp.tcl -tclargs xck26-sfvc784-2LV-c kv260.xdc reports/kv260 .

if {$argc < 4} {
  puts "ERROR: usage: synth_bp.tcl <part> <xdc> <outdir> <srcdir>"
  exit 2
}
set part   [lindex $argv 0]
set xdc    [lindex $argv 1]
set outdir [lindex $argv 2]
set src    [file normalize [lindex $argv 3]]
file mkdir $outdir

# RTL + the generated graph header (resolved via the include path on synth_design).
read_verilog -sv [file join $src bp_relay_decoder.sv]
read_xdc $xdc

synth_design -top bp_relay_decoder -part $part -mode out_of_context -include_dirs $src
write_checkpoint -force [file join $outdir post_synth.dcp]
report_utilization -file [file join $outdir util_synth.rpt]

opt_design
place_design
route_design
write_checkpoint -force [file join $outdir post_route.dcp]
report_utilization    -file [file join $outdir util_impl.rpt]
report_timing_summary -file [file join $outdir timing_impl.rpt]

# Fmax from the worst post-route setup slack: achievable period = target - WNS.
set clk    [lindex [get_clocks] 0]
set period [get_property PERIOD $clk]
set wns    [get_property SLACK [get_timing_paths -setup -max_paths 1 -nworst 1]]
set fmax   [expr {1000.0 / ($period - $wns)}]
set msg [format "RESULT part=%s target_period=%.3fns WNS=%.3fns Fmax=%.1fMHz" \
             $part $period $wns $fmax]
puts $msg
set fh [open [file join $outdir fmax.txt] w]
puts $fh $msg
close $fh
