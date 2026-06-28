# Q6-05 — non-project, out-of-context Vivado synth + impl for the surface UF decoder.
#
# Targets ANY part via -tclargs, so the same flow runs for both boards (Zybo XC7Z020, KV260 XCK26).
# Out-of-context (no I/O buffers / pin placement) because this is a *fit + Fmax* study of the PL
# block before board bring-up — in the real design `clk` comes from the PS and the wide ports go to
# AXI (Q6-07), not chip pins. Emits utilization + timing reports and a one-line Fmax result.
#
# Usage (run from hw/syn/, Vivado on PATH):
#   vivado -mode batch -source synth.tcl -tclargs <part> <xdc> <outdir>
#   e.g. vivado -mode batch -source synth.tcl -tclargs xc7z020clg400-1 zybo_z7_20.xdc reports/zybo

if {$argc < 3} {
  puts "ERROR: usage: synth.tcl <part> <xdc> <outdir>"
  exit 2
}
set part   [lindex $argv 0]
set xdc    [lindex $argv 1]
set outdir [lindex $argv 2]
set hw     [file normalize [file join [file dirname [info script]] ..]]
file mkdir $outdir

# RTL + the generated graph header (resolved via the include path on synth_design).
read_verilog -sv [file join $hw uf_surface_decoder.sv]
read_xdc $xdc

synth_design -top uf_surface_decoder -part $part -mode out_of_context -include_dirs $hw
write_checkpoint -force [file join $outdir post_synth.dcp]
report_utilization -file [file join $outdir util_synth.rpt]

opt_design
place_design
route_design
write_checkpoint -force [file join $outdir post_route.dcp]
report_utilization        -file [file join $outdir util_impl.rpt]
report_timing_summary     -file [file join $outdir timing_impl.rpt]
report_design_analysis -timing -file [file join $outdir timing_analysis.rpt]

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
