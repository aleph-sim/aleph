# P2 appliance v2-large — area / clock / CDC reports from the routed F2 checkpoint (the HDK flow writes only
# timing reports). Run ON the AWS build instance: the free Vivado elsewhere cannot open VU47P checkpoints.
#   vivado -mode batch -source aleph_reports.tcl -tclargs <post_route.dcp> <out_dir>
# Each report is attempted on its own, so one the encrypted shell refuses does not lose the others.
lassign $argv dcp out
file mkdir $out
open_checkpoint $dcp

set cl [get_cells -hierarchical -quiet -filter {ORIG_REF_NAME == cl_aleph_bp || REF_NAME == cl_aleph_bp}]
puts "aleph: CL cell = $cl"
set cmds [list \
  [list report_utilization -file $out/util_device.rpt] \
  [list report_utilization -cells $cl -file $out/util_cl.rpt] \
  [list report_utilization -cells $cl -hierarchical -hierarchical_depth 3 -file $out/util_cl_hier.rpt] \
  [list report_clock_utilization -file $out/clock_util.rpt] \
  [list report_cdc -details -file $out/cdc.rpt] \
  [list report_timing_summary -max_paths 10 -file $out/timing_summary.rpt] \
]
foreach c $cmds {
  if {[catch {{*}$c} err]} { puts "aleph: [lindex $c 0] FAILED: $err" } else { puts "aleph: [lindex $c 0] ok" }
}
