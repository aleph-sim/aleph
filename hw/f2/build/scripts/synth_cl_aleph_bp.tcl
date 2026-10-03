# P2 appliance v2-large — synthesis of cl_aleph_bp, sourced by the AWS F2 HDK flow (build_all.tcl).
# Decoder clock: env ALEPH_CORE_MHZ = 125 (default) or 150, see design/cl_aleph_bp_defines.vh.

source ${HDK_SHELL_DIR}/build/scripts/synth_cl_header.tcl

set aleph_defines [list XSDB_SLV_DIS]
if { [info exists ::env(ALEPH_CORE_MHZ)] && $::env(ALEPH_CORE_MHZ) == 150 } {
  lappend aleph_defines ALEPH_150
}
print "aleph: verilog defines = $aleph_defines"

print "Reading user source code"
read_verilog -sv [glob ${src_post_enc_dir}/*.{s,}v]

print "Reading user constraints"
read_xdc [list ${constraints_dir}/cl_synth_user.xdc ${constraints_dir}/cl_timing_user.xdc]
set_property PROCESSING_ORDER LATE [get_files cl_synth_user.xdc]
set_property PROCESSING_ORDER LATE [get_files cl_timing_user.xdc]

print "Starting synthesizing customer design ${CL}"
update_compile_order -fileset sources_1

synth_design -mode out_of_context \
             -top ${CL} \
             -verilog_define $aleph_defines \
             -part ${DEVICE_TYPE} \
             -keep_equivalent_registers

source ${HDK_SHELL_DIR}/build/scripts/synth_cl_footer.tcl
