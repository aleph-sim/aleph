# P2 appliance v2-large — place-and-route constraints for cl_aleph_bp on the F2 small shell.
# clk_hbm_ref is itself an MMCM output inside the shell, so the CL's MMCM is a cascade; allow the router to
# reach any CMT column (the shell does the same for its own cascade, mmcm_cascade.xdc).
set_property CLOCK_DEDICATED_ROUTE ANY_CMT_COLUMN [get_nets -of_objects [get_pins -hierarchical -filter {NAME =~ *u_aleph/u_mmcm/CLKIN1}]]
