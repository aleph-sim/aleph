# #520 follow-up protocol (2026-11-01)

The October report and 87 recovered local Criterion cells are historical data,
not a fresh rerun. Local mtimes/value matches establish a recovery link but do
not reconstruct missing source/process metadata. Recovered sampling modes refine the estimator label: Criterion's reported
point uses mean for Flat and slope for Linear sampling, despite old prose calling
both medians. Preserve those numbers and correct the label, rather than silently
swapping their estimator. This descriptive correction does not change the frozen
measurement scope, sample counts or median-based drift threshold.

The identified missing performance cell is existing mps_vs_cpu/large_n/gpu/n20_chi512.
Use the unchanged production MPS runtime, source/fixture and existing Criterion
bench: 10 samples, default 3 s warmup and 5 s requested measurement. Do not stop a
long cell just because it is uncompetitive; retain raw samples and confidence
intervals. The current CPU pair is measured before and after that long GPU cell.

Measure the same SV random n18 raw/fused control before/after. A >10% change of
median is reported as material observed drift, not proof of thermal throttling.
Refresh sv_readout and mps_batched between the SV controls. Do not claim a fresh
full sv_vs_cpu n24/26/28 or full MPS sweep; the October campaign already contains
those completed cells, and the new scope is explicitly bounded to its gaps.

Compile fully before timings. Run the normal Metal correctness suite and the
existing ignored chi512 n16/20 norm test before timing. Record timestamps,
load, power supply, battery charge and available pmset thermal status every 10s.
Power transitions mark the affected block mixed-power; do not pool it with a
stable-supply block or claim a clean cross-machine comparison. A stable battery
campaign is labeled battery-only, not AC-equivalent. No user services are stopped.

Safe reach: query maxBufferLength/recommended working set, exercise public n28
GHZ with a full finite/norm scan, and verify public n29 returns the existing
TooManyQubits limit 28 before any allocation. The cap existed in the old revision
too. Do not override it or turn a device buffer limit into a tested circuit reach.
No large-allocation escalation or intentional OOM is part of this task.
