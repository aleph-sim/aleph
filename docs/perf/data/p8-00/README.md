# Phase 8 integration-report receipts

Collected 2026-10-18 from main `6114dab7e18c4deac0fda3039060e00f7661b653`.
The report changes documentation only; no benchmark or production source changed.
Command logs normalize trailing blank lines; test output content is otherwise retained.

- [status.json](status.json): GitHub issue/PR state snapshot; 13/16 capabilities
  closed, #551/#561/#562 open. #550 stays open. PR heads and merge commits identify
  integration history, not the historical binaries used by child campaigns.
- [validation.json](validation.json): current local platform/toolchain, freshly
  built release CPU CLI SHA256, command argv and exits. RAYON_NUM_THREADS=2 bounds
  local correctness checks. These are not accepted timing observations.
- [check-1.log](check-1.log): 13 public-workflow contracts including a real CPU CLI
  call. Its test mocks resource admission to check output/receipt semantics on CI
  or a busy Mac; it does not establish idle-machine performance.
- [check-2.log](check-2.log): 6 pure memory-policy tests.
- [check-3.log](check-3.log): 11 sampler/noise integration and stored oracle tests.
  The Aer fixtures are stored references, not a new live Aer campaign.
- [check-4.log](check-4.log): 7 binomial + 8 bounded-experiment tests. One existing
  external PyMatching test remains ignored; it is not counted as a pass.
- [check-5.log](check-5.log): 3 threshold-plot schema tests.
- [saved-receipts.json](saved-receipts.json): all six existing P8-13 server campaign
  directories independently passed their saved-record consistency check. Four
  have `executed` status and exit 0. The failed/contaminated records correctly
  exit 2 and contain no successful summary. `passed` here means **revalidation of
  the saved classification**, not successful original hardware execution.
- [cli-smoke.txt](cli-smoke.txt): documented Bell counts command on this CLI.

No GPU, Metal, remote-server or fresh performance campaign was run for this report.
Historical numerical results are linked from the report to their child evidence;
their timestamps/source hashes must not be replaced by this validation identity.
