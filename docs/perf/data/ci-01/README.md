# CI-01 validation and routing evidence

Base: `f9781d05b3d5087b3265971435849ec8eeafdb12` after report PR #596.
This is infrastructure validation, not a simulator throughput campaign.

- `baseline-jobs.json`: all eight successful jobs from documentation PR #596's
  original CI run, including step/start/end observations. Job-active time,
  elapsed critical path and queue wait are distinct; this is not billing data.
- `shadow.json`: replay of #596's complete historical diff plus focused synthetic
  Metal/CUDA/CPU/QEC/shared/frontend/hardware/config paths against the **current**
  Cargo consumer graph. Every record labels its evidence scope. The synthetic
  paths identify package ownership; they are not claims those exact files were
  edited in independent real PRs. `tested_sha` identifies the checkout base for
  this local replay, not an executed GitHub test-merge result. Planner source
  fingerprints in `validation.json` identify the working implementation.
- `validation.json` and logs: scoped tests, workflow lint, feature-consumer builds,
  source hashes and commands for this PR. No new physical GPU performance run.

The real #596 docs diff selects only planner/docs/final gate and passes the actual
new docs checker locally. The local checker elapsed value is diagnostic on a busy
Mac, not evidence of hosted wall/queue savings. The CLI integration test separately
runs the planner against temporary real PR/push Git histories with a deliberately
unusable Cargo executable, proving the docs route never initializes Rust.

Affected backend plans retain CLI/Python feature consumers. CPU/oracle dev edges
make CPU and QEC plans broad; this is reported rather than silently pruning tests.
The mixed/config controls select code/full routes, and matrix/gate negative tests
reject missing/cancelled/skipped selected results. Full GitHub CI is required on
the infrastructure PR; its own policy change cannot take the shortcut.

Hosted docs timing (target roughly 1–3 active minutes), real queue/runner savings
and representative future backend PR observations require post-merge runs. Until
those exist, **no measured hosted speedup is claimed**. Full main-code/nightly
validation is retained as a rollout backstop, not as a substitute for known
pre-merge consumers. This evidence does not authorize changing branch protection
or waiving existing merge-monitor conditions.

## First hosted full run: timeout diagnosis (2026-10-20)

Run 39162811229 passed planner/docs, Linux stable/beta, Clippy, both Python
versions, macOS beta and RTL. macOS stable was cancelled at the 30-minute limit.
Its workspace suite finished successfully by 20:45:26 UTC on October 18; the
Metal feature suite finished successfully by 20:50:20. Cancellation at 20:50:35
occurred during the next Metal Clippy dependency build. The final aggregate
correctly rejected `macos: cancelled`; no cancelled result is counted as a pass.
The macOS job budget is now 45 minutes to cover cold release/test compilation
and the newly checked feature consumers. No command, test or failure gate was
removed. A fresh exact-head CI run is required after this workflow change.
