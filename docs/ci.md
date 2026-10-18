# CI selection and merge policy

Issue [#597](https://github.com/aleph-sim/aleph/issues/597) implements a docs fast
path and dependency-aware code checks in one PR. The workflow definitions, planner,
command builder and final gate are one review surface; changing any of them selects
full CI. Do not waive the checks of an older PR because this policy is proposed.

## Routes

| Change | PR checks |
|---|---|
| Approved root prose, docs Markdown/images, identified performance receipts | Hosted planner contracts, changed-document checks, `ci-required`. No Cargo metadata/build, self-hosted allocation, wheels, Metal/CUDA or RTL execution. |
| Metal implementation/shaders only | Metal, CLI and Python consumers on macOS, stable/beta, real Metal-feature builds/tests/lint; Python 3.10/3.12 Metal-feature wheels and existing binding contracts. Linux/CUDA and RTL omitted. |
| CUDA implementation only | CUDA/NCCL lint/build and pure contracts, CUDA-feature CLI/Python consumers, Linux stable/beta selected-package tests, macOS selected-package default compatibility, Python CUDA-feature wheels. No unrelated Metal feature/RTL suite. |
| Frontend code | Selected frontend package tests plus both supported GPU feature configurations; Python changes cannot hide cfg-gated GPU code. |
| CPU/shared/QEC code | Reverse dependency closure including optional, dev, build and target-specific consumers. Existing test/oracle dependencies legitimately make many selections broad. QEC consumers additionally select RTL golden checks. |
| Hardware only | Hosted documentation checks, RTL co-simulation, final gate. Golden-model Rust builds remain inside the hardware workflow. |
| Config/policy/manifests/lockfiles/toolchain/build scripts/unknown inputs | Full fallback. Mixed changes take the union, never the first matching directory. |

The actual package graph comes from `cargo metadata --locked --no-deps`, including inactive
optional/target and dev/build declarations as a conservative superset. It does not
assume a default-feature build covers an empty GPU crate. Python/public-benchmark
scripts have explicit frontend/GPU edges; other unmapped scripts trigger full CI.
New package roots are discovered, but changing their Cargo manifests selects full
validation until the graph/feature policy is reviewed.

Main **code-bearing pushes** retain full integration. Docs-only pushes keep the
fast path. Daily default-branch audits (03:17 UTC), `workflow_dispatch`, merge-group
validation and a `ci:full` PR label select full checks. A label can only broaden the
computed plan. Dispatch results are extra evidence, not a replacement for PR checks.
Timed benchmarks remain manual-only in `bench.yml`; release behavior is unchanged.

## Plan, execution and trust

`scripts/ci/plan.py` reads the GitHub event and a complete local NUL-delimited Git
diff. PRs compare merge-base to head; pushes compare before/after. Renames become
both deletion and addition. Missing ranges, unknown paths and symlinks fall back
broadly; failure to construct a valid plan fails CI. Native workflow path filters
and paginated/truncated file lists are not the selection authority.

`ci-plan.json` records paths, direct seeds, reverse consumer closure, feature flags,
selected groups/reasons and head/base/tested commit identities. It is uploaded once
and consumed in the same run. `run.py` creates argv lists, validates identities and
executes no shell fragments from filenames/package data. Full plans retain
workspace coverage; affected plans test selected packages and build their relevant
feature-enabled consumers. macOS remains release, Linux debug; MPS no-default
features and the existing pure CUDA contracts remain. Beta failures are explicitly
blocking, matching the prior manual policy that required every beta check green.

The planner/docs/gate always run on GitHub-hosted machines. Trusted repository
branches can use the existing Linux self-hosted runner for code; fork PRs use
hosted Linux instead. Runner trust is decided directly in workflow expressions,
never by an editable plan output. Tokens remain read-only, checkout credentials
are not persisted, and no `pull_request_target` executes head code. Workflow edits
still require review: no classifier can protect a repository from someone already
authorized to rewrite its own merge policy.

`hw.yml` is reusable and manually dispatchable. CI invokes it when selected and
includes its result in the aggregate, preventing an independent workflow from
running on docs-only main pushes. Its concurrency group is separate from the caller
so it cannot cancel its own CI run. Golden-generator dependencies now participate
through the QEC consumer closure rather than only a narrow hardware path filter.

## Required result

**`ci-required` must complete SUCCESS on the reviewed current PR revision.** It
runs with `always()`, requires successful planning and every selected group to have
succeeded, and accepts a skipped group only when the valid plan did not select it.
Missing, failed, cancelled, neutral or unexpectedly skipped selected results fail.
Matrices retain `fail-fast: false` and no `continue-on-error`, so one failed beta
or Python row cannot be hidden by another successful sibling. A missing plan
artifact fails the gate before any success is reported.

Inspect the plan and selected job outcomes during review. A skipped suite is
**unexecuted**, never a passing test. Compilation/no-device contracts do not prove
GPU execution; existing device skips retain that limitation. Physical CUDA/Metal
oracle/performance qualification is still a separate hardware activity.

This policy is not retroactive. Existing merge monitors with a fixed eight/nine-job
contract keep it until explicitly migrated. Future monitors must inspect current
head/base/mergeability/reviews, the successful `ci-required` plus its plan/selected
results and any additional failures, then use a matching-head merge guard. Never
infer success from an empty check list or enable a bypass. Branch protection/ruleset
migration, if desired, is a separate explicit repository-settings action; this PR
does not change settings. GitHub's [required-check semantics](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks)
explain why whole-workflow path skipping and `workflow_dispatch` are insufficient.

## Documentation checks

The allowlist is in `plan.py`: selected root prose; Markdown/images/PDF under docs
except fixture/testdata/generated trees; JSON/JSONL/CSV/text/log/archives and
SHA256SUMS under performance data. Executable source/patch/notebook/HTML and unknown
files do not get the shortcut. The initial repository consumer audit found report
inputs used by analysis tools, not automatic build/test fixtures; adding a consumer
must update this policy or keep the input out of the allowlist.

`docs.py` validates newly introduced inline local file/asset links (not live external
URLs or Markdown anchors), links to deleted assets, UTF-8 text, finite JSON/JSONL,
CSV parsing, SVG XML and adjacent SHA256SUMS. Existing unrelated broken links are
not made a prerequisite for a prose fix. Binary archives are inert assets; adjacent
checksums verify bytes, not the semantic contents of every historical campaign.
Known campaign receipt schemas can additionally use their existing independent
checkers. No fenced snippet is executed and no report timing is accepted as a new
performance measurement by these format checks.

## Validate and operate

```sh
python3 -m unittest discover -s scripts/ci -v
actionlint .github/workflows/ci.yml .github/workflows/hw.yml
python3 scripts/ci/replay.py --output /tmp/ci-shadow.json
# With a real event payload and the corresponding checkout/history:
GITHUB_EVENT_NAME=pull_request GITHUB_EVENT_PATH=/tmp/event.json \
  python3 scripts/ci/plan.py --output ci-plan.json
python3 scripts/ci/docs.py --plan ci-plan.json
python3 scripts/ci/run.py macos --toolchain stable --dry-run
```

The last command requires a plan selecting macOS. Never override the generated
plan to narrow a real PR. To broaden validation, apply `ci:full` (if available) or
use a full manual run as supplementary evidence. To roll back selection, revert
this workflow/planner change or force the planner to full for every event; preserve
the fail-closed gate. Do not remove failing jobs merely to make it green.

See [CI-01 evidence](perf/data/ci-01/README.md) for historical shadow routing,
local checks and the baseline docs run. Shadow fixtures prove routing/commands,
not hosted queue savings or physical GPU correctness. Real CI on this PR selects
full validation because it changes the policy itself. New post-merge docs/backend
runs should record queue wait, critical path and summed runner time separately;
there is no universal timing target for graph-connected code changes.
