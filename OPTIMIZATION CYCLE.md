# Optimization Cycle

> **Reusable step-by-step playbook for one optimization iteration.** Apply this to every optimization PR. Companion to `OPTIMIZATION GUIDE.md`.

-----

## Overview

One **optimization cycle** = one targeted improvement to one kernel or algorithm, ending in a merged PR with measurable speedup. Cycles are sequential and atomic. Don’t run two in parallel on the same code path.

The cycle has **10 steps**, grouped into 4 stages:

```
┌─────────────────────┐
│  STAGE 1: Setup     │ Steps 1-3: define what we're doing
├─────────────────────┤
│  STAGE 2: Analyze   │ Steps 4-5: profile, hypothesize
├─────────────────────┤
│  STAGE 3: Execute   │ Steps 6-8: implement, measure, validate
├─────────────────────┤
│  STAGE 4: Decide    │ Steps 9-10: merge or iterate; document
└─────────────────────┘
```

Each step has explicit inputs, outputs, and exit criteria. Skipping steps is the most common cause of failed optimizations.

-----

## Stage 1 — Setup

### Step 1: Pick the target

**Goal**: choose exactly one thing to optimize.

**Inputs**:

- Current phase from `ROADMAP.md`.
- Open issues from `BACKLOG.md` labeled `type:optimization`.
- Previous phase’s performance report (`docs/perf/phase{n-1}.md`).

**Process**:

1. Among open `type:optimization` issues in the current milestone, prefer:
- Highest priority (`priority:critical` > `priority:high`).
- Smallest dependency set (no blocking issues).
- On the optimization hierarchy (see `OPTIMIZATION GUIDE.md` § 3), highest unconsidered rank.
1. If multiple issues qualify, pick by estimated impact × inverse of effort.
1. **Don’t** improvise an optimization not in the backlog. If you found a new opportunity, file an issue first, then pick it.

**Output**: Issue ID (e.g., `P1-03 — SIMD AVX2 for 1-qubit gates`).

**Exit criterion**: The issue exists in `BACKLOG.md` and is assigned to you.

-----

### Step 2: Answer the Four Questions

**Goal**: write down what success looks like before doing any work.

**Inputs**: the chosen issue.

**Process**: post these answers as a comment on the GitHub Issue.

**Q1: What are we measuring?**

- Specific benchmark name (must exist in the benchmark suite; if not, file a sub-issue to add it first).
- Hardware reference (`workstation` / `server` / `gpu_box`).
- Thread count, RUSTFLAGS.

**Q2: What’s the current value?**

- Run the benchmark on the reference hardware.
- Record: median time, std-dev, commit hash, date.

**Q3: What’s the target?**

- A specific number. Derived from: external reference (Qiskit Aer time), roofline ceiling, or “% of peak bandwidth”.
- Format: “X ms (1.5× faster than baseline)” or “≥80% of memory bandwidth ceiling”.

**Q4: What’s the hypothesis?**

- One or two sentences. “Lever X will improve metric Y by mechanism Z.”
- Example: “Hand-written AVX2 intrinsics will improve `apply_1q_gate` throughput by 2× because the compiler currently generates scalar code for the SoA inner loop (confirmed by inspecting LLVM IR).”

**Output**: comment posted on issue with answers.

**Exit criterion**: All four answers are concrete and specific. “Faster” is not an answer.

-----

### Step 3: Create the branch

**Goal**: isolated workspace for this cycle.

**Process**:

```bash
git checkout main
git pull
git checkout -b p{phase}-{issue}-{short-desc}
# Example: git checkout -b p1-03-avx2-1q-gates
```

**Output**: clean branch from main.

**Exit criterion**: `git status` clean; branch exists.

-----

## Stage 2 — Analyze

### Step 4: Profile the baseline

**Goal**: understand *why* the current code performs as it does. Don’t skip even if you think you know.

**Inputs**: baseline benchmark from Step 2.

**Process**:

1. Run high-level counters:
   
   ```bash
   cargo bench --bench {target} --no-run
   perf stat -e cycles,instructions,cache-misses,cache-references,\
     branch-misses,L1-dcache-load-misses,LLC-load-misses \
     ./target/release/deps/{benchmark-binary} --bench
   ```
1. Run flamegraph:
   
   ```bash
   cargo flamegraph --bench {target} -- --bench
   open flamegraph.svg  # or xdg-open
   ```
1. Identify the top-3 functions by self time.
1. For each top function, compute:
- IPC (instructions / cycle).
- L1 miss rate (L1-dcache-load-misses / L1-dcache-loads).
- LLC miss rate.
- Branch miss rate.
1. Compare to roofline ceiling (from `OPTIMIZATION GUIDE.md` § 4).

**Output**: file `docs/perf/scratch/{issue-id}-profile.md` with the numbers and a brief interpretation.

**Exit criterion**: You can name the bottleneck in one sentence. Example: “Memory-bound at 60% of peak bandwidth; L1 misses at 8% on the inner loop.” If you can’t, profile more.

-----

### Step 5: Refine the hypothesis

**Goal**: profile data either confirms or refutes the hypothesis from Step 2. Update accordingly.

**Process**:

- If profile data **confirms** the hypothesis: proceed.
- If profile data **partially confirms**: refine the hypothesis to match. Re-estimate the target if needed.
- If profile data **contradicts**: stop. Either pick a different optimization or first fix the wrong assumption (often: the wrong bottleneck was assumed). Re-open Step 2.

**Output**: updated hypothesis posted as a comment on the issue.

**Exit criterion**: hypothesis matches profile data; expected speedup is grounded in observed bottleneck.

-----

## Stage 3 — Execute

### Step 6: Implement

**Goal**: write the optimization.

**Process**:

1. **Smallest possible change.** If the optimization can be staged (e.g., add SoA, then add SIMD on top of SoA), do it in separate PRs. One PR = one lever.
1. **Keep the old code path** accessible for testing. Either via feature flag or by writing the new code as an additional function that’s wired in via the dispatch layer. The old version stays as the reference until the new one is validated and merged.
1. **Add tests first** if there isn’t already coverage of the kernel you’re changing. Use property tests where possible.
1. **Don’t refactor unrelated code.** If you see something else that needs fixing, file an issue and move on.
1. **Comment the why, not the what.** Especially around tricky bit manipulation, SIMD shuffles, or unsafe blocks.

**Output**: commits on the branch.

**Exit criterion**:

- Code compiles without warnings.
- `cargo clippy --all-targets -- -D warnings` clean.
- `cargo fmt --check` clean.
- New tests added if needed.

-----

### Step 7: Measure

**Goal**: compare new vs. baseline with rigor.

**Process**:

1. Run benchmarks with `--baseline` mode:
   
   ```bash
   git checkout main
   cargo bench --bench {target} -- --save-baseline before
   git checkout {your-branch}
   cargo bench --bench {target} -- --baseline before
   ```
1. Run the **full benchmark suite**, not just the targeted one:
   
   ```bash
   cargo bench --workspace -- --baseline before
   ```
   
   This catches regressions elsewhere.
1. **Quiet the machine**:
- Close other processes.
- Disable CPU frequency scaling (Linux: `sudo cpupower frequency-set -g performance`).
- Disable turbo if you want consistent numbers (otherwise document that turbo was on).
- Run benchmarks ≥3 times; report median or trim outliers.
1. Re-profile with the new code:
   
   ```bash
   perf stat -e cycles,instructions,cache-misses,cache-references ...
   ```
   
   Confirm the bottleneck moved as predicted.
1. Compute roofline percentage. If still <85% and you can do more, decide whether to continue this cycle or stop and file a follow-up.

**Output**: benchmark results in the format from `OPTIMIZATION GUIDE.md` § 12.

**Exit criterion**:

- Improvement ≥5% on the target metric (smaller is noise).
- No regression >2% on any other benchmark (or, if there is, it’s justified and documented).
- Roofline percentage computed.

-----

### Step 8: Validate correctness

**Goal**: confirm the optimization preserves semantics.

**Process**:

1. **All existing tests pass**:
   
   ```bash
   cargo test --workspace
   ```
1. **Oracle tests pass**:
   
   ```bash
   cargo test --workspace --features oracle-tests
   ```
   
   This includes Qiskit / Stim comparisons.
1. **Property tests pass**:
   
   ```bash
   cargo test --workspace -- proptest
   ```
   
   Run with extended iteration count if the optimization touches numerical code:
   
   ```bash
   PROPTEST_CASES=10000 cargo test --workspace -- proptest
   ```
1. **Numerical tolerance**: confirm amplitudes match reference to ≤1e-10. Document any tolerance changes.
1. **Edge cases**: explicitly test:
- 1-qubit circuits.
- Maximum-supported qubit count (e.g., 28 for state vector).
- All gates the kernel can dispatch.

**Output**: test report in PR description.

**Exit criterion**: every test in the test suite passes, with no skipped tests added to bypass failures.

-----

## Stage 4 — Decide

### Step 9: Open and review the PR

**Goal**: get the optimization merged or rejected with full information.

**Process**:

1. Push the branch and open the PR:
   
   ```bash
   git push origin {your-branch}
   gh pr create --title "[P{n}-{nn}] {Brief description}" \
     --body-file pr_body.md
   ```
1. PR body template (also in `.github/PULL_REQUEST_TEMPLATE.md`):
   
   ```markdown
   ## Summary
   One paragraph: what this PR does.
   
   Closes #{issue-number}
   
   ## Approach
   - High-level approach.
   - Trade-offs accepted.
   - Anything left out (and why).
   
   ## Benchmark Results
   {as per OPTIMIZATION GUIDE.md § 12}
   
   ## Profile Analysis
   - Bottleneck before: ...
   - Bottleneck after: ...
   - Roofline %: ...
   
   ## Correctness
   - [ ] All tests pass
   - [ ] Oracle tests pass
   - [ ] Property tests pass at default count
   - [ ] No new `unsafe` without `// SAFETY:` block
   - [ ] No tolerance changes (or, if changed: justified above)
   
   ## Follow-ups
   - Issues filed for further opportunities.
   ```
1. **Self-review** the diff in the GitHub UI before requesting review. Fresh eyes catch silly mistakes.
1. Wait for CI green.
1. Address review feedback. **Don’t bundle unrelated changes** into the same PR during review.

**Exit criterion**: PR is merged (or explicitly closed with rationale).

-----

### Step 10: Update tracking and decide what’s next

**Goal**: capture what was learned; pick the next cycle.

**Process**:

1. **Update the perf log** at `docs/perf/scratch/{phase}-log.md`:
   
   ```markdown
   ## 2025-XX-XX — [P1-03] AVX2 1-qubit gates
   
   **Result**: 1.63× speedup on qft_20_singlethread.
   **Bottleneck moved**: from L1 misses (8%) to LLC misses (3%).
   **New ceiling proximity**: 80% of memory bandwidth.
   **Next**: AVX-512 will likely give another ~1.8× on supported hardware (P1-04).
   **Surprises**: GHZ benchmark was unaffected (already memory-bound in a different way).
   ```
1. **Close the GitHub issue** (auto-closed by `Closes #N`).
1. **Update the phase performance report** (`docs/perf/phase{n}.md`) — append the new row.
1. **Decide whether the original problem is solved**:
- If yes, return to Step 1 with the next issue.
- If no, file a follow-up issue with what’s left, and return to Step 1.

**Exit criterion**: the cycle ends with either a closed problem or a clearly-defined follow-up.

-----

## When to Abort a Cycle

Some cycles fail. Aborting cleanly is better than forcing.

**Abort if**:

- After Step 5, no clean hypothesis emerges. The bottleneck is something other than expected.
- After Step 7, the measured speedup is <5%. The lever doesn’t pay off.
- After Step 8, oracle tests fail and the fix is unclear. Don’t ship correctness regressions.

**How to abort**:

1. Close the branch without merging.
1. Update the issue with what was learned: “Tried X, observed Y; reason it didn’t work: Z.”
1. File follow-up issues if the investigation revealed new opportunities.
1. Don’t delete the branch — keep it for ~1 month as a reference.

Aborted cycles are valuable. They eliminate dead-ends so future cycles don’t repeat the work. The cost is the time invested; the benefit is the eliminated false path.

-----

## Example: A Cycle from Start to Finish

Concrete example to make the process tangible.

**Cycle**: P1-09 — Gate fusion pass (adjacent 1q gates).

**Step 1 — Pick**: P1-09 is open, priority:critical, depends only on Phase 0 items already merged.

**Step 2 — Four Questions**:

- Q1: `vqe_h2_ansatz_depth4_singlethread` on workstation.
- Q2: 18.4 ± 0.3 ms (commit abc123, 2025-XX-XX).
- Q3: ≤9 ms (≥2× speedup), based on observed 60% of gates being fusable 1q sequences.
- Q4: Fusing adjacent 1q gates into one Unitary1q reduces state vector passes by ~60%, expected speedup ~2× (linear in passes since each pass is memory-bound).

**Step 3 — Branch**: `p1-09-fuse-1q-gates`.

**Step 4 — Profile**: flamegraph shows 70% of time in `apply_1q_gate`, called ~3× more often than necessary. perf stat shows 1.8 IPC, L1 miss rate 1% (already memory-bound).

**Step 5 — Refine**: hypothesis confirmed; expected 2× speedup is realistic.

**Step 6 — Implement**: add `Fuse1qPass` to `aleph-ir`. Linear scan, maintain per-qubit pending 2×2 matrix, flush at 2q gates or barriers. ~150 LOC + tests.

**Step 7 — Measure**:

|Benchmark           |Before |After  |Speedup                 |
|--------------------|-------|-------|------------------------|
|vqe_h2_ansatz_depth4|18.4 ms|9.1 ms |2.02×                   |
|qft_20              |142 ms |138 ms |1.03× (mostly 2q anyway)|
|grover_16           |89 ms  |88 ms  |1.01×                   |
|ghz_20              |12.5 ms|12.5 ms|1.00×                   |

**Step 8 — Validate**: All tests pass. Oracle equivalence within 1e-12. Property tests pass at 10k cases.

**Step 9 — PR**: opened with full details, CI green, merged after self-review and approval.

**Step 10 — Update**: phase 1 perf log updated. VQE H₂ now within 1.3× of Qiskit Aer (was 2.7×). Next: P1-10 (2q+1q fusion) for further gains.

Cycle complete. Total time: 2.5 days.

-----

## Cycle Checklist (Tear-off)

Copy this into the GitHub Issue when starting a cycle.

```markdown
## Cycle Checklist

### Stage 1 — Setup
- [ ] Step 1: Issue selected from BACKLOG.md
- [ ] Step 2: Four Questions answered
- [ ] Step 3: Branch created

### Stage 2 — Analyze
- [ ] Step 4: Baseline profiled; bottleneck identified
- [ ] Step 5: Hypothesis confirmed or refined

### Stage 3 — Execute
- [ ] Step 6: Implementation complete; lint/fmt clean
- [ ] Step 7: Benchmark shows ≥5% improvement; no regression >2%
- [ ] Step 8: All tests pass; oracle tests pass

### Stage 4 — Decide
- [ ] Step 9: PR opened with full template; CI green; merged
- [ ] Step 10: Perf log updated; phase report appended; follow-ups filed
```