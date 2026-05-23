# Creating GitHub Issues from BACKLOG.md

> Instructions for **Claude Code** (or a human running `gh` manually) to populate the GitHub repository with issues, labels, and milestones from `BACKLOG.md`.

-----

## TL;DR for Claude Code

Read `BACKLOG.md` and `ROADMAP.md` first. Then:

1. Create the labels listed in `BACKLOG.md` → “Label System”.
1. Create the milestones listed in `BACKLOG.md` → “Milestones”.
1. For each issue in `BACKLOG.md` (sections starting with `### [Pn-nn]`), create a GitHub issue via `gh issue create`. The issue title is the `[Pn-nn] Title` line (without brackets). The body is all subsequent content up to the next `###` heading. Apply labels and milestone from the issue’s metadata.
1. After all issues are created, run the verification step.

Be **idempotent**: before creating each issue, check whether an issue with the same `[Pn-nn]` prefix already exists. If yes, skip (or update if `--update` flag is set). This lets the script be re-run safely after backlog edits.

-----

## Prerequisites

1. **GitHub CLI** installed and authenticated:
   
   ```bash
   gh auth status
   # Should report: "Logged in to github.com as <you>"
   ```
   
   If not: `gh auth login` and follow prompts.
1. **Repository exists** on GitHub and is the current working directory’s remote `origin`. If not:
   
   ```bash
   gh repo create <owner>/<repo> --private --source=. --remote=origin --push
   ```
1. **GitHub permissions**: the authenticated user must have write access to Issues, Labels, and Milestones on the target repo.

-----

## Step 1 — Create Labels

The label list comes from `BACKLOG.md` § Label System. Run this script:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Area labels (purple-ish)
gh label create "area:core"        --color "BFD4F2" --description "Core types and primitives" --force
gh label create "area:parser"      --color "BFD4F2" --description "OpenQASM and other parsers" --force
gh label create "area:ir"          --color "BFD4F2" --description "Circuit IR and optimization passes" --force
gh label create "area:backend"     --color "BFD4F2" --description "Backend trait and dispatch" --force
gh label create "area:backend-sv"  --color "BFD4F2" --description "State vector backend" --force
gh label create "area:backend-mps" --color "BFD4F2" --description "MPS tensor network backend" --force
gh label create "area:backend-stab" --color "BFD4F2" --description "Stabilizer backend" --force
gh label create "area:backend-gpu" --color "BFD4F2" --description "GPU acceleration" --force
gh label create "area:backend-dist" --color "BFD4F2" --description "Distributed / multi-GPU" --force
gh label create "area:bench"       --color "BFD4F2" --description "Benchmarking" --force
gh label create "area:infra"       --color "BFD4F2" --description "Build, CI, tooling" --force
gh label create "area:docs"        --color "BFD4F2" --description "Documentation" --force
gh label create "area:python"      --color "BFD4F2" --description "Python bindings" --force
gh label create "area:cli"         --color "BFD4F2" --description "CLI tool" --force

# Type labels (green/blue)
gh label create "type:feature"      --color "1D76DB" --description "New feature or capability" --force
gh label create "type:optimization" --color "0E8A16" --description "Performance optimization" --force
gh label create "type:bug"          --color "D73A4A" --description "Defect fix" --force
gh label create "type:refactor"     --color "FBCA04" --description "Code cleanup, no behavior change" --force
gh label create "type:test"         --color "5319E7" --description "Tests or test infrastructure" --force
gh label create "type:docs"         --color "0075CA" --description "Documentation changes" --force
gh label create "type:infra"        --color "C5DEF5" --description "Build / CI / tooling" --force

# Priority labels (red gradient)
gh label create "priority:critical" --color "B60205" --description "Blocking; must address now" --force
gh label create "priority:high"     --color "D93F0B" --description "Important; address soon" --force
gh label create "priority:medium"   --color "FBCA04" --description "Address when possible" --force
gh label create "priority:low"      --color "C2E0C6" --description "Nice to have" --force

# Difficulty / community labels
gh label create "good-first-issue"  --color "7057FF" --description "Good for newcomers" --force
gh label create "help-wanted"       --color "008672" --description "Help from community welcome" --force
gh label create "research"          --color "FF6B6B" --description "Open-ended; novel territory" --force

echo "All labels created."
```

Save this as `scripts/create-labels.sh`, `chmod +x`, and run.

-----

## Step 2 — Create Milestones

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)

create_milestone() {
  local title="$1"
  local description="$2"
  
  # Check if milestone already exists
  existing=$(gh api "repos/$REPO/milestones" --jq ".[] | select(.title==\"$title\") | .number" 2>/dev/null || true)
  if [ -n "$existing" ]; then
    echo "Milestone '$title' already exists (#$existing); skipping."
    return
  fi
  
  gh api "repos/$REPO/milestones" \
    --method POST \
    --field title="$title" \
    --field description="$description" \
    --field state="open" >/dev/null
  echo "Created milestone: $title"
}

create_milestone "Phase 0 — Foundation" \
  "Working end-to-end pipeline: parser → IR → naive backend → measurement. Correctness over speed."
create_milestone "Phase 1 — Single-Thread CPU Optimization" \
  "Single-thread state vector backend within 2× of Qiskit Aer single-thread."
create_milestone "Phase 2 — Multi-Thread CPU" \
  "Near-linear scaling on 16+ cores."
create_milestone "Phase 3 — Alternative Backends" \
  "Stabilizer + MPS backends working; automatic backend selection."
create_milestone "Phase 4 — Algorithm Benchmarks & v0.1 Release" \
  "Comprehensive benchmarks; first public release."
create_milestone "Phase 5 — GPU Backend" \
  "GPU state vector backend within 1.5× of cuQuantum standalone."
create_milestone "Phase 6 — Multi-GPU & Distributed" \
  "Distributed state vector across multiple GPUs and nodes."

echo "All milestones created."
```

Save as `scripts/create-milestones.sh` and run.

-----

## Step 3 — Create Issues from BACKLOG.md

### For Claude Code (recommended)

Tell Claude Code:

> Read `BACKLOG.md`. For each issue section (those starting with `### [P{n}-{nn}] Title`), do the following:
> 
> 1. Parse the issue ID (e.g., `P0-01`), title, labels, milestone, estimate, dependencies, and full body.
> 1. Check if an issue with that ID prefix already exists: `gh issue list --search "[P0-01] in:title" --json number,title`.
> 1. If no existing issue: create with `gh issue create --title "[P0-01] Setup Rust workspace and project structure" --milestone "Phase 0 — Foundation" --label "area:infra,type:infra,priority:critical" --body-file <tmp>`.
> 1. Write the body to a temp file first to handle multi-line content cleanly.
> 1. Track progress and print a summary at the end.
> 1. If a flag `--update` is passed, update existing issues’ bodies rather than skipping.

Claude Code will iterate through all ~60 issues. Expect this to take a few minutes due to API rate limits.

### Alternative: Shell script driver

If running by hand without an AI agent:

```bash
#!/usr/bin/env bash
# scripts/sync-issues.sh
# Naive but functional issue creator. Assumes BACKLOG.md is well-formed.
set -euo pipefail

BACKLOG="BACKLOG.md"
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# Split BACKLOG.md by issue (### [Pn-nn] ...)
csplit -z -f "$TMPDIR/issue-" -b "%03d.md" "$BACKLOG" '/^### \[P[0-9]\+-[0-9]\+\]/' '{*}' >/dev/null

for f in "$TMPDIR"/issue-*.md; do
  # Skip the file that contains the preamble (before any issue)
  if ! head -n 1 "$f" | grep -q '^### \[P[0-9]\+-[0-9]\+\]'; then
    continue
  fi
  
  # Extract title line
  title_line=$(head -n 1 "$f")
  # "### [P0-01] Setup Rust workspace and project structure" → "[P0-01] Setup Rust workspace and project structure"
  title="${title_line#### }"
  id=$(echo "$title" | grep -oE '\[P[0-9]+-[0-9]+\]' | tr -d '[]')
  
  # Check if issue already exists
  if gh issue list --search "[$id] in:title" --state all --json title --jq '.[].title' | grep -qF "$title"; then
    echo "Skipping existing: $title"
    continue
  fi
  
  # Extract labels (from "**Labels:** `a`, `b`, `c`" line)
  labels=$(grep -m 1 '^\*\*Labels:\*\*' "$f" | sed 's/\*\*Labels:\*\* *//; s/`//g; s/ //g')
  
  # Extract milestone (from "**Milestone:** Phase n" line)
  milestone_num=$(grep -m 1 '^\*\*Milestone:\*\* *Phase' "$f" | grep -oE 'Phase [0-9]+' | head -1 | awk '{print $2}')
  milestone="Phase $milestone_num — "
  case $milestone_num in
    0) milestone+="Foundation" ;;
    1) milestone+="Single-Thread CPU Optimization" ;;
    2) milestone+="Multi-Thread CPU" ;;
    3) milestone+="Alternative Backends" ;;
    4) milestone+="Algorithm Benchmarks & v0.1 Release" ;;
    5) milestone+="GPU Backend" ;;
    6) milestone+="Multi-GPU & Distributed" ;;
  esac
  
  # Body is everything after the title line
  tail -n +2 "$f" > "$TMPDIR/body.md"
  
  echo "Creating: $title"
  gh issue create \
    --title "$title" \
    --label "$labels" \
    --milestone "$milestone" \
    --body-file "$TMPDIR/body.md"
done

echo "Done."
```

Run with: `bash scripts/sync-issues.sh`.

This script is **idempotent**: re-running skips issues that already exist (matched by title). To update existing issues’ bodies, use Claude Code with the `--update` flag.

-----

## Step 4 — Verify

```bash
# Count issues created
gh issue list --state all --limit 100 --json number,title --jq 'length'
# Should match the count in BACKLOG.md Appendix (60 by default)

# List by milestone
for n in 0 1 2 3 4 5 6; do
  count=$(gh issue list --state all --milestone "Phase $n — " --limit 100 --json number --jq 'length')
  echo "Phase $n: $count issues"
done

# List by label (sanity)
gh issue list --state all --label "priority:critical" --limit 100 --json number,title
```

Expected counts (from `BACKLOG.md` § Appendix):

- Phase 0: 12
- Phase 1: 14
- Phase 2: 5
- Phase 3: 7
- Phase 4: 8
- Phase 5: 8
- Phase 6: 6
- **Total: 60**

-----

## Step 5 — Set Up a Project Board (Optional)

```bash
# Create a project board for tracking
gh project create --owner "@me" --title "Quantum Simulator Roadmap"

# Add all issues
for issue in $(gh issue list --state open --limit 100 --json number --jq '.[].number'); do
  gh project item-add <project-number> --owner "@me" --url "https://github.com/<owner>/<repo>/issues/$issue"
done
```

Configure columns: `Backlog`, `Ready`, `In Progress`, `In Review`, `Done`.
Add views by milestone, by label, etc.

-----

## Updating Issues After Backlog Edits

Workflow:

1. Edit `BACKLOG.md`.
1. Commit the change.
1. Re-run issue sync. For Claude Code:

> Re-read `BACKLOG.md` and update any existing issues whose body has diverged. Add any new issues. Do not delete issues missing from the backlog (just label them `stale` if needed).
1. For the shell script: re-run with explicit update mode (you’d need to extend the script to support `--update`).

-----

## Tips for Working with Claude Code on This Repo

When invoking Claude Code, give it the full context up front:

```
Read ROADMAP.md and BACKLOG.md. Then read CREATE_ISSUES.md.
Your task: ensure the GitHub repository has all 60 issues from BACKLOG.md
created with correct labels and milestones. Be idempotent — skip existing
issues. Print a summary at the end.
```

If you want Claude Code to also start implementing:

```
Read ROADMAP.md, BACKLOG.md, and CREATE_ISSUES.md. After ensuring the
GitHub backlog is synced, start work on issue P0-01. Open a branch
`p0-01-rust-workspace`, follow the acceptance criteria, open a draft PR.
```

Keep PRs small (one issue per PR, typically). Tag the PR with the issue ID in the title: `[P0-01] Set up Rust workspace`.

-----

## Common Gotchas

- **gh CLI rate limits**: ~5000 requests/hour for authenticated users. 60 issue creates + label lookups stay well within this.
- **Label color format**: hex without `#` prefix.
- **Milestone matching**: `gh issue create --milestone` matches by title exactly; even one extra space breaks it. Use the exact strings from `create-milestones.sh`.
- **Body file with backticks**: use `--body-file` not `--body` to avoid shell-escaping nightmares with markdown content.
- **Em-dash in milestone names**: the `—` character is U+2014. Copy-paste it verbatim; do not type a regular dash.

-----

## File Reference

- `ROADMAP.md` — strategic overview.
- `BACKLOG.md` — detailed issue specifications (source of truth).
- `CREATE_ISSUES.md` — this file.
- `scripts/create-labels.sh` — label creation (copy from Step 1).
- `scripts/create-milestones.sh` — milestone creation (copy from Step 2).
- `scripts/sync-issues.sh` — issue creation/update (copy from Step 3).