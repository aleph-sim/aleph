#!/usr/bin/env bash
# scripts/sync-qec-issues.sh
#
# Create the QEC-decoder track labels, milestones, and GitHub issues from
# docs/qec/BACKLOG.md. Idempotent: re-running skips labels/milestones/issues
# that already exist (issues matched by exact title).
#
# Usage: bash scripts/sync-qec-issues.sh
#
# Mirrors the mainline `CREATE ISSUES.md` flow but for the `Q{n}-{nn}` track
# and the `Phase Q{n}` milestones.
set -euo pipefail

BACKLOG="docs/qec/BACKLOG.md"
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
echo "Repo: $REPO"

# ---------------------------------------------------------------------------
# Step 1 — labels (new ones for the QEC track; existing area/type/priority
# labels are reused as-is). --force makes this idempotent.
# ---------------------------------------------------------------------------
gh label create "area:qec"     --color "5319E7" --description "QEC codes, noise, syndromes, DEM" --force
gh label create "area:decoder" --color "5319E7" --description "Decoders: matching / union-find / BP" --force
gh label create "area:fpga"    --color "5319E7" --description "FPGA decoder implementation" --force
gh label create "area:asic"    --color "5319E7" --description "ASIC decoder (North Star)" --force

# ---------------------------------------------------------------------------
# Step 2 — milestones (Phase Q0 … Q7). Em-dash is U+2014, verbatim.
# ---------------------------------------------------------------------------
create_milestone() {
  local title="$1" description="$2"
  local existing
  existing=$(gh api "repos/$REPO/milestones" --jq ".[] | select(.title==\"$title\") | .number" 2>/dev/null || true)
  if [ -n "$existing" ]; then
    echo "Milestone '$title' already exists (#$existing); skipping."
    return
  fi
  gh api "repos/$REPO/milestones" --method POST \
    --field title="$title" --field description="$description" --field state="open" >/dev/null
  echo "Created milestone: $title"
}

create_milestone "Phase Q0 — Experiment Loop Foundation" \
  "Close the noise→syndrome→decode→logical-error loop; reproduce the surface-code threshold."
create_milestone "Phase Q1 — MWPM Decoder" \
  "From-scratch minimum-weight perfect matching decoder, benchmarked vs PyMatching."
create_milestone "Phase Q2 — Union-Find Decoder" \
  "Almost-linear-time Delfosse-Nickerson decoder; hardware-friendly precursor."
create_milestone "Phase Q3 — GPU Decoder" \
  "GPU decoder + end-to-end GPU Monte-Carlo. The differentiator (CUDA depth)."
create_milestone "Phase Q4 — Real-Time / Streaming" \
  "Sliding/parallel-window decoding; per-round latency budget toward < 1 µs."
create_milestone "Phase Q5 — qLDPC Frontier" \
  "Bivariate-bicycle (gross) codes + BP+OSD; the genuinely open frontier."
create_milestone "Phase Q6 — FPGA" \
  "Union-Find on FPGA with measured latency; GPU-vs-FPGA comparison."
create_milestone "Phase Q7 — ASIC (North Star)" \
  "Decoder ASIC architecture, RTL core, tape-out feasibility + customer gate."

milestone_for() {
  case "$1" in
    Q0) echo "Phase Q0 — Experiment Loop Foundation" ;;
    Q1) echo "Phase Q1 — MWPM Decoder" ;;
    Q2) echo "Phase Q2 — Union-Find Decoder" ;;
    Q3) echo "Phase Q3 — GPU Decoder" ;;
    Q4) echo "Phase Q4 — Real-Time / Streaming" ;;
    Q5) echo "Phase Q5 — qLDPC Frontier" ;;
    Q6) echo "Phase Q6 — FPGA" ;;
    Q7) echo "Phase Q7 — ASIC (North Star)" ;;
    *)  echo "" ;;
  esac
}

# ---------------------------------------------------------------------------
# Step 3 — issues, split out of the backlog by issue heading.
# ---------------------------------------------------------------------------
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT
# Split the backlog into one file per issue (portable; macOS csplit lacks -z/{*}).
# Content before the first issue heading is ignored (n==0).
awk -v dir="$TMPDIR" '
  /^### \[Q[0-9]+-[0-9]+\]/ { n++; file=sprintf("%s/issue-%03d.md", dir, n) }
  n>0 { print > file }
' "$BACKLOG"

created=0 skipped=0
for f in "$TMPDIR"/issue-*.md; do
  head -n 1 "$f" | grep -q '^### \[Q[0-9]\+-[0-9]\+\]' || continue

  title_line=$(head -n 1 "$f")
  title="${title_line#"### "}"                                 # strip leading '### ' (quoted: '####' would parse as the '##' operator)
  phase=$(echo "$title" | grep -oE '\[Q[0-9]+-' | grep -oE 'Q[0-9]+')
  milestone=$(milestone_for "$phase")

  if gh issue list --search "in:title \"$title\"" --state all --json title --jq '.[].title' \
       | grep -qxF "$title"; then
    echo "Skip existing: $title"
    skipped=$((skipped+1)); continue
  fi

  labels=$(grep -m1 '^\*\*Labels:\*\*' "$f" | sed 's/\*\*Labels:\*\* *//; s/`//g; s/ //g')
  tail -n +2 "$f" > "$TMPDIR/body.md"

  echo "Create: $title  [$labels]  ($milestone)"
  gh issue create --title "$title" --label "$labels" --milestone "$milestone" \
    --body-file "$TMPDIR/body.md" >/dev/null
  created=$((created+1))
done

echo "Done. Created $created, skipped $skipped."
