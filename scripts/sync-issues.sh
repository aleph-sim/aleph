#!/usr/bin/env bash
# Create GitHub issues from BACKLOG.md.
# Idempotent: existing issues (matched by [Pn-nn] prefix in title) are skipped.
# Portable across BSD (macOS) and GNU userland — uses awk for splitting, not csplit.
set -euo pipefail

BACKLOG="${BACKLOG:-BACKLOG.md}"
if [ ! -f "$BACKLOG" ]; then
  echo "ERROR: $BACKLOG not found. Run from repo root." >&2
  exit 1
fi

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

# Split BACKLOG.md by '### [Pn-nn]' headings into per-issue files.
# Each output file: $TMPDIR/issue-<ID>.md
awk -v outdir="$TMPDIR" '
  /^### \[P[0-9]+(\.[0-9]+)?-[0-9]+\]/ {
    match($0, /\[P[0-9]+(\.[0-9]+)?-[0-9]+\]/);
    id = substr($0, RSTART+1, RLENGTH-2);
    out = outdir "/issue-" id ".md";
    capture = 1;
    print > out;
    next;
  }
  capture { print >> out; }
' "$BACKLOG"

# Cache existing titles once (single API hit instead of one per issue).
echo "Fetching existing issues…"
existing_titles=$(gh issue list --state all --limit 500 --json title --jq '.[].title' || true)

phase_milestone() {
  case "$1" in
    0) echo "Phase 0 — Foundation" ;;
    1) echo "Phase 1 — Single-Thread CPU Optimization" ;;
    2) echo "Phase 2 — Multi-Thread CPU" ;;
    3) echo "Phase 3 — Alternative Backends" ;;
    4) echo "Phase 4 — Algorithm Benchmarks & v0.1 Release" ;;
    4.5) echo "Phase 4.5 — CPU Parity" ;;
    4.6) echo "Phase 4.6 — CPU Depth" ;;
    5) echo "Phase 5 — GPU Backend" ;;
    5.5) echo "Phase 5.5 — Apple/Metal GPU" ;;
    6) echo "Phase 6 — Multi-GPU & Distributed" ;;
    *) echo "" ;;
  esac
}

created=0
skipped=0
failed=0

for f in "$TMPDIR"/issue-*.md; do
  title_line=$(head -n 1 "$f")
  # "### [P0-01] Foo" -> "[P0-01] Foo". Use quoted pattern so bash parses
  # the operator as `#` + literal `### `, not `##` + `## `.
  title="${title_line#"### "}"
  id=$(printf '%s' "$title" | grep -oE 'P[0-9]+(\.[0-9]+)?-[0-9]+' | head -1)

  if printf '%s\n' "$existing_titles" | grep -qF "[$id]"; then
    echo "skip   [$id] (already exists)"
    skipped=$((skipped + 1))
    continue
  fi

  labels=$(grep -m 1 '^\*\*Labels:\*\*' "$f" \
    | sed -e 's/^\*\*Labels:\*\* *//' -e 's/`//g' -e 's/ //g' || true)
  if [ -z "$labels" ]; then
    echo "WARN   [$id] no labels parsed" >&2
  fi

  phase_num=$(grep -m 1 '^\*\*Milestone:\*\*' "$f" \
    | grep -oE 'Phase [0-9]+(\.[0-9]+)?' | head -1 | awk '{print $2}' || true)
  milestone=$(phase_milestone "$phase_num")
  if [ -z "$milestone" ]; then
    echo "WARN   [$id] could not resolve milestone (phase='$phase_num')" >&2
  fi

  # Body = everything after the title line.
  tail -n +2 "$f" > "$TMPDIR/body.md"

  echo "create [$id] $title"
  if gh issue create \
      --title "$title" \
      --label "$labels" \
      --milestone "$milestone" \
      --body-file "$TMPDIR/body.md" >/dev/null; then
    created=$((created + 1))
  else
    echo "ERROR  [$id] gh issue create failed" >&2
    failed=$((failed + 1))
  fi
done

echo
echo "Summary: created=$created  skipped=$skipped  failed=$failed"
[ "$failed" -eq 0 ]
