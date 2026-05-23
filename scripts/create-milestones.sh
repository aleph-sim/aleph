#!/usr/bin/env bash
# Create the 7 phase milestones from BACKLOG.md. Idempotent.
# Em-dash (—, U+2014) in titles MUST be preserved verbatim — `gh issue create
# --milestone` matches by exact title string.
set -euo pipefail

REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)

create_milestone() {
  local title="$1"
  local description="$2"

  local existing
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
