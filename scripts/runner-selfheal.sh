#!/usr/bin/env bash
# runner-selfheal.sh — auto-restart the self-hosted Actions runner when wedged.
#
# Deployed to the EPYC runner box (sd-185281) at
# /usr/local/bin/aleph-runner-selfheal.sh and driven by root cron every 5 min:
#   */5 * * * * /usr/local/bin/aleph-runner-selfheal.sh >/dev/null 2>&1
# This copy is the versioned source of truth; redeploy from here.
#
# Why it exists: the runner periodically keeps its heartbeat (GitHub shows it
# `online`, `busy=false`) but its Listener stops claiming jobs, so CI jobs sit
# `queued` for an hour with no signal (seen 2026-06-20 after a Bench run was
# cancelled mid-flight; six jobs stuck 75+ min, fixed by `systemctl restart`).
#
# Wedge = self-hosted CI job(s) queued > STALE_MIN on GitHub while THIS box has
# no Runner.Worker process (runner is idle but not claiming work). Restarting an
# idle runner cannot interrupt a real job, so the no-Worker guard makes the
# restart safe. "What's queued" comes from the GitHub REST API with a
# fine-grained PAT (Actions: read-only) in TOKEN_FILE; "idle" is detected
# locally via pgrep, so the token never needs admin/runner scopes.
#
# Companion to scripts/runner-watch.sh (the detect-only, exit-code variant meant
# to run interactively or under /loop on a workstation).
set -euo pipefail

REPO="aleph-sim/aleph"
RUNNER_LABEL="self-hosted"
STALE_MIN="${STALE_MIN:-8}"                          # queued longer than this ⇒ suspect
SERVICE="actions.runner.ruslan-splynx-aleph.aleph-linux-x64.service"
TOKEN_FILE="/etc/aleph-runner-watch.token"
LOG="/var/log/aleph-runner-selfheal.log"
STAMP="/run/aleph-runner-selfheal.last-restart"
MIN_RESTART_GAP_MIN="${MIN_RESTART_GAP_MIN:-15}"     # never restart more often than this
DRY_RUN="${DRY_RUN:-0}"

log(){ echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" >>"$LOG" 2>/dev/null || true; }

# 1. A running Worker means the runner is healthy and busy with a real job — bail.
if pgrep -f "Runner.Worker" >/dev/null 2>&1; then exit 0; fi

# 2. Need a token to ask GitHub what's queued. Silent until it's installed.
[ -r "$TOKEN_FILE" ] || exit 0
TOKEN="$(cat "$TOKEN_FILE" 2>/dev/null || true)"
[ -n "$TOKEN" ] || exit 0

api(){ curl -fsS --max-time 25 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/$1"; }

# 3. Active run ids (queued + in_progress).
run_ids="$( { api "repos/$REPO/actions/runs?status=queued&per_page=50"      | jq -r '.workflow_runs[].id'
             api "repos/$REPO/actions/runs?status=in_progress&per_page=50"  | jq -r '.workflow_runs[].id'
           } 2>/dev/null | sort -u )" || { log "API error listing runs"; exit 0; }

# 4. Count stale queued self-hosted jobs (age computed in jq via now/fromdateiso8601).
stale=0; worst=0
for rid in $run_ids; do
  ages="$(api "repos/$REPO/actions/runs/$rid/jobs" 2>/dev/null \
    | jq -r --arg L "$RUNNER_LABEL" \
      '.jobs[] | select(.status=="queued") | select([.labels[]]|index($L)) | ((now - (.created_at|fromdateiso8601))/60|floor)')" || continue
  for a in $ages; do
    [ -z "$a" ] && continue
    if [ "$a" -ge "$STALE_MIN" ]; then stale=$((stale+1)); [ "$a" -gt "$worst" ] && worst="$a"; fi
  done
done

[ "$stale" -eq 0 ] && exit 0   # nothing stuck ⇒ healthy, exit quietly

# 5. Stale queued jobs + no local Worker = wedged. Rate-limit restarts.
now_s="$(date +%s)"
if [ -f "$STAMP" ]; then
  last="$(cat "$STAMP" 2>/dev/null || echo 0)"
  gap=$(( (now_s - last) / 60 ))
  if [ "$gap" -lt "$MIN_RESTART_GAP_MIN" ]; then
    log "WEDGED (stale=$stale worst=${worst}m) but last restart ${gap}m ago (<${MIN_RESTART_GAP_MIN}m); waiting"
    exit 0
  fi
fi

if [ "$DRY_RUN" = "1" ]; then
  log "DRY_RUN: would restart $SERVICE (stale=$stale worst=${worst}m, no Worker)"; exit 0
fi

log "WEDGED: $stale self-hosted job(s) queued (worst ${worst}m), no Worker → restarting $SERVICE"
if systemctl restart "$SERVICE"; then echo "$now_s" >"$STAMP"; log "restart issued OK"; else log "restart FAILED rc=$?"; fi
