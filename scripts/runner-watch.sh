#!/usr/bin/env bash
# runner-watch.sh — detect a wedged self-hosted GitHub Actions runner.
#
# The failure we keep hitting: the self-hosted EPYC runner keeps its heartbeat
# (GitHub shows it `online`, `busy=false`) but its job dispatcher stops claiming
# work, so CI jobs that target `self-hosted` sit `queued` forever. A healthy idle
# runner claims a matching queued job within seconds — minutes of "idle runner +
# stale queued self-hosted job" is the smoking gun that it's hung and needs a
# `systemctl restart actions.runner.*` on the box.
#
# This script flags exactly that, plus the plain `offline` case. It needs only
# `gh` (authenticated); all time math runs in gh's embedded jq (`now` /
# `fromdateiso8601`), so there's no GNU/BSD `date` portability mess.
#
# Usage:
#   scripts/runner-watch.sh                 # one-shot check, prints status, sets exit code
#   scripts/runner-watch.sh --watch 60      # re-check every 60s until interrupted
#   scripts/runner-watch.sh --notify        # also fire a macOS desktop notification on alert
#   STALE_MIN=3 REPO=owner/name scripts/runner-watch.sh
#
# Exit codes (one-shot mode): 0 = healthy/idle-no-backlog, 1 = WEDGED, 2 = OFFLINE.
# In --watch mode the loop runs until you Ctrl-C; it notifies on each transition.
set -euo pipefail

REPO="${REPO:-aleph-sim/aleph}"
# A self-hosted job queued longer than this (minutes) while the runner is online
# and idle is treated as a wedge. 5 min is far above normal dispatch latency
# (seconds) yet well under the 30-min job timeout.
STALE_MIN="${STALE_MIN:-5}"
RUNNER_LABEL="${RUNNER_LABEL:-self-hosted}"

WATCH_INTERVAL=""
NOTIFY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --watch) WATCH_INTERVAL="${2:?--watch needs an interval in seconds}"; shift 2 ;;
    --notify) NOTIFY=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 64 ;;
  esac
done

notify() {
  # Best-effort desktop ping on macOS; silently no-op elsewhere.
  [ "$NOTIFY" -eq 1 ] || return 0
  local msg="$1"
  if command -v osascript >/dev/null 2>&1; then
    osascript -e "display notification \"${msg//\"/\'}\" with title \"aleph runner-watch\"" >/dev/null 2>&1 || true
  fi
  printf '\a' >&2  # terminal bell
}

# One check. Echoes a human status line; returns 0 healthy, 1 wedged, 2 offline.
check_once() {
  # --- self-hosted runners: are any online, and is at least one idle? ---
  # One line per self-hosted runner: "<name> <status> <busy>".
  local runners
  runners="$(gh api "repos/$REPO/actions/runners" --paginate \
    --jq ".runners[] | select([.labels[].name] | index(\"$RUNNER_LABEL\")) | \"\(.name) \(.status) \(.busy)\"" \
    2>/dev/null || true)"

  if [ -z "$runners" ]; then
    echo "UNKNOWN: no $RUNNER_LABEL runner registered on $REPO (or gh unauthenticated)"
    return 2
  fi

  local any_online=0 any_idle=0
  while read -r _name status busy; do
    [ "$status" = "online" ] && any_online=1
    [ "$status" = "online" ] && [ "$busy" = "false" ] && any_idle=1
  done <<<"$runners"

  # --- stale queued self-hosted jobs across all active runs ---
  # Collect run ids that are queued or in_progress, then inspect their jobs.
  local run_ids
  run_ids="$( { gh api "repos/$REPO/actions/runs?status=queued&per_page=50"  --jq '.workflow_runs[].id' 2>/dev/null
                gh api "repos/$REPO/actions/runs?status=in_progress&per_page=50" --jq '.workflow_runs[].id' 2>/dev/null
              } | sort -u)"

  local max_age=0 stale_count=0 worst=""
  local rid line age name
  for rid in $run_ids; do
    # Per queued self-hosted job: "<age_minutes> <job name>".
    while IFS=$'\t' read -r age name; do
      [ -z "${age:-}" ] && continue
      if [ "$age" -ge "$STALE_MIN" ]; then
        stale_count=$((stale_count + 1))
        if [ "$age" -gt "$max_age" ]; then max_age="$age"; worst="$name"; fi
      fi
    done < <(gh api "repos/$REPO/actions/runs/$rid/jobs" \
              --jq ".jobs[] | select(.status==\"queued\") | select([.labels[]] | index(\"$RUNNER_LABEL\")) | \"\((now - (.created_at|fromdateiso8601))/60|floor)\t\(.name)\"" \
              2>/dev/null || true)
  done

  # --- verdict ---
  if [ "$any_online" -eq 0 ]; then
    if [ "$stale_count" -gt 0 ]; then
      echo "OFFLINE: runner offline AND $stale_count self-hosted job(s) queued (worst ${max_age}m: $worst) — start the runner on the box"
      return 2
    fi
    echo "OFFLINE: runner offline, but no jobs waiting"
    return 2
  fi

  if [ "$stale_count" -gt 0 ] && [ "$any_idle" -eq 1 ]; then
    echo "WEDGED: runner online+idle but $stale_count self-hosted job(s) stuck ≥${STALE_MIN}m (worst ${max_age}m: $worst) — restart actions.runner on the EPYC box"
    return 1
  fi

  if [ "$stale_count" -gt 0 ]; then
    echo "BUSY: $stale_count self-hosted job(s) queued ${max_age}m but runner is busy — likely just backlog, watch it"
    return 0
  fi

  echo "OK: runner online, no stale self-hosted backlog"
  return 0
}

run_and_report() {
  local out rc
  set +e
  out="$(check_once)"; rc=$?
  set -e
  echo "[$(date +%H:%M:%S)] $out"
  if [ "$rc" -ne 0 ]; then notify "$out"; fi
  return "$rc"
}

if [ -z "$WATCH_INTERVAL" ]; then
  run_and_report
  exit $?
fi

# --watch mode: loop forever, notify on every alert transition.
prev=""
while true; do
  set +e
  out="$(check_once)"; rc=$?
  set -e
  echo "[$(date +%H:%M:%S)] $out"
  state="${out%%:*}"
  if [ "$state" != "$prev" ] && [ "$rc" -ne 0 ]; then notify "$out"; fi
  prev="$state"
  sleep "$WATCH_INTERVAL"
done
