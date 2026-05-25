#!/usr/bin/env bash
# Regenerate oracle/fixtures/*.json from oracle/circuits/*.qasm.
#
# Idempotent except for the `generated_at` timestamp inside each
# fixture. After running, commit any non-timestamp changes; if only
# timestamps moved, `git diff --stat` will show modifications but the
# numerical content is unchanged.
set -euo pipefail
cd "$(dirname "$0")/../oracle"
uv run python gen.py
