#!/usr/bin/env bash
# Assemble the Zenodo deposit for the preprint: campaign CSVs + appliance-v1 release assets + reports.
# Usage (from paper/zenodo/): ./bundle.sh   -> writes out/ and out/SHA256SUMS; upload out/* to Zenodo
# together with .zenodo.json, then paste the minted DOI into paper/sections/07-reproducibility.tex,
# README.md and hw/product/README.md.
set -euo pipefail
cd "$(dirname "$0")"; ROOT="$(git rev-parse --show-toplevel)"
rm -rf out && mkdir -p out/data out/appliance-v1 out/reports
cp "$ROOT"/docs/perf/data/qec-q5-*.csv "$ROOT"/docs/perf/data/qec-q7-*.csv out/data/
cp "$ROOT"/docs/perf/qec-q7-fixed-bp.md "$ROOT"/docs/perf/q7-02-*.md "$ROOT"/docs/qec/q7-06-ac1-batched-dma.md \
   "$ROOT"/docs/qec/q7-07-nonconvergence-policy.md "$ROOT"/docs/qec/asic-architecture.md out/reports/
gh release download appliance-v1 --repo aleph-sim/aleph --dir out/appliance-v1 --clobber
{
  echo "Zenodo deposit for the preprint (aleph-sim/aleph, tag preprint-v1)."; echo
  echo "data/      campaign CSVs copied from docs/perf/data/ (message-width sweep qec-q7-fixed-bp.csv,"
  echo "           schedule budget qec-q7-budget.csv, non-convergence qec-q7-nonconv-*.csv, Q5 software baselines)"
  echo "reports/   the dated Markdown reports every number in the paper is copied from"
  echo "appliance-v1/  KV260 bitstream (.bit), hardware handoff (.hwh), Python driver, golden vectors, SHA256SUMS"
} > out/MANIFEST.txt
(cd out && find . -type f ! -name SHA256SUMS -exec shasum -a 256 {} \; | sort -k2 > SHA256SUMS)
du -sh out; wc -l out/SHA256SUMS
