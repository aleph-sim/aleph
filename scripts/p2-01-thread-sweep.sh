#!/usr/bin/env bash
# P2-01: prove the rayon-parallel gate kernels are thread-count invariant.
#
# Each parallel kernel writes pairwise-disjoint amplitude blocks with no
# cross-thread floating-point reduction, so the result must be bit-identical
# regardless of how many worker threads rayon uses. We force the parallel
# path at small n (ALEPH_PAR_MIN_AMPS=0) and run, across thread counts:
#   * the SoA == AoS == Naive oracle workhorse (1e-12 equivalence), and
#   * the full aleph-sv unit/integration suite (per-kernel oracle anchors
#     and indexing-coverage tests from P1-05..P1-08).
# A non-identical result at any thread count would fail one of these.
#
# Run from the workspace root. On an AVX-512 host (the EPYC bench server)
# this exercises the SIMD kernels; on a non-x86 host it exercises the
# scalar dispatch and the par_blocks driver itself — still meaningful.
set -euo pipefail

cd "$(dirname "$0")/.."

for t in 1 2 4 8; do
  echo "== RAYON_NUM_THREADS=$t (ALEPH_PAR_MIN_AMPS=0, forced parallel) =="
  RAYON_NUM_THREADS=$t ALEPH_PAR_MIN_AMPS=0 \
    cargo test -p aleph-sv --quiet
  RAYON_NUM_THREADS=$t ALEPH_PAR_MIN_AMPS=0 \
    cargo test -p aleph-oracle --test soa_vs_naive --quiet
done

echo
echo "All thread counts (1/2/4/8) agree: parallel kernels are thread-count invariant within 1e-12."
