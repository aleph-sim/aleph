#!/usr/bin/env bash
# P2-05: prove the rayon-parallel kernels are thread-count invariant on the
# full Tier-1 set (GHZ/QFT/Grover/random) that the `tier1_scaling` bench
# measures. Each parallel kernel writes pairwise-disjoint amplitude blocks
# with no cross-thread reduction, so the result must be bit-identical
# regardless of worker-thread count. We force the parallel path at n=15
# (ALEPH_PAR_MIN_AMPS=0) and run the AoS==SoA equivalence across thread
# counts; a non-identical result at any count fails the 1e-12 assert.
#
# Run from the workspace root. On an AVX-512 host (EPYC) this exercises the
# SIMD kernels; on a non-x86 host it exercises the scalar dispatch and the
# par_blocks driver — still a meaningful invariance proof.
set -euo pipefail

cd "$(dirname "$0")/.."

for t in 1 2 4 8; do
  echo "== RAYON_NUM_THREADS=$t (ALEPH_PAR_MIN_AMPS=0, forced parallel) =="
  RAYON_NUM_THREADS=$t ALEPH_PAR_MIN_AMPS=0 \
    cargo test -p aleph-oracle --test tier1_scaling_invariance --quiet
done

echo
echo "All thread counts (1/2/4/8) agree: Tier-1 parallel kernels are thread-count invariant within 1e-12."
