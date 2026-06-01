#!/usr/bin/env bash
# P2-03 NUMA benchmark — three memory-placement policies on a multi-node host.
#
# Run from the workspace root on the 2-socket Xeon Silver 4114 (or any host
# with >1 NUMA node). Measures the same high-qubit scaling workload under:
#   1. baseline    — default allocator, all pages faulted onto node 0
#   2. interleave  — numactl --interleave=all (round-robin pages)
#   3. first-touch — --features aleph-sv/numa (parallel first-touch init)
# plus an optional Tier-2 pinned run. Expected ordering:
#   first-touch >= interleave > baseline.
#
# The scaling bench (`qft_scaling`, package `aleph-benches`) lives behind the
# `scaling-bench` opt-in feature and sweeps thread counts internally, so it
# exercises both sockets. AVX-512 needs `target-cpu=native` on the Xeon.
#
# Per CLAUDE.md: confirm the box is IDLE first (uptime ~0, no competing
# `cargo bench`) — a CI race on the shared runner silently inflates baselines.
set -euo pipefail

PKG="aleph-benches"
BENCH="${BENCH:-qft_scaling}"
export RUSTFLAGS="-C target-cpu=native"

echo "=== idle check (CLAUDE.md: must be ~0 load, no competing bench) ==="
uptime
pgrep -af 'cargo bench|bencher run|Runner.Worker' || echo "(clean)"

echo; echo "=== NUMA topology (expect >1 node) ==="
numactl --hardware | sed -n '1,6p'

echo; echo "### Policy 1: baseline (default allocator → all pages on node 0)"
cargo bench -p "$PKG" --features scaling-bench --bench "$BENCH"

echo; echo "### Policy 2: interleave (numactl --interleave=all, round-robin pages)"
numactl --interleave=all \
    cargo bench -p "$PKG" --features scaling-bench --bench "$BENCH"

echo; echo "### Policy 3: first-touch, Tier 1 (parallel init, no pinning)"
cargo bench -p "$PKG" --features scaling-bench,aleph-sv/numa --bench "$BENCH"

echo; echo "### Policy 3b: first-touch, Tier 2 (+ external per-node pinning)"
echo "# Pin the process to node-local CPUs+memory so each worker's contiguous"
echo "# chunk stays local. Adjust the node list to this box's \`numactl -H\`:"
echo "#   numactl --cpunodebind=0,1 --localalloc \\"
echo "#     cargo bench -p $PKG --features scaling-bench,aleph-sv/numa --bench $BENCH"
