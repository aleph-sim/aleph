# NUMA-aware allocation (P2-03)

On a multi-node NUMA machine the default allocator faults the entire state
vector onto the node of the *allocating* thread (node 0). Worker threads on
other nodes then pay the remote-access penalty (~2.1× on the reference Xeon,
`node distances` 10/21) for their share of every gate sweep.

The `numa` feature replaces the buffer's lazy zero-fill with a **first-touch**
parallel init: [`AlignedBuf::zeroed_first_touch`] allocates uninitialised, then
zeroes the buffer from rayon's global pool in contiguous, page-aligned chunks,
so the worker that faults a page is — under matched partitioning + pinning —
the one that later computes on it. Pages thus distribute across nodes instead
of piling onto node 0.

## Enabling

The feature lives in `aleph-core` and is forwarded by `aleph-sv`:

```bash
# Build / test the SV backends on the first-touch path
cargo build --release -p aleph-sv --features numa
cargo test            -p aleph-sv --features numa

# Run the scaling bench on the first-touch path (note the cross-crate feature)
cargo bench -p aleph-benches --features scaling-bench,aleph-sv/numa --bench qft_scaling
```

Off by default: without the feature, `AlignedBuf` is dependency-free (no rayon)
and its codegen is byte-for-byte unchanged.

## Locality contract

First-touch only yields *true* locality when both hold:

1. **OS policy is first-touch** (the Linux default) — i.e. NOT running under
   `numactl --interleave`.
2. **Workers are pinned** so a worker's contiguous chunk stays on the node it
   faulted. Without pinning, first-touch still spreads pages across nodes
   (≈ balanced, like interleave) and already beats the all-on-node-0 default,
   but it will not exceed interleave until pinning is in place. Pin via
   `numactl --cpunodebind=… --localalloc` (see `scripts/numa-bench.sh`).

## Fallback: interleave (no feature needed)

```bash
numactl --interleave=all cargo run --release -p aleph-cli -- run circuit.qasm
```

Round-robins pages across nodes: no locality, but balanced bandwidth and
robust to bad partitioning. The zero-code default for unknown topologies.

## Benchmark methodology

`scripts/numa-bench.sh` measures three placement policies on the same machine
and circuit (`qft_scaling`, a high-qubit workload that sweeps the full state,
where remote access bites hardest):

1. **Baseline** — default allocator, all pages on node 0.
2. **Interleave** — `numactl --interleave=all`.
3. **First-touch** — `--features aleph-sv/numa` (Tier 1 unpinned; Tier 2 + pinning).

Expected ordering: **first-touch ≥ interleave > baseline**.

## Results — 2-socket Intel Xeon Silver 4114 (2× 10C/20T, 2 NUMA nodes)

> **Pending measurement on the Xeon (P2-03 Task 9).** Both current bench boxes
> (EPYC 8124P, Ryzen 9 3900) are single NUMA node and cannot exhibit a NUMA
> effect; the EPYC reports `available: 1 nodes (0)`. The table below is filled
> from the real 2-socket run before the PR merges.

| Workload | Baseline (node 0) | Interleave | First-touch (Tier 1) | First-touch + pin (Tier 2) |
|----------|------------------:|-----------:|---------------------:|---------------------------:|
| _TBD_    | _TBD_             | _TBD_      | _TBD_                | _TBD_                      |

[`AlignedBuf::zeroed_first_touch`]: ../crates/aleph-core/src/aligned.rs
