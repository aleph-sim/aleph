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

Measured 2026-06-01 on a 2-socket Xeon Silver 4114 (2 NUMA nodes, `node
distances` 10/21 ≈ 2.1× remote penalty, 40 logical CPUs, 123 GB). Workload:
`qft_scaling` (QFT through the AoS+AVX-512 `NaiveSvBackend`, `target-cpu=native`)
at `RAYON_NUM_THREADS=40` so both sockets are fully engaged. Criterion
`sample_size = 10`; each policy compared to the baseline via `--baseline`.
Allocation happens inside the timed path, so first-touch's upfront parallel-zero
cost is included in these numbers, not hidden.

The EPYC 8124P and Ryzen 9 3900 bench boxes are single NUMA node and cannot
exhibit any NUMA effect (the EPYC reports `available: 1 nodes (0)`); this is why
the measurement runs on the 2-socket Xeon.

The box was verified idle before this run — in particular `cat /proc/mdstat`
showed no RAID resync (an earlier run was discarded because the NVMe RAID1 was
mid-resync at ~205 MB/s, contaminating a bandwidth-bound benchmark).

| Workload (n=25, 512 MiB) | Median time | vs baseline |
|--------------------------|------------:|------------:|
| Baseline (default alloc, all pages on node 0) | 5.496 s | — |
| Interleave (`numactl --interleave=all`)       | 3.747 s | **−31.8 %** (1.47×) |
| First-touch, Tier 1 (no pinning)              | 3.426 s | **−37.7 %** (1.60×) |
| First-touch, Tier 2 (`numactl --localalloc`)  | 3.432 s | **−37.6 %** (1.60×) |

| Workload (n=22, 64 MiB) | Median time | vs baseline |
|-------------------------|------------:|------------:|
| Baseline                | 471.0 ms | — |
| Interleave              | 386.3 ms | −18.0 % |
| First-touch, Tier 1     | 336.5 ms | −28.6 % |
| First-touch, Tier 2     | 340.7 ms | −27.7 % |

All non-baseline changes are statistically significant (criterion `p = 0.00 <
0.05`). The ordering **first-touch ≥ interleave > baseline** holds, and notably
first-touch **beats interleave even without pinning** (Tier 1, −37.7 % vs
−31.8 % at n=25): the parallel init already spreads pages across both nodes *and*
biases each page toward the worker that faults it. External `--localalloc`
pinning (Tier 2) adds nothing beyond Tier 1 (−37.6 % vs −37.7 %, within noise),
so in-process per-worker affinity was not needed (kept out of scope — see P2-03
design §5). The effect grows with state size (−37.7 % at 512 MiB vs −28.6 % at
64 MiB), consistent with the bandwidth-wall analysis (ADR 0008): the larger the
state, the more of each gate sweep hits remote memory under the node-0 baseline.

[`AlignedBuf::zeroed_first_touch`]: ../crates/aleph-core/src/aligned.rs
