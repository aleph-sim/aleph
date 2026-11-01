# P8-14 receipts

Report: [resident overlap paging](../../p8-14-resident-overlap.md).
Base main: `ab3aed8027132f42f9df4e9ddc614403da5a3147`.

## Evidence and exact source

- `protocol.json`: pre-hardware gate, training/held-out cells, equal-payload larger
  synchronous tiles, FP64/resource scope and fixed sampling policy.
- `aleph-564-campaign.tar.gz`: all 160 primary samples, stderr modeled counters,
  exact arguments/binary hash, manifest and 200 ms GPU telemetry. No outliers removed.
- `aleph-564-measured-sources.tar.gz`: source/protocol used for primary measurements.
  Reproduce by overlaying it on `git archive` of the base main above. Production
  `paged.rs` and the benchmark probe are byte-identical to final source.
- `aleph-564-supplementary.tar.gz`: 16 post-primary per-gate synchronous controls,
  9 Nsight Systems `.nsys-rep` + SQLite timelines, profiler arguments/binary hash,
  environment log and native test/lint/doc logs. No profile duration selects adoption.
- `local-validation.tar.gz`: local stable/beta workspace Clippy and examples logs.
- `analysis.json`, `profile-analysis.json`: independent checked summaries.
- `source-fingerprints.json`: final and measured production/probe/test SHA256.
- `SHA256SUMS`: artifact/summary fingerprints (excluding this checksum file itself).

The measured snapshot has the earlier oracle test fixture. Final oracle coverage adds
SWAP/CCX matrix scratch and a host validation error after enqueue, with backend reuse;
it passed on the same final production source. Audit logs retain two test-only compile
fixes: `GateInstance` references require pointer-order comparison rather than PartialEq,
and the unsupported-k fixture needs the gate's shared matrix-container type. Neither
changed timed production/probe code. Final CUDA all-target lint and oracle logs passed.

## Reproduce

Linux with a real CUDA device; no GPU-less skip counts as a hardware pass:

```sh
cargo build --release -p aleph-cuda --features cuda --example paged_resident_probe
ALEPH_REQUIRE_CUDA=1 cargo test --release -p aleph-cuda --features cuda --test paged_resident_overlap_oracle
ALEPH_REQUIRE_CUDA=1 cargo test --release -p aleph-cuda --features cuda --test paged_overlap_oracle --test paged_batched_oracle
cargo test -p aleph-cuda --features cuda --lib sv::paged::tests
cargo test -p aleph-cuda --features cuda --doc
python3 scripts/python/bench_resident_overlap.py /path/to/paged_resident_probe /tmp/new-primary
# Run after primary timing, not concurrently; needs `nsys` on PATH:
python3 scripts/python/profile_resident_overlap.py /path/to/paged_resident_probe /tmp/new-profiles
```

All commands use fresh output directories. API timings include pinned/ring allocations,
initialization, execute/drain; exclude backend compilation and norm/export. Host state
is ≤1 GiB and amplitude ring ≤1 GiB for these forced-paging cells. This measures real
paging on states which fit in-core, not out-of-core reach. Services/clocks stayed unchanged.

Check committed receipts without CUDA:

```sh
mkdir -p /tmp/p8-14-receipts
for name in campaign supplementary; do
  tar -xzf "docs/perf/data/p8-14-resident-overlap/aleph-564-$name.tar.gz" -C /tmp/p8-14-receipts
done
python3 scripts/python/check_resident_overlap.py /tmp/p8-14-receipts/aleph-564-campaign
python3 scripts/python/check_resident_profiles.py /tmp/p8-14-receipts/aleph-564-profiles
```

The first checker verifies every timing row/identity/norm and independently derives
modeled pass/byte counts. A deliberately inconsistent transfer receipt was rejected.
The second verifies actual timeline copy totals and kernel counts, computes interval
unions/intersections and retains API synchronization counts. Setup includes allocation,
host initialization and stream construction, not pure allocator latency. Copy/kernel
busy times overlap and cannot be summed into wall time. Four paired blocks are descriptive
evidence, not confidence intervals; repeated deterministic circuits are not a workload
population sample.
