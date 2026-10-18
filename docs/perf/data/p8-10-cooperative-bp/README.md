# P8-10 receipts

Report: [warp-cooperative BP](../../p8-10-cooperative-bp.md). Base main:
`e9f61b6923cef985d9ff7178a2f2a6f59e978e27` (P8-09).

## Artifacts and source identity

- `protocol.json`: primary gate/matrix and supplementary-trace distinction.
- `aleph-560-campaign.tar.gz`: all primary CSVs, logs, CPU mask oracles, binary hashes,
  exact arguments and 200 ms GPU telemetry. Four mirrored blocks; no deleted samples.
- `aleph-560-supplementary.tar.gz`: all varied-input trace receipts/oracles/telemetry,
  15 matched Nsight profiles with arguments/binary hash and initial/passing test logs.
- `aleph-560-measured-sources.tar.gz`: exact primary timing/profiling source snapshot.
  Reconstruct by overlaying it on `git archive` of the base main above.
- `original-probe.patch.gz`: apply to the base main to build the original control.
  This adds the same probe/dev dependencies; original kernel/driver remain unchanged.
- `validation-logs.tar.gz`: final-source GPU tests, example compilation, Clippy,
  documentation test and local stable/beta lint logs, plus Cargo dependency metadata.
- `source-fingerprints.json`: final source SHA256 and measured snapshot SHA256.
- `analysis.json`, `trace-analysis.json`, `profile-analysis.json`: independent summaries.
- `SHA256SUMS`: committed artifacts and summaries, excluding this checksum file itself.

Primary/profiling binaries precede the final CSR ownership validator, extra pure/debug
safety tests and supplementary `--trace` probe support. Constructor validation is outside
the timed endpoint. The final CUDA kernel differs from the measured kernel only in comments;
its executable body is unchanged. Trace uses the validator and extended probe. Final-source
GPU unit tests and lint were rerun. Do not relabel archived measurements as a build of the
final commit: manifests bind the actual binaries and this archive preserves their source.
The final test seed formatting change preserves its numeric value.

Large oracle receipts establish CPU/GPU correction equality. Full hard-decision/convergence
checks are a separate native fixture suite. CPU flags in timing oracles are not public GPU
diagnostics. The original failed expanded fixture assumed one input word per shot; Gross
has multiple words, so its prefix check was fixed. The initial Clippy failure was a new
hexadecimal grouping lint; both failed and passing logs are included for audit.

## Reproduce or independently check

From repository root on Linux with a real CUDA GPU:

```sh
cargo build --release -p aleph-cuda --features cuda --example bp_cooperative_probe
ALEPH_REQUIRE_CUDA=1 cargo test --release -p aleph-cuda --features cuda --test qec_bp_oracle
ALEPH_REQUIRE_CUDA=1 cargo test --release -p aleph-cuda --features cuda --lib qec::bp::tests
cargo test -p aleph-cuda --features cuda --doc
python3 scripts/python/bench_cooperative_bp.py /path/to/prototype /path/to/original /tmp/new-primary
python3 scripts/python/trace_cooperative_bp.py /path/to/prototype /tmp/new-trace
python3 scripts/python/profile_cooperative_bp.py /path/to/prototype /tmp/new-primary /tmp/new-profiles
```

The primary script requires the original and prototype **binary paths**; inspect its
usage for the explicit arguments. Fresh output directories prevent accidental mixing.
Constructor/JIT/packing/allocation are outside the warm timing scope. Profiles use
`--clock-control none --cache-control all`, one kernel after two warmups; no clock/service
changes. They are counter evidence, not acceptance timings.

To check committed receipts without CUDA:

```sh
mkdir -p /tmp/p8-10-receipts
for file in docs/perf/data/p8-10-cooperative-bp/aleph-560-{campaign,supplementary}.tar.gz; do
  tar -xzf "$file" -C /tmp/p8-10-receipts
done
python3 scripts/python/check_cooperative_bp.py /tmp/p8-10-receipts/aleph-560-campaign
python3 scripts/python/check_cooperative_trace.py /tmp/p8-10-receipts/aleph-560-trace
python3 scripts/python/check_cooperative_profiles.py /tmp/p8-10-receipts/aleph-560-profiles
```

The trace checker validates every returned mask, metadata and buffer payload. It rejects
an altered CPU mask (negative control). Percentiles use nearest rank; pooled repeats
are not independent physical fault draws or population-tail confidence bounds.
