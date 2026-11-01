# P8-09: measured GPU BP scratch layouts

Base revision: `1de1494a3180e784817ac42039fb501e746465f6`.

- `campaign.tar.gz`: all 624 invocation CSV/stderr files, 1872 timed observations,
  24 cells, original-kernel controls, GPU telemetry and binary hashes in its manifest.
- `aleph-559-receipts.tar.gz`: raw Nsight CSV/stderr, exact profiler argv manifests,
  environment and actual-GPU oracle/lint logs, including final host validation.
- `campaign-analysis.json`: independent CSV/manifest/checksum validation and four-block ratios.
- `profile-analysis.json`: values extracted from the matched `cache-control=all` profiles.
  Exploratory `cache-control=none` data remain in the archive and are not mixed in.
- `iteration-analysis.json`: separate exact iteration-count diagnostic and checks against
  original correction checksums. All three layouts have identical per-shot iteration counts.
- `iteration-instrumentation.py`: exact temporary diagnostic transformation. This is
  restricted to these single-observable fixtures: it packs iteration count into high
  output bits, keeps the logical correction in low bits, and is never used for timing.
  It changes no production sources or production output schema.
- `original-probe.patch.gz`: only the bounded probe added to the unchanged base kernel.
- `measured-candidate.patch.gz`: exact three-layout snapshot used in the fixed campaign.
  It includes native oracle tests; apply to a fresh base snapshot, not the current tree.
- `validation.json`: source/patch fingerprints and test boundaries. Final production CUDA
  equals measured CUDA. The later host delta preserves legacy extra-word prefix behavior
  and expands tests/comments; final device validation was rerun.
- `mac-stable-clippy.log`, `mac-beta-clippy.log`: full workspace lint receipts on macOS.
- `linux-initial-dependency-clippy.log`: retained pre-existing QEC dependency lint from
  server Rust 1.96. Final scoped CUDA all-target `--no-deps` lint passed; unrelated QEC
  code was not edited. Loose logs normalize terminal blank lines.

Raw archives retain their original output. Profiler durations are not campaign timings.
Services were not stopped, clocks were not changed and no isolated-system claim is made.
No result-based repetitions/outlier deletion occurred. Subsequent diagnostic profiles
explain the low-rate regression; they do not replace the fixed timing endpoints.

Validate the archived campaign without GPU or original build directories:

```sh
mkdir /tmp/p8-09-replay
tar -xzf docs/perf/data/p8-09-bp-scratch/campaign.tar.gz -C /tmp/p8-09-replay
python3 scripts/python/check_bp_scratch.py /tmp/p8-09-replay/aleph-559-campaign
```

Reconstruct the exact measured sources in two new temporary directories (from repo root):

```sh
mkdir /tmp/p8-09-original /tmp/p8-09-candidate
git archive 1de1494a3180e784817ac42039fb501e746465f6 -o /tmp/p8-09-base.tar
tar -xf /tmp/p8-09-base.tar -C /tmp/p8-09-original
tar -xf /tmp/p8-09-base.tar -C /tmp/p8-09-candidate
gzip -dc docs/perf/data/p8-09-bp-scratch/original-probe.patch.gz | git -C /tmp/p8-09-original apply -
gzip -dc docs/perf/data/p8-09-bp-scratch/measured-candidate.patch.gz | git -C /tmp/p8-09-candidate apply -
```

Build each with `cargo build --release -p aleph-cuda --features cuda --example bp_layout_probe`
in its source directory, using **separate fresh** `CARGO_TARGET_DIR` paths. Then pass the
candidate binary, original binary and a new output directory to
`scripts/python/bench_bp_scratch.py`. Run only on an authorized suitable GPU host;
this is a fixed several-minute experiment, not correctness CI. Retained source/Rand lock
and seed 559 reconstruct all per-cell Tanner graphs and physical-mechanism draws.
The original probe accepts the trailing `shot` selector used by the experiment; its
kernel and host library are unchanged from base.
