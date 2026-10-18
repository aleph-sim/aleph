# P7-00 integration receipts

Base: `41aae16f3784cda0061e3248081060c0cde2dc48` (main after #590).
The base plus `snapshot-provenance.json` identifies the actual integration source;
the base hash alone does not include the patch. `validation.json` additionally hashes
the transferred patch, original baseline archive and root build/toolchain config.

- `aleph-542-results.tar.gz`: complete Linux nine-entry-point smoke, exact Cargo JSON
  artifact selections, all binary/source/artifact hashes, 16 command logs, saved
  studies and all small legacy-example outputs. Initial executable builds are all
  `fresh=false`, using a new target directory.
- `mac-studies.tar.gz`: both full deterministic studies generated on macOS arm64.
- `cross-platform.json`: independently validated stable projection of both runs;
  counts, witness identities/outcomes, stream data and query answers agree.
- `aleph-542-environment.log`: Linux CPU/toolchain and load receipt. This was an
  authorized CPU run on the shared 40-logical-CPU Xeon Silver 4114 machine; no
  services/CI were stopped and no GPU was used.
- `aleph-542-tests.log`: both new CI integration tests passed on Linux.
- `stable-clippy.log`, `beta-clippy.log`: workspace all-target lint checks.
- `mac-tests.log`: complete affected-crate release tests and rustdoc tests.
- `validation.json`: root hashes, Mac toolchain and independent checker negative
  controls. Deliberately changing a saved QASM or witness fault set was rejected.

These are correctness receipts, not a new performance or statistical coverage campaign.
Native replay and deletion sweeps check witness semantics; the independent Python
checker verifies artifact identities, analytic controls and binomial intervals. It
is not an independent decoder implementation. Prior large leaf campaigns are not
copied here; use the primary reports linked from the phase exit report.

Recheck both archived studies from the repository root:

```sh
mkdir /tmp/p7-linux-replay /tmp/p7-mac-replay
tar -xzf docs/perf/data/p7-integration/aleph-542-results.tar.gz -C /tmp/p7-linux-replay
tar -xzf docs/perf/data/p7-integration/mac-studies.tar.gz -C /tmp/p7-mac-replay
python3 scripts/python/check_research_instruments.py /tmp/p7-linux-replay/aleph-542-results --compare /tmp/p7-mac-replay
```

These commands validate archived research outputs without needing either original
build directory. Rebuilding/running all nine endpoints is documented in the report.

Loose test logs normalize terminal blank lines; archived command outputs remain raw.
