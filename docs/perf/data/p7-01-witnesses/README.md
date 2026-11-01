# P7-01 validation receipts

See `docs/qec/p7-01-decoder-witnesses.md` for interpretation and reproduction commands.

- `oracle.json`: independent Stim/PyMatching output from `scripts/python/check_decoder_witnesses.py`.
- `release-tests.log`: initial full QEC release suite, including the then-seven witness tests.
- `final-witness-tests.log`: final eight focused release tests after source-validation additions.
- `clippy.log`, `doc-tests.log`: workspace/all-target lint and the public API example.
- `local-timings.json`, `local-criterion-samples.json`, `local-criterion.log`: two ABBA control
  blocks and opt-in costs on a busy local Mac. These do not establish clean-host performance.
- `provenance.json`: baseline, compiler, source fingerprints and limitations.

Small witness JSON and exact `.syn`/v2 `.ref` regression fixtures are checked into
`crates/aleph-qec/tests/data/witness/`. The initial local receipts involved no remote transfer; the separately authorized follow-up is below.

## Authorized Linux follow-up

- `remote-campaign.json`: all accepted runs, preflight rejections, CPU guards, feature costs,
  hardware/compiler metadata and binary hashes. All control attempts and executed feature runs are retained; one feature preflight skip has no serialized CPU readings.
- `remote-samples.json`, `remote-criterion.log`: underlying Criterion samples and outputs.
- `remote-summary.json`: paired four-block geometric ratios and descriptive bootstrap ranges;
  regenerate with `python3 docs/perf/data/p7-01-witnesses/summarize.py`.
- `remote-provenance.json`: exact tested head/baseline, matching source fingerprints, transferred
  archive/patch hashes and environment scope.
- `remote-witness-tests.log`, `remote-experiment-tests.log`, `remote-oracle.json`,
  `remote-example.log`: Linux correctness receipts.
- `campaign.py`: exact runner, including bounded core-overlap gates. Set ROOT to a new temporary
  directory containing separately compiled `target-base` and `target-patch` snapshots to rerun.

The pinned-server follow-up supersedes the pending performance item. The local campaign remains
inconclusive and is retained unchanged. No services or CI jobs were stopped.
