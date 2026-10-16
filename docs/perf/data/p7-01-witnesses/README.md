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
`crates/aleph-qec/tests/data/witness/`. No remote snapshot was transferred for these receipts.
