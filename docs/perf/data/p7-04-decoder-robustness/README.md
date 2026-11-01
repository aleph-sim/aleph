# P7-04 reproducibility receipts

`campaign.json` retains all 36 fixed surface/Gross-BB cases and the tiny exact-oracle case,
with full model identities, process laws, stream tallies, costs, bounds and 31 witnesses.
`cases.csv`/`summary.json` are compact views; no cases or unfavorable outcomes were filtered.
`coverage.json` retains every final interval for 1000 fixed seeds per process and finite-time
exact truth curves. `oracle.json`/`coverage-oracle.json` retain independent checks.

`static-curves.*` and `dynamic-intervals.*` are reproducible scientific figures from
`scripts/python/plot_decoder_robustness.py`. The Python scripts require NumPy/SciPy/Stim/
PyMatching; figures also require Matplotlib. No packages were installed on the shared host.

`provenance.json` records baseline, exact patch/source hashes and executable fingerprints.
Apply `aleph-546.patch.gz` (decompressed) to baseline 879e5d299d0bea1338b72e07e129ab5d4b453ef9
for the original measured campaign. `aleph-546-final.patch.gz` reproduces final native tests,
coverage, benchmark configuration and oracle/plot scripts. The native algorithm is unchanged
between stages; the final source has an added causality test, offline coverage example,
Criterion output path, scripts and a clarified call-count comment. Source fingerprints were
verified on the remote host before collecting receipts.

`run-campaign.py` and `run-criterion.py` are actual runners and retain the original temporary
paths; change roots to reproduce. Both select CPU3 and monitor sibling23. Shared services and
CI remain running; timings are descriptive, with about 1.2% external CPU interference recorded.
Criterion samples, estimates and load trajectories are retained without excluding outliers.

Local complete release suite: 257 tests +9 doc tests, 22 existing ignores, before adding the
final causality-prefix test. Final focused Linux checks cover 7 robustness, 8 legacy witness,
and 10 stratification tests; the new 7-test robustness suite also passes on Mac. Final workspace
beta Clippy/fmt pass (Cargo beta prints existing aleph-py manifest warnings).
All 31 exported Linux witnesses pass native CLI replay on Linux and Mac. Existing v1/v2 golden
bytes stay covered by witness/stratification suites. The known default PyMatching parallel-edge
merge-policy discrepancy is explicitly documented in the report and oracle.

There is no familywise confidence claim across the sweep, no fitted prior refresh and no
speedup claim. At 64 streams, Hoeffding intervals are intentionally too wide to establish
most dynamic mismatch effects; zero observed failures or differences are not proof of zero risk.
