# P7-02 rare-event receipts

`campaign.json` retains all 64 fixed runs and both 2000-seed coverage designs. `summary.json`
is derived; `oracle.json` contains independent Decimal/SciPy/Stim/PyMatching validation.
`campaign-load.json` and `criterion-load.json` retain host trajectories and affinity protocol;
`criterion-samples.json`/`criterion-estimates.json` retain raw measurements.

The original measured source (before the additive automatic dispatcher) is reproduced by
applying `measured-source.patch.gz` to baseline 1790f094a2f9fff9645f6a07038ea7eedb716595.
`provenance.json` records exact source, archive, patch and executable hashes and environment.
`run-campaign.py` and `run-criterion.py` are the actual runners; temporary paths identify the
original authorized remote run. Change those roots for reproduction. No services were stopped.

The package example reproduces fixed case/seed/budget definitions:

```sh
RAYON_NUM_THREADS=1 cargo run --release -p aleph-qec --example rare_events -- OUTDIR 2000 16
python scripts/python/check_rare_event_bounds.py OUTDIR/summary.json
cargo bench -p aleph-qec --bench rare_events
```

Use the matching source revision for exact receipt reproduction. Newer example output also
includes automatic decisions, independently of the original explicit-method comparisons.
The score is a heuristic; four representative cases do not establish universal routing quality.
Timing is descriptive on a shared host, and no general speedup is claimed. The report explains
the default PyMatching importer policy discrepancy rather than treating it as agreement.

`automatic.json` contains the final-source supplemental routing results and hashes for the
changed source files; `auto-*-tests.log` verifies selected-method equality on Mac and Linux.
The original `provenance.json`/`measured-source.patch.gz` remain tied to the pre-dispatcher
measurement campaign, rather than relabeling those measurements as final-source timings.
