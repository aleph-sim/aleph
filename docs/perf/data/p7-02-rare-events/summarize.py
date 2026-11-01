"""Derive the compact summary from the preserved fixed campaign; no filtering of runs."""
import json
from pathlib import Path
from statistics import median

root = Path(__file__).resolve().parent
raw = json.loads((root / 'campaign.json').read_text())
cases = []
for case in raw['cases']:
    runs = case['runs']
    strata = [r['stratified'] for r in runs]
    direct = [r['direct'] for r in runs]
    cases.append(dict(
        case=case['case'], calls=strata[0]['decoder_calls'],
        strat_width=median(r['failure_probability']['upper'] - r['failure_probability']['lower'] for r in strata),
        direct_width=median(r['upper'] - r['lower'] for r in direct),
        median_width_ratio=median(r['interval_width_ratio_stratified_over_direct'] for r in runs),
        strat_ms=1000 * median(r['elapsed_seconds'] for r in strata),
        direct_ms=1000 * median(r['seconds'] for r in direct),
        direct_errors=sum(r['errors'] for r in direct),
        status_counts={s: sum(r['status'] == s for r in strata) for s in sorted({r['status'] for r in strata})},
        tail_upper=strata[0]['unsampled_mass']['upper']))
coverage = {name: {k: v for k, v in result.items() if k != 'intervals'}
            for name, result in raw['coverage'].items()}
(root / 'summary.json').write_text(json.dumps(dict(cases=cases, coverage=coverage), indent=2) + '\n')
