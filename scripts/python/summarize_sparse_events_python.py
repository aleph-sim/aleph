#!/usr/bin/env python3
"""Validate isolated-wheel #511 samples and compute all-sample median rates."""
import json
import math
import pathlib
import statistics
import sys


def main():
    root = pathlib.Path(sys.argv[1])
    manifest = json.loads((root / 'manifest.json').read_text())
    assert len(manifest['runs']) == 8
    groups = {(v, d): [] for v in ['main', 'candidate'] for d in [5, 9]}
    hashes = {5: set(), 9: set()}
    rows = []
    for run in manifest['runs']:
        assert run['exit_code'] == 0
        assert all(s['overall'] <= 10 and s['sibling'] <= 5 for s in run['samples']), 'Host-invalid campaign'
        data = json.loads((root / (run['label'] + '.json')).read_text())
        assert data['threads'] == 1
        assert f"{run['variant']}-python" in data['native_module']
        assert [r['distance'] for r in data['rows']] == [5, 9]
        for row in data['rows']:
            d = row['distance']
            assert row['shots'] == {5: 20000, 9: 5000}[d]
            assert row['seed'] == 1 and row['probability'] == .003
            rates = row['samples_shots_per_second']
            assert len(rates) == 7
            assert all(math.isfinite(x) and x > 0 for r in rates for x in r.values())
            groups[run['variant'], d].extend(rates)
            hashes[d].add(row['prediction_sha256'])
            rows.append(dict(label=run['label'], variant=run['variant'], **row))
    assert all(len(h) == 1 for h in hashes.values()), 'predictions changed'
    result = []
    for d in [5, 9]:
        rates = {}
        references = {}
        for v in ['main', 'candidate']:
            samples = groups[v, d]
            assert len(samples) == 28
            rates[v] = statistics.median(r['aleph'] for r in samples)
            references[v] = statistics.median(r['pymatching'] for r in samples)
        result.append(dict(distance=d, shots_per_second=rates, pymatching_shots_per_second=references,
                           current_to_candidate_speedup=rates['candidate'] / rates['main'],
                           candidate_to_pymatching_ratio=rates['candidate'] / references['candidate'],
                           prediction_sha256=next(iter(hashes[d]))))
    (root / 'summary.json').write_text(json.dumps(dict(summary=result, runs=rows), indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
