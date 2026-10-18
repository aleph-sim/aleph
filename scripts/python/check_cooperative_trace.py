#!/usr/bin/env python3
"""Validate supplementary P8-10 varied-input tails. Usage: TRACE_DIRECTORY.
Percentiles describe 64 fixed physical-noise inputs per cell, replayed eight times
per arm. Repeats are not independent noise samples or a population-tail guarantee.
"""
import csv
import json
import pathlib
import statistics
import sys
from check_cooperative_bp import quantile


def check(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    records = manifest['measurements']
    assert manifest['supplementary'] and manifest['unique_inputs_per_cell'] == 64
    assert len(records) == 144 and len({r['label'] for r in records}) == 144
    assert manifest['cells'] == [[m, p] for m in ['surface7', 'surface11', 'gross'] for p in [.001, .03]]
    cells = []
    for model, rate in manifest['cells']:
        oracle = json.loads((root / 'oracles' / f'{model}-p{rate}.json').read_text())
        assert (oracle['shots'], oracle['seed'], oracle['max_iter'], oracle['alpha']) == (64, 560, 64, .875)
        for key in ['masks', 'converged', 'logical_failure']:
            assert len(oracle[key]) == 64
        assert all(isinstance(x, bool) for key in ['converged', 'logical_failure'] for x in oracle[key])
        assert all(isinstance(x, int) and 0 <= x < (1 << 64) for x in oracle['masks'])
        assert oracle['invalid'] == sum(not x for x in oracle['converged'])
        assert oracle['logical_failures'] == sum(oracle['logical_failure'])
        for key in ['dem_sha256', 'input_sha256']:
            assert len(oracle[key]) == 64 and int(oracle[key], 16) >= 0
        selected = [r for r in records if r['argv'][1:4] == [model, '64', str(rate)]]
        assert len(selected) == 24
        for r in selected:
            assert r['argv'][-1] == '--trace'
            with (root / (r['label'] + '.csv')).open() as f:
                assert list(csv.DictReader(f)) == r['rows']
            mode = r['argv'][5]
            metadata = dict(w.split('=', 1) for w in (root / (r['label'] + '.log')).read_text().split() if '=' in w)
            for key in ['dem_sha256', 'input_sha256']:
                assert metadata[key] == oracle[key]
            for key in ['invalid', 'logical_failures']:
                assert int(metadata[key]) == oracle[key]
            assert int(metadata['oracle_shots']) == 64
            payload = 16 * int(metadata['edges']) + int(metadata['vars']) + int(metadata['checks']) + 8 + 4 * ((int(metadata['checks']) + 31) // 32)
            assert int(metadata['per_shot_bytes']) == payload
            assert int(metadata['retained_scratch_bytes']) == payload * (32 if mode == 'warp' else 1)
            assert len(r['rows']) == 64
            for i, row in enumerate(r['rows']):
                assert (row['model'], row['shots'], row['rate'], row['mode'], row['repeat']) == (model, '1', str(rate), mode, str(i))
                assert int(row['ns']) > 0 and int(row['checksum']) == oracle['masks'][i]
        stats = {}
        for mode in ['shot', 'warp', 'coop']:
            values = [int(t['ns']) for r in selected if r['argv'][5] == mode for t in r['rows']]
            assert len(values) == 512
            stats[mode] = dict(median_ns=statistics.median(values), p95_ns=quantile(values, .95), p99_ns=quantile(values, .99), max_ns=max(values))
        block_ratios = []
        for block in range(4):
            values = {mode: [int(t['ns']) for r in selected if r['label'].startswith(f'b{block}-') and r['argv'][5] == mode for t in r['rows']] for mode in stats}
            assert all(len(v) == 128 for v in values.values())
            block_ratios.append(min(statistics.median(values[m]) for m in ['shot', 'warp']) / statistics.median(values['coop']))
        cells.append(dict(model=model, rate=rate, stats=stats, block_median_speedup=block_ratios, invalid=oracle['invalid'], logical_failures=oracle['logical_failures']))
    return dict(status='PASS', invocations=144, timed_calls=9216, unique_inputs=384, cells=cells, scope='Supplementary fixed 64-input distribution per cell; repeated inputs are not independent trials; does not replace primary go gate.')


if __name__ == '__main__':
    print(json.dumps(check(pathlib.Path(sys.argv[1])), indent=2))
