#!/usr/bin/env python3
"""Validate and summarize the frozen #511 Criterion campaign; retain per-run CIs."""
import json
import math
import pathlib
import re
import statistics
import sys


def case_coverage(samples, case_ids):
    """Match observed warmup episodes to the ordered cases, including crossing intervals."""
    counts = [0] * len(case_ids)
    current = -1
    previous = 'setup'
    active_phases = {'Warming up', 'Collecting'}
    for sample in samples:
        start, end = sample['phase_start'], sample['phase_end']
        assert start == previous
        touched = set()
        if start in active_phases:
            assert current >= 0
            touched.add(current)
        if end == 'Warming up' and start != 'Warming up':
            current += 1
            assert current < len(case_ids)
        if end in active_phases:
            assert current >= 0
            touched.add(current)
        assert bool(touched) == sample['timing_active'], 'Missing/ambiguous phase episode'
        for case in touched:
            counts[case] += 1
        previous = end
    assert current + 1 == len(case_ids), 'Missing observed warmup episode'
    assert all(n >= 2 for n in counts), 'Missing per-case host coverage'
    return dict(zip(case_ids, counts))


def main():
    root = pathlib.Path(sys.argv[1])
    manifest = json.loads((root / 'manifest.json').read_text())
    ablation = manifest['mode'] != 'primary'
    blocks = 2 if ablation else 4
    variants = {'ablation':['main','neighbor','candidate'], 'guard_ablation':['candidate','queued'], 'primary':['original','main','candidate']}[manifest['mode']]
    expected = [11] if ablation else [7, 9, 11, 13, 15, 17]
    runs = manifest['runs']
    assert len(runs) == blocks * 2 * len(variants)
    rows = []
    coverage = {}
    seen = set()
    for run in runs:
        assert run['exit_code'] == 0
        if manifest['mode'] != 'ablation':
            active = [s for s in run['samples'] if s['timing_active']]
            assert len(active) >= 2 * len(expected)
            assert all(s['overall'] <= 10 and s['sibling'] <= 5 for s in active), 'Host-invalid campaign'
            log = (root / (run['label'] + '.log')).read_text()
            cases = re.findall(r'Benchmarking (.+?): Warming up', log)
            assert [int(re.search(r'/d(\d+)_', case).group(1)) for case in cases] == expected
            coverage[run['label']] = case_coverage(run['samples'], cases)
        assert run['label'] not in seen
        seen.add(run['label'])
        files = list(pathlib.Path(run['criterion_home']).glob('**/new/estimates.json'))
        assert len(files) == len(expected), (run['label'], files)
        found = []
        for file in files:
            meta = json.loads((file.parent / 'benchmark.json').read_text())
            distance = int(meta['value_str'].split('_')[0].removeprefix('d'))
            assert meta['group_id'] == 'mwpm_decode' and meta['function_id'] == 'sparse'
            found.append(distance)
            estimate = json.loads(file.read_text())['mean']
            sample = json.loads((file.parent / 'sample.json').read_text())
            assert len(sample['iters']) == len(sample['times']) >= 10
            assert all(math.isfinite(x) and x > 0 for x in sample['iters'] + sample['times'])
            point = estimate['point_estimate'] / 256 / 1000
            ci = estimate['confidence_interval']
            assert 0 < ci['lower_bound'] <= estimate['point_estimate'] <= ci['upper_bound']
            rows.append(dict(label=run['label'], variant=run['variant'], block=run['block'], distance=distance,
                             us_per_shot=point, samples=len(sample['iters']), mean_ns=estimate))
        assert sorted(found) == expected
    summaries = []
    for distance in expected:
        values = {}
        by_block = {}
        for variant in variants:
            group = [r for r in rows if r['distance'] == distance and r['variant'] == variant]
            assert len(group) == blocks * 2
            values[variant] = statistics.median(r['us_per_shot'] for r in group)
            by_block[variant] = []
            for block in range(blocks):
                group_block = [r['us_per_shot'] for r in group if r['block'] == block]
                assert len(group_block) == 2
                by_block[variant].append(statistics.median(group_block))
        ratios = {v: values[v] / values['candidate'] for v in variants if v != 'candidate'}
        paired = {v: [a / b for a, b in zip(by_block[v], by_block['candidate'])] for v in ratios}
        summaries.append(dict(distance=distance, median_us=values, candidate_speedup_vs=ratios, block_ratios=paired))
    result = dict(summary_rule='Median of process Criterion mean point estimates; no pooled CI.', summaries=summaries, per_case_active_host_intervals=coverage, runs=rows)
    (root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
