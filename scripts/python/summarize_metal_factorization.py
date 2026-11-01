"""Summarize the frozen P8-11 matrix campaign; never infer fused MPS speedup."""
import argparse
import collections
import json
import math
import pathlib
import statistics


def summarize(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row['rows'], row['cols'], row['fixture'], row['candidate'])].append(row)
    expected = {(2*c, 2*c) for c in [32, 64, 128, 256, 512]}
    expected |= {(2*c, c) for c in [32, 64, 128, 256, 512]}
    expected |= {(c, 2*c) for c in [32, 64, 128, 256, 512]}
    assert set((m, n) for m, n, _, _ in groups) == expected, 'incomplete shape sweep'
    assert len(groups) == 120 and all(len(v) == 2 for v in groups.values()), 'incomplete paired repeats'
    metrics = ['pack_ms', 'factor_ms', 'finalize_guard_ms', 'sort_install_ms',
               'cpu_factor_ms', 'cpu_split_ms', 'qr_ms', 'cpu_qr_ms',
               'reconstruction', 'retained_isometry', 'discarded_error',
               'norm_error', 'observable_error', 'qr_reconstruction', 'qr_isometry']
    cells = []
    for (m, n, fixture, candidate), data in sorted(groups.items()):
        cell = dict(rows=m, cols=n, fixture=fixture, candidate=candidate,
                    all_numerical_accept=all(x['numerical_accept'] for x in data),
                    guard_rejections=sum(not x['guard_accept'] for x in data),
                    sweeps=[x['sweeps'] for x in data], dispatches=[x['dispatches'] for x in data])
        for metric in metrics:
            values = [x[metric] for x in data]
            assert all(isinstance(x, (int, float)) and math.isfinite(x) for x in values), metric
            cell[metric] = dict(median=statistics.median(values), minimum=min(values), maximum=max(values))
        # Intrusive independent-command stage sum is a research budget, NOT
        # measured latency of production's fused command buffer or an MPS circuit.
        budgets = [x['pack_ms'] + x['factor_ms'] + x['finalize_guard_ms'] + x['fallback_ms'] for x in data]
        cell['instrumented_stage_sum_ms'] = statistics.median(budgets)
        cells.append(cell)
    regimes = []
    for m, n in sorted(expected):
        data = [x for x in cells if x['rows'] == m and x['cols'] == n]
        eligible = max(m, n)//2 >= 128
        numerical = all(x['all_numerical_accept'] for x in data if x['candidate'])
        ratios = []
        for fixture in ['well', 'clustered', 'ill', 'rank_deficient']:
            ref = next(x for x in data if x['fixture'] == fixture and not x['candidate'])
            new = next(x for x in data if x['fixture'] == fixture and x['candidate'])
            ratios.append(min(ref['instrumented_stage_sum_ms'], new['cpu_split_ms']['median']) / new['instrumented_stage_sum_ms'])
        regimes.append(dict(rows=m, cols=n, eligible=eligible, all_fixtures_numerical=numerical,
                            minimum_budget_speedup=min(ratios),
                            passes_necessary_gate=eligible and numerical and min(ratios) >= 1.25))
    return dict(receipt_count=len(rows), fixture_count=60, cells=cells, regimes=regimes,
                guard_rejections=sum(not x['guard_accept'] for x in rows),
                numerical_rejections=dict(reference=sum(not x['numerical_accept'] for x in rows if not x['candidate']),
                                          candidate=sum(not x['numerical_accept'] for x in rows if x['candidate'])),
                verdict='NEEDS_END_TO_END_VALIDATION' if any(x['passes_necessary_gate'] for x in regimes) else 'NO_GO',
                limitation='Instrumented stage sums are necessary gates only; no fused production or end-to-end improvement is established.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=pathlib.Path)
    parser.add_argument('output', type=pathlib.Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    result = summarize(rows)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(result['verdict'], result['receipt_count'], 'receipts', result['numerical_rejections'])
