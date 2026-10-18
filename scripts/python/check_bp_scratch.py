#!/usr/bin/env python3
"""Independently check every retained timing row and summarize four paired blocks.
Usage: check_bp_scratch.py CAMPAIGN_DIRECTORY
Ranges are min/max block ratios, not confidence intervals. No decoder is reimplemented.
"""
import csv
import json
import pathlib
import statistics
import sys


def check(root):
    manifest = json.loads((root/'manifest.json').read_text())
    records = manifest['measurements']
    assert len(records) == 624 and len(manifest['cells']) == 24
    assert len({r['label'] for r in records}) == len(records)
    checksums = {}
    for record in records:
        with (root/(record['label']+'.csv')).open() as data:
            assert list(csv.DictReader(data)) == record['rows'], record['label']
        assert len(record['rows']) == 3
        assert (root/(record['label']+'.stderr')).exists()
        key = tuple(record['argv'][1:4])
        for i, row in enumerate(record['rows']):
            assert row['repeat'] == str(i) and int(row['ns']) > 0
            assert (row['distance'],row['shots'],row['rate']) == key
            assert checksums.setdefault(key,row['checksum']) == row['checksum']
    cells = []
    for d,n,p in manifest['cells']:
        key = [str(d),str(n),str(p)]
        rows = [r for r in records if not r['label'].startswith('control') and r['argv'][1:4] == key]
        medians = {}
        ratios = {arm:[] for arm in ['edge','warp']}
        for arm in ['shot','edge','warp']:
            values = [int(v['ns']) for r in rows if r['argv'][-1] == arm for v in r['rows']]
            assert len(values) == 24
            medians[arm] = statistics.median(values)
        for block in range(4):
            block_medians = {}
            for arm in ['shot','edge','warp']:
                observations = [int(v['ns']) for r in rows if r['label'].startswith(f'b{block}-') and r['argv'][-1] == arm for v in r['rows']]
                assert len(observations) == 6
                block_medians[arm] = statistics.median(observations)
            for arm in ratios:
                ratios[arm].append(block_medians['shot']/block_medians[arm])
        cells.append(dict(distance=d,shots=n,rate=p,median_ns=medians,
                          speedup={arm:medians['shot']/medians[arm] for arm in ratios},
                          block_speedup=ratios))
    controls = []
    for d in [3,7,11]:
        ratios = []
        for block in range(4):
            groups = {}
            for arm in ['original','shot']:
                values = [int(v['ns']) for r in records if r['label'].startswith(f'control-b{block}-d{d}-') and r['label'].endswith(arm) for v in r['rows']]
                assert len(values) == 6
                groups[arm] = statistics.median(values)
            ratios.append(groups['original']/groups['shot'])
        controls.append(dict(distance=d,original_over_compiled_reference=ratios))
    return dict(status='PASS',invocations=624,timed_observations=1872,
                cells=cells,original_controls=controls,
                scope='four shared-host block ratios; no statistical confidence/speed guarantee; checksums establish observed correction consistency, native CPU tests validate decoding separately')


if __name__ == '__main__':
    print(json.dumps(check(pathlib.Path(sys.argv[1])),indent=2))
