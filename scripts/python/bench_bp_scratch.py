#!/usr/bin/env python3
"""P8-09 balanced same-decoder layout experiment; no result-based retries or best-of.
Usage: bench_bp_scratch.py BINARY ORIGINAL_BASELINE NEW_OUTPUT_DIRECTORY
GPU services/clocks are left unchanged. Timed Rust regions include upload/kernel/download,
exclude packing, allocation warmup, compilation and process startup. Raw rows are retained.
"""
import csv
import hashlib
import io
import json
import pathlib
import statistics
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    binary, baseline, directory = sys.argv[1:]
    root = pathlib.Path(directory)
    root.mkdir()
    cells = [(d, n, p) for d in (3, 7, 11) for n in (1, 128, 4096) for p in (.001, .03)]
    cells += [(7, 4096, .15), (11, 4096, .15), (7, 33, .03), (7, 4097, .03), (3, 16384, .03), (7, 16384, .03)]
    manifest = dict(protocol='four fixed mirrored three-arm blocks; rotate leading arm by block; 3 timed repeats per invocation, one untimed allocation warmup; all observations retained', binary_sha256=digest(binary), original_baseline_sha256=digest(baseline), cells=cells, started_unix=time.time(), measurements=[])
    with (root/'gpu-monitor.csv').open('w') as monitor:
        process = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,memory.used,clocks.sm,power.draw,temperature.gpu', '--format=csv,noheader,nounits', '--loop-ms=200'], stdout=monitor, stderr=subprocess.STDOUT)
        try:
            checksums = {}
            def measure(executable, d, shots, rate, layout, label):
                argv = [executable, str(d), str(shots), str(rate), '3', layout]
                result = subprocess.run(argv, text=True, capture_output=True, check=True)
                (root/(label+'.csv')).write_text(result.stdout)
                (root/(label+'.stderr')).write_text(result.stderr)
                rows = list(csv.DictReader(io.StringIO(result.stdout)))
                assert len(rows) == 3
                key = (d, shots, rate)
                checksum = rows[0]['checksum']
                assert all(r['checksum'] == checksum for r in rows)
                assert checksums.setdefault(key, checksum) == checksum, (key, layout)
                manifest['measurements'].append(dict(label=label, argv=argv, rows=rows))
            for block in range(4):
                arms = ['shot','edge','warp']
                arms = arms[block % 3:] + arms[:block % 3]
                order = arms + list(reversed(arms))
                cases = cells if block % 2 == 0 else list(reversed(cells))
                for d, n, p in cases:
                    for position, arm in enumerate(order):
                        measure(binary,d,n,p,arm,f'b{block}-d{d}-n{n}-p{p}-i{position}-{arm}')
            # Compiler-preserved shot-major control vs the truly unchanged original kernel.
            for block in range(4):
                for d,n,p in [(3,128,.03),(7,4096,.03),(11,4096,.03)]:
                    for position,arm in enumerate(['original','shot','shot','original']):
                        measure(baseline if arm=='original' else binary,d,n,p,'shot',f'control-b{block}-d{d}-i{position}-{arm}')
        finally:
            process.terminate()
            process.wait()
            manifest['finished_unix'] = time.time()
            (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    summary=[]
    for d,n,p in cells:
        rows=[m for m in manifest['measurements'] if not m['label'].startswith('control') and m['argv'][1:4]==[str(d),str(n),str(p)]]
        medians={arm:statistics.median(int(r['ns']) for m in rows if m['argv'][-1]==arm for r in m['rows']) for arm in ['shot','edge','warp']}
        ratios={arm:medians['shot']/medians[arm] for arm in ['edge','warp']}
        summary.append(dict(distance=d,shots=n,rate=p,median_ns=medians,speedup=ratios))
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
