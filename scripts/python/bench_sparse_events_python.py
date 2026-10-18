#!/usr/bin/env python3
"""Two fixed ABBA blocks of the unchanged #511 public Python fixtures."""
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

from bench_sparse_events import busy, stat


def main():
    config_path, validation_path, directory = sys.argv[1:]
    config = json.loads(pathlib.Path(config_path).read_text())
    validation = pathlib.Path(validation_path)
    root = pathlib.Path(directory)
    root.mkdir()
    cpu, sibling = config['cpu'], config['sibling']
    assert pathlib.Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list').read_text().strip() == f'{cpu},{sibling}'
    python = config['python']
    script = pathlib.Path(__file__).resolve().with_name('bench_mwpm_allocations.py')
    manifest = dict(started_unix=time.time(), runs=[], helper_sha256=hashlib.sha256(script.read_bytes()).hexdigest())
    for block in range(2):
        for position, variant in enumerate(['main', 'candidate', 'candidate', 'main']):
            label = f'b{block}-i{position}-{variant}'
            preflights = []
            for attempt in range(120):
                before = stat()
                time.sleep(2)
                after = stat()
                row = dict(time=time.time(), overall=busy(before, after, 'cpu'), cpu=busy(before, after, f'cpu{cpu}'), sibling=busy(before, after, f'cpu{sibling}'))
                preflights.append(row)
                if row['overall'] < 10 and max(row['cpu'], row['sibling']) < 1.1:
                    break
                time.sleep(3)
            else:
                raise RuntimeError('No quiet core/sibling; services preserved')
            env = dict(os.environ, PYTHONPATH=str(validation / (variant + '-python')) + ':' + str(script.parent), RAYON_NUM_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            argv = ['taskset', '-c', str(cpu), python, str(script), '--repeats', '7']
            samples = []
            print('START', label, flush=True)
            with (root / (label + '.json')).open('w') as output, (root / (label + '.stderr')).open('w') as error:
                child = subprocess.Popen(argv, env=env, stdout=output, stderr=error)
                previous = stat()
                while child.poll() is None:
                    time.sleep(1)
                    current = stat()
                    samples.append(dict(time=time.time(), overall=busy(previous, current, 'cpu'), sibling=busy(previous, current, f'cpu{sibling}')))
                    previous = current
            manifest['runs'].append(dict(label=label, block=block, variant=variant, argv=argv, preflights=preflights, samples=samples, exit_code=child.returncode))
            (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
            assert child.returncode == 0, label
            assert max(s['overall'] for s in samples) <= 10 and max(s['sibling'] for s in samples) <= 5, 'Whole campaign invalid: host activity; raw observations preserved'
            result = json.loads((root / (label + '.json')).read_text())
            assert str(validation / (variant + '-python')) in result['native_module']
            assert [r['distance'] for r in result['rows']] == [5, 9]
            assert all(len(r['samples_shots_per_second']) == 7 for r in result['rows'])
            print('PASS', label, flush=True)
    manifest['finished_unix'] = time.time()
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('PASS python', flush=True)


if __name__ == '__main__':
    main()
