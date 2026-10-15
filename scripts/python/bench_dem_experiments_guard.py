#!/usr/bin/env python3
"""Paired #565 Criterion runs on a shared Linux host; never stops other jobs.

Artifacts must contain independently built bench-base and bench-candidate binaries.
Uses CPU 3, monitors it and SMT sibling 23; the monitor runs on CPU 39. Override
these arguments for another topology. Rejects >2% preflight busy or >=3% sampled
external CPU overlap. Short tasks/migrations and cache/turbo interference can be
missed: this is bounded shared-host evidence, not exclusive CPU isolation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', type=Path)
    parser.add_argument('--cpu', type=int, default=3)
    parser.add_argument('--sibling', type=int, default=23)
    parser.add_argument('--monitor', type=int, default=39)
    args = parser.parse_args()
    root = args.artifacts.resolve()
    hashes = {v: digest(root / f'bench-{v}') for v in ['base', 'candidate']}
    if hashes['base'] == hashes['candidate']:
        raise ValueError('identical benchmark binaries; verify source/target provenance')
    selected = {args.cpu, args.sibling}
    topology = Path(f'/sys/devices/system/cpu/cpu{args.cpu}/topology/thread_siblings_list').read_text().strip()
    if topology != ','.join(map(str, sorted(selected))):
        raise ValueError(f'check topology: observed siblings {topology}, configured {selected}')
    if args.monitor in selected:
        raise ValueError('monitor must run off the benchmark physical core')
    os.sched_setaffinity(0, {args.monitor})
    ticks_per_second = os.sysconf('SC_CLK_TCK')

    def busy():
        def read():
            result = {}
            for line in Path('/proc/stat').read_text().splitlines():
                cols = line.split()
                if cols[0] in {f'cpu{c}' for c in selected}:
                    values = list(map(int, cols[1:9]))
                    result[int(cols[0][3:])] = (sum(values), values[3] + values[4])
            return result
        before = read()
        time.sleep(1)
        after = read()
        return {c: 1 - (i - before[c][1]) / max(1, t - before[c][0]) for c, (t, i) in after.items()}

    def tasks():
        result = {}
        for process in Path('/proc').glob('[0-9]*'):
            try:
                for task in (process / 'task').iterdir():
                    try:
                        data = (task / 'stat').read_text().rsplit(')', 1)[1].split()
                        result[(int(process.name), int(task.name))] = (int(data[11]) + int(data[12]), int(data[36]))
                    except (OSError, ValueError, IndexError):
                        pass
            except OSError:
                pass
        return result

    cases = ['null_total/1024', 'null_total/131072', 'sample_1024',
             'null_decode_1024', 'score_1024', 'mwpm_d3_total_1024']
    records = []
    # A fresh artifact directory is mandatory: never reuse overwritten Criterion samples.
    events = root / 'guard.jsonl'
    if events.exists():
        raise ValueError('guard.jsonl exists; use a fresh artifact directory')
    for suffix in cases:
        case = 'dem_experiment/' + suffix
        for run in ['base-1', 'candidate-1', 'candidate-2', 'base-2']:
            version = run.rsplit('-', 1)[0]
            for attempt in range(1, 7):
                deadline = time.monotonic() + 60
                while True:
                    preflight = busy()
                    if max(preflight.values()) <= 0.02:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f'no idle preflight for {case}, {run}')
                output = root / f'{run}-{suffix.replace("/", "_")}-{attempt}'
                env = dict(os.environ, RAYON_NUM_THREADS='1', ALEPH_BENCH_OUTPUT=str(output))
                previous = tasks()
                overlap = 0
                start = time.monotonic()
                with output.with_suffix('.log').open('w') as log:
                    child = subprocess.Popen(['taskset', '-c', str(args.cpu), str(root / f'bench-{version}'),
                                              case, '--exact', '--bench', '--noplot'], env=env, stdout=log, stderr=subprocess.STDOUT)
                    while child.poll() is None:
                        time.sleep(0.1)
                        current = tasks()
                        for key, (ticks, cpu) in current.items():
                            if key[0] in {child.pid, os.getpid()} or key not in previous:
                                continue
                            old_ticks, old_cpu = previous[key]
                            if cpu in selected or old_cpu in selected:
                                overlap += max(0, ticks - old_ticks) / ticks_per_second
                        previous = current
                duration = time.monotonic() - start
                accepted = child.returncode == 0 and overlap / duration < 0.03
                event = dict(case=case, run=run, attempt=attempt, accepted=accepted,
                             exit=child.returncode, preflight=preflight, wall_seconds=duration,
                             external_cpu_seconds=overlap, overlap_fraction=overlap / duration,
                             binary_sha256=hashes[version], output=str(output))
                with events.open('a') as f:
                    f.write(json.dumps(event) + '\n')
                if child.returncode:
                    raise RuntimeError(output.with_suffix('.log').read_text()[-3000:])
                if accepted:
                    estimates = list(output.rglob('new/estimates.json'))
                    if len(estimates) != 1:
                        raise ValueError(f'expected one exact benchmark, found {estimates}')
                    new = estimates[0].parent
                    metadata = json.loads((new / 'benchmark.json').read_text())
                    assert metadata['full_id'] == case
                    records.append(dict(case=case, run=run, estimates=json.loads(estimates[0].read_text()),
                                        samples=json.loads((new / 'sample.json').read_text())))
                    print('ACCEPTED', run, case, flush=True)
                    break
            else:
                raise RuntimeError(f'excessive overlap: {case}, {run}')
    assert len(records) == 24
    (root / 'criterion.json').write_text(json.dumps(records, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    main()
