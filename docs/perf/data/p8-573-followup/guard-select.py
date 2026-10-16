#!/usr/bin/env python3
"""Guard paired noisy-driver or fused-gate Criterion runs on a shared Linux CPU host.

The fused suite uses RAYON_NUM_THREADS=1; configure affinity and monitored SMT
siblings for the selected core. For noisy runs, run select_noisy_bench_cores.py
first. The artifact directory must contain
bench-base and bench-candidate executables compiled in independent targets.
Before each run all selected cores/SMT siblings must be <=2% busy. During the
run external CPU overlap on active benchmark cores must be <3% of nominal
worker CPU capacity. This is bounded shared-host evidence, not exclusive CPU
isolation; do not advertise small gains from this protocol. No other process
has its affinity changed or is stopped.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
import pathlib
import subprocess
import time

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', type=pathlib.Path)
    parser.add_argument('--suite', choices=['noisy', 'fused'], default='noisy')
    parser.add_argument('--case', action='append', help='Exact suite case to run (repeatable); default: every case')
    args = parser.parse_args()
    root = args.artifacts
    hashes = {}
    for version in ['base', 'candidate']:
        with (root / f'bench-{version}').open('rb') as binary:
            digest = hashlib.sha256()
            for chunk in iter(lambda: binary.read(1024 * 1024), b''):
                digest.update(chunk)
            hashes[version] = digest.hexdigest()
    if hashes['base'] == hashes['candidate']:
        raise ValueError('baseline and candidate binaries are identical; verify build provenance')
    (root / 'guard-binary-sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    lock = (root / '.guard.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    policy = json.loads((root / 'core-policy.json').read_text())
    selected = set(policy['monitored_smt_siblings'])
    sibling_map = {}
    for cpu in selected:
        path = pathlib.Path('/sys/devices/system/cpu') / f'cpu{cpu}' / 'topology'
        core = (path / 'core_id').read_text()
        socket = (path / 'physical_package_id').read_text()
        sibling_map[cpu] = {other for other in selected if (pathlib.Path('/sys/devices/system/cpu') / f'cpu{other}' / 'topology/core_id').read_text() == core and (pathlib.Path('/sys/devices/system/cpu') / f'cpu{other}' / 'topology/physical_package_id').read_text() == socket}
    os.sched_setaffinity(0, {policy['monitor_cpu']})
    clock_ticks = os.sysconf('SC_CLK_TCK')
    (root / 'clean-events.jsonl').touch()
    history = [json.loads(line) for line in (root / 'clean-events.jsonl').read_text().splitlines() if line]
    accepted = {(e['case'], e['run']) for e in history if e['accepted']}

    def tasks():
        values = {}
        for pid in pathlib.Path('/proc').iterdir():
            if not pid.name.isdigit() or int(pid.name) == os.getpid():
                continue
            try:
                for task in (pid / 'task').iterdir():
                    try:
                        raw = (task / 'stat').read_text()
                        data = raw.rsplit(')', 1)[1].split()
                        values[int(pid.name), int(task.name)] = (int(data[11]) + int(data[12]), int(data[36]), raw.split('(', 1)[1].rsplit(')', 1)[0], data[0])
                    except (OSError, ValueError, IndexError):
                        pass
            except OSError:
                pass
        return values

    def overlap(before, after, bench_pid):
        """Charge sampled external CPU time to active benchmark physical cores.

 Runnable endpoints, CPU migrations and SMT siblings are included conservatively.
 Brief tasks/migrations and shared cache/turbo effects can still be missed.
 """
        active = set()
        for (key, (ticks, cpu, name, state)) in after.items():
            if key[0] != bench_pid:
                continue
            old = before.get(key, (0, cpu, name, state))
            if ticks > old[0] or state == 'R' or old[3] == 'R':
                active.update(sibling_map.get(cpu, set()))
                active.update(sibling_map.get(old[1], set()))
        for (key, (_, cpu, _, state)) in before.items():
            if key[0] == bench_pid and state == 'R':
                active.update(sibling_map.get(cpu, set()))
        hits = []
        for (key, (ticks, cpu, name, state)) in after.items():
            if key[0] == bench_pid:
                continue
            if key in before:
                (oldticks, oldcpu, _, _) = before[key]
                delta = max(0, ticks - oldticks)
                if delta and (cpu in active or oldcpu in active):
                    hits.append({'pid': key[0], 'tid': key[1], 'name': name, 'cpu_seconds': delta / clock_ticks})
        return hits

    def core_idle():

        def read():
            out = {}
            for line in pathlib.Path('/proc/stat').read_text().splitlines():
                f = line.split()
                if f[0].startswith('cpu') and f[0][3:].isdigit() and (int(f[0][3:]) in selected):
                    n = list(map(int, f[1:9]))
                    out[int(f[0][3:])] = (sum(n), n[3] + n[4])
            return out
        a = read()
        time.sleep(1)
        b = read()
        return {cpu: 1 - (idle - a[cpu][1]) / max(1, total - a[cpu][0]) for (cpu, (total, idle)) in b.items()}

    def record(event):
        with (root / 'clean-events.jsonl').open('a') as f:
            f.write(json.dumps(event) + '\n')
    cases = [f'p8_trajectories/workers{workers}/{kind}_n{n}_shots{shots}' for workers in [1, 4] for kind in ['pauli', 'kraus'] for (n, shots) in [(4, 1024), (16, 64), (20, 8)]]
    if args.suite == 'fused':
        cases = [f'p8_fused/{backend}/n{n}_k{k}' for backend in ['aos', 'soa', 'fp32'] for n in [10, 20] for k in [2, 3, 5]]
        cases += [f'p8_fused/cold_{backend}/n10_k5' for backend in ['aos', 'soa', 'fp32']]
    if args.case:
        unknown = set(args.case) - set(cases)
        if unknown:
            parser.error('unknown suite cases: ' + ', '.join(sorted(unknown)))
        cases = [case for case in cases if case in args.case]
    for case in cases:
        key = case.replace('/', '__')
        for run in ['base-1', 'candidate-1', 'candidate-2', 'base-2']:
            if (case, run) in accepted:
                continue
            version = run.rsplit('-', 1)[0]
            workers = int(case.split('/')[1].replace('workers', '')) if args.suite == 'noisy' else 1
            attempt = max([e['attempt'] for e in history if e['case'] == case and e['run'] == run] + [0])
            while True:
                idle = core_idle()
                if max(idle.values()) > 0.02:
                    continue
                attempt += 1
                env = os.environ.copy()
                env['ALEPH_BENCH_OUTPUT'] = str(root / f'clean-{run}')
                if args.suite == 'fused':
                    env['RAYON_NUM_THREADS'] = '1'
                log = root / f'clean-{run}-{key}-attempt{attempt}.log'
                hits = []
                before = tasks()
                start = time.time()
                with log.open('w') as f:
                    process = subprocess.Popen(['taskset', '-c', (root / 'affinity.txt').read_text().strip(), str(root / f'bench-{version}'), case, '--bench', '--noplot'], env=env, stdout=f, stderr=subprocess.STDOUT)
                    while process.poll() is None:
                        time.sleep(0.2)
                        after = tasks()
                        hits.extend(overlap(before, after, process.pid))
                        before = after
                end = time.time()
                cpu_overlap = sum((h['cpu_seconds'] for h in hits))
                clean = cpu_overlap / max(0.001, workers * (end - start)) < 0.03
                record({'case': case, 'run': run, 'attempt': attempt, 'start_epoch': start, 'end_epoch': end, 'preflight_load': idle, 'competing': hits, 'overlap_cpu_seconds': cpu_overlap, 'overlap_fraction_one_core': cpu_overlap / max(0.001, end - start), 'exit': process.returncode, 'accepted': clean and process.returncode == 0, 'guard_policy': 'active-benchmark-physical-cores-v3-shared', 'overlap_fraction_cpu_capacity': cpu_overlap / max(0.001, workers * (end - start)), 'overlap_limit_cpu_capacity': 0.03})
                if process.returncode != 0:
                    raise RuntimeError(log.read_text()[-3000:])
                if clean:
                    print('ACCEPTED', run, case, 'attempt', attempt, 'overlap seconds', cpu_overlap, flush=True)
                    break
                print('RETRY', run, case, 'attempt', attempt, 'overlap seconds', cpu_overlap, flush=True)
    (root / 'clean-done').write_text('DONE\n')
    records = []
    for run in ['base-1', 'candidate-1', 'candidate-2', 'base-2']:
        for estimate in sorted((root / f'clean-{run}').rglob('new/estimates.json')):
            new = estimate.parent
            metadata = json.loads((new / 'benchmark.json').read_text())
            records.append({'run': run, 'case': metadata['full_id'], 'estimates': json.loads(estimate.read_text()), 'samples': json.loads((new / 'sample.json').read_text())})
    assert len(records) == 4 * len(cases), len(records)
    (root / 'clean-criterion.json').write_text(json.dumps(records, separators=(',', ':')) + '\n')
    with (root / 'clean-summary.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['case', 'base1_median_ns', 'base2_median_ns', 'candidate1_median_ns', 'candidate2_median_ns', 'baseline_mean_median_ns', 'candidate_mean_median_ns', 'speedup'])
        for case in sorted(set((r['case'] for r in records))):
            v = {r['run']: r['estimates']['median']['point_estimate'] for r in records if r['case'] == case}
            a = (v['base-1'] + v['base-2']) / 2
            b = (v['candidate-1'] + v['candidate-2']) / 2
            writer.writerow([case, v['base-1'], v['base-2'], v['candidate-1'], v['candidate-2'], a, b, a / b])
if __name__ == '__main__':
    main()
