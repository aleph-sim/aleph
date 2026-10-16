#!/usr/bin/env python3
"""Paired public-route experiments. Stdlib-only controller; never changes services."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone

VERSION = 1
ROOT = Path(__file__).resolve().parents[2]
STAGES = ('startup_ms', 'parse_compile_ms', 'execution_ms', 'sampling_ms',
          'materialization_ms', 'formatting_ms', 'run_api_ms')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def number(x):
    return type(x) in (int, float) and math.isfinite(x)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_spec(s):
    require(isinstance(s, dict), 'spec must be an object')
    require(type(s.get('schema_version')) is int and s['schema_version'] == VERSION, 'unsupported schema_version')
    require(s.get('route') in ('cpu-cli', 'cuda-python'), 'invalid route')
    require(s.get('comparison') in ('same-algorithm', 'algorithm-change', 'historical'),
            'declare comparison class; historical/cross-algorithm results stay separate')
    for k in ('name', 'model', 'precision', 'topology', 'stop_rule'):
        require(isinstance(s.get(k), str) and bool(s[k].strip()), f'missing {k}')
    require(s['precision'] in ('f64', 'f32'), 'invalid precision')
    budget = s.get('error_budget', {})
    require(isinstance(budget, dict), 'error_budget must be an object')
    for key, cap in [('amplitude_atol', 1e-10 if s['precision'] == 'f64' else 1e-5), ('norm_atol', 1e-4)]:
        require(number(budget.get(key)) and 0 < budget[key] <= cap, f'invalid or relaxed {key}')
    for k, low, high in [('qubits', 2, 22), ('shots', 1, 1024), ('pairs', 2, 30),
                         ('threads', 1, 256), ('workers', 1, 256), ('seed', 0, 2**32-1)]:
        require(type(s.get(k)) is int and low <= s[k] <= high, f'invalid {k}')
    require(s['workers'] == 1, 'reference adapters execute one worker; workers must be 1')
    require(type(s.get('warmup_calls')) is int and s['warmup_calls'] in (0, 1), 'warmup_calls must be 0 or 1')
    require(s['route'] != 'cpu-cli' or s['warmup_calls'] == 0,
            'CLI always creates a fresh process; cannot label it warmed')
    require(number(s.get('max_load_per_cpu')) and s['max_load_per_cpu'] > 0,
            'declare max_load_per_cpu')
    require(number(s.get('max_gpu_utilization')) and 0 <= s['max_gpu_utilization'] <= 100,
            'declare max_gpu_utilization')
    require(number(s.get('bottleneck_fraction')) and 0 <= s['bottleneck_fraction'] < 1,
            'bottleneck_fraction must be in [0,1)')
    require(number(s.get('target_speedup')) and s['target_speedup'] >= 1,
            'target_speedup must be >=1')
    require(1/(1-s['bottleneck_fraction']) >= s['target_speedup'],
            'Amdahl upper bound cannot meet target; stop before broad campaign')
    arms = s.get('arms', [])
    require(isinstance(arms, list) and len(arms) == 2 and all(isinstance(a, dict) for a in arms)
            and {a.get('name') for a in arms} == {'a', 'b'}, 'need arms a/b')
    for arm in arms:
        require(re.fullmatch(r'[0-9a-f]{40}', arm.get('revision', '')) is not None,
                'each arm needs the full source revision (declared, not inferred from binary)')
        require(isinstance(arm.get('optimizer'), bool), 'declare optimizer per arm')
        require(isinstance(arm.get('algorithm'), str) and arm['algorithm'], 'declare algorithm')
        for key in ('build_command', 'rustflags', 'compiler'):
            require(isinstance(arm.get(key), str) and arm[key], f'declare {key} (use unset if absent)')
        require(isinstance(arm.get('executable'), str) and arm['executable'], 'declare executable path')
        if s['route'] == 'cpu-cli':
            require(arm['optimizer'] is False, 'CPU reference runs unoptimized CLI (no new CPU optimizer)')
    if s['comparison'] == 'same-algorithm':
        require(arms[0]['algorithm'] == arms[1]['algorithm'], 'algorithm mismatch requires separate class')
    gate = s.get('preflight', {})
    require(isinstance(gate, dict), 'preflight must be an object')
    require(gate.get('kind') in ('counter-change', 'control'), 'declare preflight gate')
    if gate['kind'] == 'counter-change':
        require(isinstance(gate.get('counter'), str), 'preflight needs counter')
    else:
        require(isinstance(gate.get('reason'), str) and gate['reason'], 'control needs reason')
    require(s['model'] in ('brickwall', 'ghz'), 'reference models: brickwall or ghz')
    return s


def qasm(s):
    n = s['qubits']
    lines = ['OPENQASM 3.0; include "stdgates.inc";', f'qubit[{n}] q;']
    if s['model'] == 'ghz':
        lines += ['h q[0];'] + [f'cx q[{q}],q[{q+1}];' for q in range(n-1)]
    else:
        rng = random.Random(s['seed'])
        for layer in range(6):
            for q in range(n):
                lines += [f'ry({rng.random():.17g}) q[{q}];', f'rz({rng.random():.17g}) q[{q}];']
            lines += [f'cx q[{q}],q[{q+1}];' for q in range(layer % 2, n-1, 2)]
    return '\n'.join(lines)


def probe(command):
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=5)
        return dict(command=command, returncode=p.returncode, stdout=p.stdout.strip(), stderr=p.stderr[-1000:])
    except (OSError, subprocess.TimeoutExpired) as e:
        return dict(command=command, returncode=None, error=str(e))


def resources(gpu=False):
    r = dict(time_utc=datetime.now(timezone.utc).isoformat(), load=os.getloadavg(),
             cpu_count=os.cpu_count())
    # Names only, never full command lines/environment; unrelated command lines may contain secrets.
    ps = probe(['ps', '-eo', 'comm='])
    names = ps.get('stdout', '').splitlines()
    r['competing_processes'] = [x.strip() for x in names if any(
        token in x for token in ('Runner.Worker', 'cargo', 'bencher'))]
    if gpu:
        r['gpu'] = probe(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,utilization.gpu,memory.used,memory.free',
                          '--format=csv,noheader,nounits'])
    return r


def resource_ok(s, r):
    if r['load'][0] / (r['cpu_count'] or 1) > s['max_load_per_cpu'] or r['competing_processes']:
        return False
    if s['route'] == 'cuda-python':
        p = r['gpu']
        if p['returncode'] != 0:
            return False
        try:
            lines = p['stdout'].splitlines()
            return bool(lines) and all(float(line.split(',')[-3]) <= s['max_gpu_utilization'] for line in lines)
        except (ValueError, IndexError):
            return False
    return True


def validate_receipt(r, s):
    require(isinstance(r, dict), 'receipt must be an object')
    require(type(r.get('protocol')) is int and r['protocol'] == VERSION and r.get('status') == 'executed', 'missing execution receipt')
    require(r.get('route') == s['route'], 'wrong public route')
    require(r.get('hardware_executed') is (s['route'] == 'cuda-python'), 'hardware receipt mismatch')
    require(r.get('warmup_calls') == s['warmup_calls'], 'incorrect warm/cold receipt')
    require(r.get('shots_observed') == s['shots'], 'public counts contract failed')
    require(r.get('qubits') == s['qubits'], 'workload mismatch')
    require(isinstance(r.get('counters'), dict), 'missing counters')
    require(all(number(v) for v in r['counters'].values()), 'non-finite counter')
    require(isinstance(r.get('stages'), dict), 'stages must be an object')
    for k in STAGES:
        require(k in r.get('stages', {}), f'missing stage {k}; use null for unavailable')
        require(r['stages'][k] is None or number(r['stages'][k]) and r['stages'][k] >= 0,
                f'invalid stage {k}')


def invoke(s, arm, work, path, preflight=False):
    env = dict(os.environ)
    env['PYTHONOPTIMIZE'] = '0'  # hardware correctness asserts must execute
    for k in ('RAYON_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[k] = str(s['threads'])
    if s['route'] == 'cpu-cli':
        env['ALEPH_BENCH_CPU_DIAGNOSTICS'] = '1'
        command = [arm['executable'], 'run', str(path), '--backend', 'sv', '--precision', s['precision'],
                   '--shots', str(s['shots']), '--seed', str(s['seed'])]
    else:
        request = work / ('request-' + arm['name'] + '.json')
        request.write_text(json.dumps(dict(spec=s, arm=arm, qasm=str(path), preflight=preflight)))
        command = [arm['executable'], str(Path(__file__).with_name('cuda_worker.py')), str(request)]
    if sha256(arm['executable']) != arm['executable_sha256']:
        return dict(status='failed', reason='executable changed during campaign', command=command)
    samples = [resources(s['route'] == 'cuda-python')]
    if not resource_ok(s, samples[0]):
        return dict(status='contaminated', reason='resource preflight rejected', resources=samples, command=command)
    done = threading.Event()
    def monitor():
        while not done.wait(.5):
            samples.append(resources(s['route'] == 'cuda-python'))
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    started = time.perf_counter()
    try:
        p = subprocess.run(command, env=env, text=True, capture_output=True, timeout=180)
        caller_ms = (time.perf_counter() - started) * 1000
    except (OSError, subprocess.TimeoutExpired) as e:
        return dict(status='failed', reason=str(e), command=command, resources=samples)
    finally:
        done.set()
        thread.join()
    samples.append(resources(s['route'] == 'cuda-python'))
    base = dict(command=command, caller_ms=caller_ms, returncode=p.returncode,
                stdout=p.stdout[-16000:], stderr=p.stderr[-16000:], resources=samples)
    if p.returncode != 0:
        return dict(base, status='failed', reason='subprocess failed')
    try:
        if s['route'] == 'cpu-cli':
            stats = json.loads(next(x.split('=', 1)[1] for x in p.stderr.splitlines() if x.startswith('CPU_EXECUTION=')))
            count_rows = re.findall(r'\|([01]+)⟩\s+(\d+)\s+\(', p.stdout)
            require(all(len(bits) == s['qubits'] for bits, _ in count_rows), 'wrong count width')
            if s['model'] == 'ghz':
                require(all(bits in ('0'*s['qubits'], '1'*s['qubits']) for bits, _ in count_rows),
                        'GHZ output support oracle failed')
            receipt = dict(protocol=VERSION, status='executed', route=s['route'], hardware_executed=False,
                           warmup_calls=0, shots_observed=sum(int(v) for _, v in count_rows), qubits=stats['qubits'],
                           counters=dict(output_rows=len(count_rows)), stages=dict.fromkeys(STAGES))
            receipt['stages'].update(parse_compile_ms=stats['parse_ms'], execution_ms=stats['execution_ms'],
                                     sampling_ms=stats['sampling_ms'], materialization_ms=stats['materialization_ms'],
                                     formatting_ms=stats['formatting_ms'])
        else:
            receipt = json.loads(p.stdout)
        validate_receipt(receipt, s)
        if preflight and s['route'] == 'cuda-python':
            require(receipt.get('oracle_executed') is True, 'preflight CPU oracle did not execute')
        if s['route'] == 'cuda-python':
            require(number(receipt.get('caller_ms')) and receipt['caller_ms'] > 0, 'invalid API caller clock')
            base['process_wall_ms'] = base['caller_ms']
            base['caller_ms'] = receipt['caller_ms']
    except (ValueError, KeyError, StopIteration, TypeError) as e:
        return dict(base, status='failed', reason='invalid receipt: ' + str(e))
    # During execution own kernels may raise GPU utilization. Poll values are
    # retained for review; pre/post GPU use and CPU load/competitors gate results.
    quiet = all(resource_ok(s, dict(r, gpu=samples[0].get('gpu'))) for r in samples)
    quiet = quiet and resource_ok(s, samples[-1])
    return dict(base, status='executed' if quiet else 'contaminated', receipt=receipt,
                reason=None if quiet else 'resource monitoring rejected')


def summarize(rows, seed):
    pairs = {}
    for r in rows:
        if r['phase'] == 'measurement' and r['status'] == 'executed':
            pairs.setdefault(r['pair'], {})[r['arm']] = r['caller_ms']
    complete = [p for p in pairs.values() if set(p) == {'a', 'b'}]
    if not complete:
        return dict(pairs=0)
    ratios = [p['a']/p['b'] for p in complete]
    rng = random.Random(seed)
    boot = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(2000))
    return dict(pairs=len(complete), median_paired_speedup=statistics.median(ratios),
                paired_speedup_range=[min(ratios), max(ratios)],
                paired_bootstrap_95=[boot[50], boot[1949]],
                uncertainty_note='paired median bootstrap; small N is exploratory, not a stable small-win claim',
                arms={a: dict(median_ms=statistics.median(p[a] for p in complete),
                              range_ms=[min(p[a] for p in complete), max(p[a] for p in complete)]) for a in ('a', 'b')})


def run_campaign(s, output, skip_hardware=False):
    validate_spec(s)
    s = copy.deepcopy(s)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    manifest = dict(schema_version=VERSION, status='running', spec=s,
                    source=probe(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']),
                    dirty=probe(['git', '-C', str(ROOT), 'status', '--porcelain']),
                    harness_sha256={str(p.relative_to(ROOT)):sha256(p) for p in (Path(__file__), Path(__file__).with_name('cuda_worker.py'))},
                    compiler=probe(['rustc', '-Vv']), platform=platform.platform(),
                    cpu=platform.processor(), cpu_topology=probe(['lscpu', '-J']), python=sys.version,
                    affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
                    environment={k: str(s['threads']) for k in ('RAYON_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                    clock_contract=('CLI caller is subprocess launch through exit; CUDA caller is parse/run/read/result-release inside worker; process_wall includes imports/warmups/oracle'),
                    source_revision_note='arm revisions are operator declarations; artifact hashes identify executed files',
                    sampling_interval_seconds=.5, amdahl_upper_bound=1/(1-s['bottleneck_fraction']))
    def save():
        (output/'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    save()
    def append(row):
        rows.append(row)
        with (output/'runs.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
    try:
        if skip_hardware and s['route'] == 'cuda-python':
            append(dict(phase='preflight', status='skipped', reason='explicit --skip-hardware', hardware_executed=False))
            manifest['status'] = 'skipped'
            return manifest
        for arm in s['arms']:
            arm['executable'] = str(Path(arm['executable']).resolve())
            arm['executable_sha256'] = sha256(arm['executable'])
        path = output/'workload.qasm'
        path.write_text(qasm(s))
        manifest['workload_sha256'] = sha256(path)
        with tempfile.TemporaryDirectory(prefix='aleph-benchmark-') as tmp:
            work = Path(tmp)
            for pair in range(-1, s['pairs']):
                phase = 'preflight' if pair == -1 else 'measurement'
                arms = s['arms'] if pair % 2 == 0 else list(reversed(s['arms']))
                for arm in arms:
                    row = dict(invoke(s, arm, work, path.resolve(), preflight=phase == 'preflight'), phase=phase, pair=pair, arm=arm['name'])
                    if row['status'] == 'executed' and s['route'] == 'cuda-python':
                        imported = row['receipt'].get('imported_artifact')
                        require(isinstance(imported, dict) and re.fullmatch(r'[0-9a-f]{64}', imported.get('sha256', '')),
                                'missing imported native artifact hash')
                        previous = arm.setdefault('imported_artifact', imported)
                        if previous != imported:
                            row.update(status='failed', reason='imported native artifact changed during campaign')
                    append(row)
                    if row['status'] != 'executed':
                        manifest['status'] = row['status']
                        return manifest
                if pair == -1 and s['preflight']['kind'] == 'counter-change':
                    counter = s['preflight']['counter']
                    values = [r['receipt']['counters'].get(counter) for r in rows]
                    require(all(number(v) for v in values) and values[0] != values[1],
                            f'inert or missing swept counter {counter}; stop before campaign')
            manifest['status'] = 'executed'
            manifest['summary'] = summarize(rows, s['seed'])
        return manifest
    except (ValueError, OSError) as e:
        manifest.update(status='failed', error=str(e))
        return manifest
    finally:
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()


def validate_result(directory):
    """Recheck a saved experiment without launching a process or requiring a GPU."""
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    validate_spec(manifest['spec'])
    require(manifest.get('schema_version') == VERSION, 'unsupported result version')
    status = manifest.get('status')
    require(status in ('executed', 'failed', 'skipped', 'contaminated'), 'incomplete result')
    raw = directory/'runs.jsonl'
    rows = [json.loads(x) for x in raw.read_text().splitlines()] if raw.is_file() else []
    require(rows or status == 'failed', 'missing raw records')
    for row in rows:
        require(row.get('status') in ('executed', 'failed', 'skipped', 'contaminated'), 'invalid run status')
        if row['status'] == 'executed':
            validate_receipt(row['receipt'], manifest['spec'])
            require(number(row.get('caller_ms')) and row['caller_ms'] > 0, 'invalid caller time')
    if status == 'executed':
        s = manifest['spec']
        expected = [(pair, arm) for pair in range(-1,s['pairs'])
                    for arm in (('a','b') if pair%2==0 else ('b','a'))]
        require([(r.get('pair'),r.get('arm')) for r in rows] == expected, 'missing or misordered raw pairs')
        require(all(r['status']=='executed' for r in rows), 'non-executed row in successful campaign')
        if s['route'] == 'cuda-python':
            require(all(r['receipt'].get('oracle_executed') is True for r in rows[:2]), 'missing preflight oracle')
        if s['preflight']['kind'] == 'counter-change':
            counter = s['preflight']['counter']
            values = [r['receipt']['counters'].get(counter) for r in rows[:2]]
            require(all(number(v) for v in values) and values[0] != values[1], 'inert preflight counter')
        require(manifest.get('summary') == summarize(rows,s['seed']), 'summary does not match raw pairs')
        require(sha256(directory/'workload.qasm') == manifest['workload_sha256'], 'workload hash mismatch')
    else:
        require('summary' not in manifest, 'failed/skipped/contaminated experiment cannot have a success summary')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('spec', type=Path, nargs='?')
    p.add_argument('--output', type=Path)
    p.add_argument('--validate-result', type=Path)
    p.add_argument('--skip-hardware', action='store_true')
    args = p.parse_args()
    if args.validate_result:
        if args.spec or args.output or args.skip_hardware:
            p.error('--validate-result is a separate read-only operation')
        result = validate_result(args.validate_result)
    else:
        if not args.spec or not args.output:
            p.error('spec and --output are required for a campaign')
        result = run_campaign(json.loads(args.spec.read_text()), args.output, args.skip_hardware)
    print(json.dumps(dict(status=result['status'], summary=result.get('summary'))))
    return 0 if result['status'] == 'executed' else 2


if __name__ == '__main__':
    sys.exit(main())
