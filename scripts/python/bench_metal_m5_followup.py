"""#520 bounded follow-up of the interrupted M5 MPS cell with power/drift receipts.

Build and verify correctness before invoking. No compilation, service changes,
cap overrides or OOM probes occur here. Existing bench executables are explicit.
"""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import threading
import time


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(command):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['sv-fused', 'sv-readout', 'mps-batched', 'mps-vs-cpu']:
        parser.add_argument('--'+name, type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    binaries = {name: getattr(args, name.replace('-', '_')).resolve() for name in ['sv-fused', 'sv-readout', 'mps-batched', 'mps-vs-cpu']}
    manifest = dict(started_utc=utc(), source_head=read(['git', 'rev-parse', 'HEAD']),
                    binaries={name: dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for name, p in binaries.items()},
                    rust=read(['rustc', '--version']), macos=read(['sw_vers', '-productVersion']),
                    rayon_num_threads=os.environ.get('RAYON_NUM_THREADS', 'unset (default pool)'),
                    commands=[], samples=10, criterion_warmup_seconds=3,
                    criterion_requested_measurement_seconds=5,
                    scope='Missing n20/chi512 MPS CPU/GPU, repeated CPU and SV controls, readout and layer-batching refresh; historical full-suite data remains separate.')
    stop = threading.Event()
    monitor_errors = []

    def monitor():
        with (output/'host-power.jsonl').open('w') as log:
            while not stop.is_set():
                try:
                    row = dict(utc=utc(), load=os.getloadavg(), power=read(['pmset', '-g', 'batt']),
                               thermal_status=read(['pmset', '-g', 'therm']))
                except Exception as error:
                    monitor_errors.append(str(error))
                    return
                log.write(json.dumps(row)+'\n');log.flush()
                stop.wait(10)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        steps = [
            ('sv_before', 'sv-fused', r'^sv_fused/random/(unfused|fused)/18$'),
            ('readout', 'sv-readout', r'^sv_readout/'),
            ('batching', 'mps-batched', r'^mps_batched/'),
            ('cpu_before', 'mps-vs-cpu', r'^mps_vs_cpu/large_n/cpu/n20_chi512$'),
            ('gpu_missing', 'mps-vs-cpu', r'^mps_vs_cpu/large_n/gpu/n20_chi512$'),
            ('cpu_after', 'mps-vs-cpu', r'^mps_vs_cpu/large_n/cpu/n20_chi512$'),
            ('sv_after', 'sv-fused', r'^sv_fused/random/(unfused|fused)/18$'),
        ]
        for label, binary, pattern in steps:
            home=output/'criterion'/label
            env=dict(os.environ, CRITERION_HOME=str(home))
            command=[str(binaries[binary]), '--bench', '--noplot', pattern, '--sample-size', '10']
            record=dict(label=label, started_utc=utc(), command=command, power_before=read(['pmset', '-g', 'batt']))
            manifest['commands'].append(record)
            print(label,record['started_utc'],flush=True)
            with (output/(label+'.log')).open('w') as log, (output/(label+'-events.jsonl')).open('w') as events:
                proc=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                for line in proc.stdout:
                    log.write(line);log.flush();events.write(json.dumps(dict(utc=utc(),line=line.rstrip()))+'\n');events.flush()
                code=proc.wait()
            record.update(finished_utc=utc(),returncode=code,power_after=read(['pmset','-g','batt']))
            (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            if monitor_errors: raise RuntimeError('Power/load telemetry failed: '+monitor_errors[0])
            if code: raise RuntimeError(f'{label} failed with exit {code}')
            if not list(home.glob('**/new/sample.json')):
                raise RuntimeError(f'{label} matched no actual measured cells')
    finally:
        stop.set();watcher.join(timeout=30)
        manifest['telemetry_errors']=monitor_errors
        manifest['finished_utc']=utc()
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    main()
