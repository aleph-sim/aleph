#!/usr/bin/env python3
"""Select four quiet Linux physical cores on one socket and record SMT siblings."""
import pathlib
import os
import time
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('artifacts', type=pathlib.Path)
    out = parser.parse_args().artifacts
    out.mkdir(parents=True, exist_ok=True)

    def cpu_times():
        values = {}
        for line in pathlib.Path('/proc/stat').read_text().splitlines():
            fields = line.split()
            if fields[0].startswith('cpu') and fields[0][3:].isdigit():
                nums = list(map(int, fields[1:9]))
                values[int(fields[0][3:])] = (sum(nums), nums[3] + nums[4])
        return values
    first = cpu_times()
    time.sleep(2)
    last = cpu_times()
    load = {cpu: 1 - (idle - first[cpu][1]) / max(1, total - first[cpu][0]) for (cpu, (total, idle)) in last.items()}
    groups = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        p = pathlib.Path('/sys/devices/system/cpu') / f'cpu{cpu}' / 'topology'
        socket = int((p / 'physical_package_id').read_text())
        core = int((p / 'core_id').read_text())
        groups.setdefault(socket, {}).setdefault(core, []).append(cpu)
    choices = []
    for (socket, cores) in groups.items():
        if len(cores) >= 4:
            ranked = sorted(cores.items(), key=lambda item: max((load[cpu] for cpu in item[1])))[:4]
            choices.append((sum((max((load[cpu] for cpu in siblings)) for (_, siblings) in ranked)), socket, ranked))
    if not choices:
        raise SystemExit('Need four physical cores')
    (_, socket, chosen) = min(choices)
    affinity = [siblings[0] for (_, siblings) in chosen]
    siblings = sorted((cpu for (_, cpus) in chosen for cpu in cpus))
    monitor = next((cpu for cpu in reversed(sorted(os.sched_getaffinity(0))) if cpu not in siblings))
    (out / 'affinity.txt').write_text(','.join(map(str, affinity)) + '\n')
    (out / 'core-policy.json').write_text(json.dumps({'socket': socket, 'affinity': affinity, 'monitored_smt_siblings': siblings, 'monitor_cpu': monitor, 'selection_load': {cpu: load[cpu] for cpu in siblings}}, indent=2) + '\n')
if __name__ == '__main__':
    main()
