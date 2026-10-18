#!/usr/bin/env python3
"""Validate and summarize P8-10 Nsight receipts. Usage: PROFILE_DIRECTORY.
Profiler replay durations are intentionally separate from acceptance timings.
"""
import csv
import io
import json
import pathlib
import sys


def check(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    assert len(manifest['profiles']) == 15
    assert len({r['label'] for r in manifest['profiles']}) == 15
    profiles = []
    for record in manifest['profiles']:
        assert record['exit_code'] == 0
        text = (root / (record['label'] + '.csv')).read_text()
        rows = list(csv.DictReader(io.StringIO(text[text.index('"ID",'):])))
        assert len(rows) == 8 and all(r['Kernel Name'] == 'bp_decode' for r in rows)
        values = {r['Metric Name']: float(r['Metric Value'].replace(',', '')) for r in rows}
        profiles.append(dict(label=record['label'], block=rows[0]['Block Size'], grid=rows[0]['Grid Size'],
            ld_sectors_per_request=values['l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum'] / values['l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum'],
            dram_read_bytes=values['dram__bytes_read.sum'], dram_write_bytes=values['dram__bytes_write.sum'],
            occupancy_percent=values['sm__warps_active.avg.pct_of_peak_sustained_active'],
            active_lanes=values['smsp__thread_inst_executed_per_inst_executed.ratio'],
            registers=values['launch__registers_per_thread'], profiled_kernel_ns=values['gpu__time_duration.sum']))
    return dict(status='PASS', profiles=profiles, scope='15 matched cold-cache Nsight counter profiles; profiler replay durations are not primary acceptance timings.')


if __name__ == '__main__':
    print(json.dumps(check(pathlib.Path(sys.argv[1])), indent=2))
