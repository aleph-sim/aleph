#!/usr/bin/env python3
"""P8-14 frozen balanced paging experiment. Usage: BINARY NEW_DIRECTORY.
Run only after correctness and an idle/load/memory check. Fixed 4 mirrored blocks,
no retries/outlier deletion; retain allocation-inclusive API endpoints and telemetry.
"""
import csv
import hashlib
import io
import json
import pathlib
import subprocess
import sys
import time


def main():
    binary, directory = sys.argv[1:]
    root = pathlib.Path(directory)
    root.mkdir()
    cells = [('locality',24,20),('locality',26,22),('alternating',24,20),('short',24,20),('locality',24,22),('qft',24,20)]
    manifest = dict(version=1,binary_sha256=hashlib.sha256(pathlib.Path(binary).read_bytes()).hexdigest(),cells=cells,started_unix=time.time(),measurements=[])
    with (root/'gpu-monitor.csv').open('w') as log:
        monitor = subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,clocks.sm,power.draw,temperature.gpu','--format=csv,noheader,nounits','--loop-ms=200'],stdout=log,stderr=subprocess.STDOUT)
        try:
            for block in range(4):
                for model,n,m in (cells if block%2==0 else list(reversed(cells))):
                    modes=['resident','overlap','combined']
                    if model=='locality' and n-m==4:modes.append('resident-large')
                    modes=modes[block%len(modes):]+modes[:block%len(modes)]
                    for position,mode in enumerate(modes+list(reversed(modes))):
                        label=f'b{block}-{model}-n{n}-m{m}-i{position}-{mode}'
                        argv=[binary,model,str(n),str(m+2 if mode=='resident-large' else m),'resident' if mode=='resident-large' else mode,str(m)]
                        r=subprocess.run(argv,text=True,capture_output=True,check=True)
                        (root/(label+'.csv')).write_text(r.stdout);(root/(label+'.log')).write_text(r.stderr)
                        rows=list(csv.DictReader(io.StringIO(r.stdout)));assert len(rows)==1
                        manifest['measurements'].append(dict(label=label,mode=mode,argv=argv,rows=rows))
        finally:
            monitor.terminate();monitor.wait();manifest['finished_unix']=time.time()
            (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',invocations=len(manifest['measurements'])),indent=2))


if __name__=='__main__':main()
