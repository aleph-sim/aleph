#!/usr/bin/env python3
"""P8-10 fixed balanced experiment. Usage: BINARY ORIGINAL_BINARY NEW_OUTPUT_DIRECTORY.
Four mirrored blocks, fixed inputs and endpoints; no retries/outlier removal. CPU mask
caches bind DEM/input SHA256 and configuration, never unchecked GPU output. Services and
clocks are unchanged; only the Rust upload/kernel/download boundary is timed.
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
    binary, original, directory = sys.argv[1:]
    root = pathlib.Path(directory)
    root.mkdir()
    (root/'oracles').mkdir()
    cells = [(m,n,p) for m in ['surface7','surface11','gross'] for n in [1,128,4096] for p in [.001,.03]]
    cells += [(m,n,.03) for m in ['surface7','gross'] for n in [33,4097]]
    manifest = dict(version=1,protocol='four mirrored blocks; rotate leading arm and reverse cells in odd blocks; 64 timed repeats for single shots, three for batches; CPU validation and final GPU warmup excluded',cells=cells,binary_sha256=hashlib.sha256(pathlib.Path(binary).read_bytes()).hexdigest(),original_binary_sha256=hashlib.sha256(pathlib.Path(original).read_bytes()).hexdigest(),started_unix=time.time(),measurements=[])
    with (root/'gpu-monitor.csv').open('w') as log:
        monitor = subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,clocks.sm,power.draw,temperature.gpu','--format=csv,noheader,nounits','--loop-ms=200'],stdout=log,stderr=subprocess.STDOUT)
        try:
            def measure(executable,model,n,p,mode,label,repeats):
                oracle=root/'oracles'/f'{model}-n{n}-p{p}.json'
                argv=[executable,model,str(n),str(p),str(repeats),mode,str(oracle)]
                result=subprocess.run(argv,capture_output=True,text=True,check=True)
                (root/(label+'.csv')).write_text(result.stdout)
                (root/(label+'.log')).write_text(result.stderr)
                rows=list(csv.DictReader(io.StringIO(result.stdout)))
                assert len(rows)==repeats
                manifest['measurements'].append(dict(label=label,argv=argv,rows=rows))
            for block in range(4):
                modes=['shot','warp','coop'];modes=modes[block%3:]+modes[:block%3]
                order=modes+list(reversed(modes))
                for model,n,p in (cells if block%2==0 else list(reversed(cells))):
                    for position,mode in enumerate(order):
                        measure(binary,model,n,p,mode,f'b{block}-{model}-n{n}-p{p}-i{position}-{mode}',64 if n==1 else 3)
            for block in range(4):
                for model,n,p in [('surface7',128,.03),('surface11',4096,.03),('gross',128,.03)]:
                    for position,arm in enumerate(['original','shot','shot','original']):
                        measure(original if arm=='original' else binary,model,n,p,'shot',f'control-b{block}-{model}-i{position}-{arm}',3)
        finally:
            monitor.terminate();monitor.wait()
            manifest['finished_unix']=time.time()
            (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',invocations=len(manifest['measurements']),output=str(root)),indent=2))


if __name__=='__main__':
    main()
