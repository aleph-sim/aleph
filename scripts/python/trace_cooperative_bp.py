#!/usr/bin/env python3
"""Supplementary P8-10 workload-tail experiment; never replaces the frozen go gate.
Usage: BINARY NEW_OUTPUT_DIRECTORY. 64 deterministic physical-noise inputs per cell,
replayed as individual API calls in a persistent decoder. All CPU masks/flags retained.
Four mirrored blocks compare the same ordered inputs, three arms, no retries/filtering.
"""
import csv,hashlib,io,json,pathlib,subprocess,sys,time


def main():
    binary,directory=sys.argv[1:];root=pathlib.Path(directory);root.mkdir();(root/'oracles').mkdir()
    cells=[(model,p) for model in ['surface7','surface11','gross'] for p in [.001,.03]]
    manifest=dict(version=1,supplementary=True,unique_inputs_per_cell=64,cells=cells,binary_sha256=hashlib.sha256(pathlib.Path(binary).read_bytes()).hexdigest(),started_unix=time.time(),measurements=[])
    with (root/'gpu-monitor.csv').open('w') as log:
        monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,clocks.sm,power.draw,temperature.gpu','--format=csv,noheader,nounits','--loop-ms=200'],stdout=log,stderr=subprocess.STDOUT)
        try:
            for block in range(4):
                modes=['shot','warp','coop'];modes=modes[block%3:]+modes[:block%3]
                for model,p in (cells if block%2==0 else list(reversed(cells))):
                    for position,mode in enumerate(modes+list(reversed(modes))):
                        label=f'b{block}-{model}-p{p}-i{position}-{mode}'
                        argv=[binary,model,'64',str(p),'1',mode,str(root/'oracles'/f'{model}-p{p}.json'),'--trace']
                        result=subprocess.run(argv,text=True,capture_output=True,check=True)
                        (root/(label+'.csv')).write_text(result.stdout);(root/(label+'.log')).write_text(result.stderr)
                        rows=list(csv.DictReader(io.StringIO(result.stdout)));assert len(rows)==64
                        manifest['measurements'].append(dict(label=label,argv=argv,rows=rows))
        finally:
            monitor.terminate();monitor.wait();manifest['finished_unix']=time.time()
            (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',invocations=len(manifest['measurements'])),indent=2))


if __name__=='__main__':main()
