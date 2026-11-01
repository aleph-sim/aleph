#!/usr/bin/env python3
"""P8-14 supplementary sync controls and CUDA timelines. Usage BINARY NEW_DIRECTORY.
Run after primary timings, never concurrently. Profiles are counter/timeline evidence,
not acceptance timings. No service or GPU clock changes.
"""
import csv
import hashlib
import io
import json
import pathlib
import subprocess
import sys


def main():
    binary,directory=sys.argv[1:]
    root=pathlib.Path(directory);root.mkdir()
    manifest=dict(binary_sha256=hashlib.sha256(pathlib.Path(binary).read_bytes()).hexdigest(),controls=[],profiles=[])
    for model in ['locality','alternating']:
        for block in range(4):
            for i in range(2):
                name=f'sync-{model}-b{block}-i{i}'
                argv=[binary,model,'24','20','sync','20']
                result=subprocess.run(argv,text=True,capture_output=True,check=True)
                (root/(name+'.csv')).write_text(result.stdout);(root/(name+'.log')).write_text(result.stderr)
                manifest['controls'].append(dict(label=name,argv=argv,rows=list(csv.DictReader(io.StringIO(result.stdout)))))
                (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for model,n,m in [('locality',24,20),('short',24,20),('locality',24,22)]:
        for mode in ['resident','overlap','combined']:
            name=f'{model}-n{n}-m{m}-{mode}'
            argv=['nsys','profile','--trace=cuda','--sample=none','--cpuctxsw=none','--force-overwrite=false','--stats=true','-o',str(root/name),binary,model,str(n),str(m),mode,str(m)]
            result=subprocess.run(argv,text=True,capture_output=True)
            (root/(name+'.out')).write_text(result.stdout);(root/(name+'.log')).write_text(result.stderr)
            manifest['profiles'].append(dict(label=name,argv=argv,exit_code=result.returncode))
            (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            assert result.returncode==0,(name,result.stdout,result.stderr)
            if not (root/(name+'.sqlite')).exists():
                subprocess.run(['nsys','export','--type','sqlite','--output',str(root/(name+'.sqlite')),str(root/(name+'.nsys-rep'))],check=True,capture_output=True)
    print('PASS: 16 synchronous controls and 9 CUDA timelines')


if __name__=='__main__':main()
