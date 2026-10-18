#!/usr/bin/env python3
"""Frozen #511 native CPU campaign. Usage: CONFIG.json NEW_DIRECTORY ablation|guard_ablation|primary.
Config binds each variant to an independently built binary. Idle core/sibling gates,
fixed mirrored blocks, no services stopped, no retries or deleted observations.
"""
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time


def stat():
    rows={}
    for line in pathlib.Path('/proc/stat').read_text().splitlines():
        f=line.split()
        if f[0].startswith('cpu'):
            nums=list(map(int,f[1:9]));rows[f[0]]=(sum(nums),nums[3]+nums[4])
    return rows


def busy(before,after,key):
    total=after[key][0]-before[key][0];idle=after[key][1]-before[key][1]
    return 100*(total-idle)/max(1,total)


def main():
    config_path,directory,mode=sys.argv[1:];config=json.loads(pathlib.Path(config_path).read_text())
    assert mode in ['ablation','guard_ablation','primary'];root=pathlib.Path(directory);root.mkdir()
    variants={'ablation':['main','neighbor','candidate'], 'guard_ablation':['candidate','queued'], 'primary':['original','main','candidate']}[mode]
    cpu,sibling=config['cpu'],config['sibling']
    actual=pathlib.Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list').read_text().strip()
    assert actual==f'{cpu},{sibling}',actual
    manifest=dict(mode=mode,config=config,cpu=cpu,sibling=sibling,started_unix=time.time(),binary_sha256={v:hashlib.sha256(pathlib.Path(config[v]).read_bytes()).hexdigest() for v in variants},runs=[])
    env=dict(os.environ,RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def record(row):
        with (root/'host-conditions.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    try:
        for block in range(4 if mode=='primary' else 2):
            arms=variants[block%len(variants):]+variants[:block%len(variants)]
            for position,variant in enumerate(arms+list(reversed(arms))):
                label=f'b{block}-i{position}-{variant}'
                for attempt in range(120):
                    before=stat();time.sleep(2);after=stat()
                    row=dict(phase='preflight',label=label,time=time.time(),load=os.getloadavg(),overall=busy(before,after,'cpu'),cpu=busy(before,after,f'cpu{cpu}'),sibling=busy(before,after,f'cpu{sibling}'))
                    record(row)
                    if row['overall']<10 and max(row['cpu'],row['sibling'])<1.1:break
                    time.sleep(3)
                else:raise RuntimeError('No quiet core/sibling; services preserved')
                home=root/label
                argv=['taskset','-c',str(cpu),config[variant],'--bench','sparse/d11' if mode!='primary' else 'sparse/','--sample-size','15','--measurement-time','3','--warm-up-time','1','--noplot']
                start=time.time();samples=[]
                print('START',label,flush=True)
                with (root/(label+'.log')).open('w') as log:
                    child=subprocess.Popen(argv,env=dict(env,CRITERION_HOME=str(home)),stdout=log,stderr=subprocess.STDOUT)
                    previous=stat()
                    while child.poll() is None:
                        time.sleep(1);current=stat()
                        samples.append(dict(time=time.time(),overall=busy(previous,current,'cpu'),sibling=busy(previous,current,f'cpu{sibling}')));previous=current
                run=dict(label=label,variant=variant,block=block,argv=argv,criterion_home=str(home),duration_seconds=time.time()-start,exit_code=child.returncode,samples=samples)
                manifest['runs'].append(run);record(dict(phase='finished',**run))
                (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
                assert child.returncode==0,label
                if mode!='ablation':
                    assert max(s['overall'] for s in samples)<=10 and max(s['sibling'] for s in samples)<=5, 'Whole campaign invalid: host activity; raw observations preserved'
                print('FINISHED',label,flush=True)
    finally:
        manifest['finished_unix']=time.time();(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('PASS',mode,flush=True)


if __name__=='__main__':main()
