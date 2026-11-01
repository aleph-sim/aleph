import json,os,pathlib,subprocess,time
root=pathlib.Path('/tmp/aleph-514-IoS4E1');out=root/'artifacts'
for _ in range(180):
    path=out/'clippy.log'
    if path.exists() and 'Finished' in path.read_text():break
    time.sleep(2)
else:raise RuntimeError('final validation not complete')
assert 'Ran 36 tests' in (out/'python-tests.log').read_text() and '\nOK\n' in (out/'python-tests.log').read_text()
def cpu():
    result={}
    for line in pathlib.Path('/proc/stat').read_text().splitlines():
        f=line.split()
        if f[0].startswith('cpu'):
            n=list(map(int,f[1:9]));result[f[0]]=(sum(n),n[3]+n[4])
    return result
def busy(a,b,key):
    total=b[key][0]-a[key][0];idle=b[key][1]-a[key][1]
    return 100*(total-idle)/max(1,total)
pair=pathlib.Path('/sys/devices/system/cpu/cpu2/topology/thread_siblings_list').read_text().strip()
siblings=[]
for span in pair.split(','):
    ends=list(map(int,span.split('-')))
    siblings.extend(range(ends[0],ends[-1]+1))
others=[v for v in siblings if v!=2]
for variant,iteration in [('base',1),('candidate',1),('candidate',2),('base',2)]:
    for _ in range(120):
        a=cpu();time.sleep(2);b=cpu()
        state={'variant':variant,'round':iteration,'time':time.time(),'load':os.getloadavg(),'cpu':2,'siblings':others,'overall_busy':busy(a,b,'cpu'),'core_busy':busy(a,b,'cpu2'),'sibling_busy':[busy(a,b,f'cpu{x}') for x in others]}
        with (out/'audit-host.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
        if state['overall_busy']<5 and max([state['core_busy'],*state['sibling_busy']])<=1.1:break
        time.sleep(5)
    else:raise RuntimeError('host/core busy; services unchanged')
    env=dict(os.environ,RAYON_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR=str(root/'matplotlib'),PYTHONPATH=str(root/variant/'crates/aleph-py/python')+':'+str(root/'audit'))
    command=['taskset','-c','2','/root/cqvenv/bin/python',str(root/'audit/audit_relay_budget.py'),'--shots','256','--repeats','3']
    if variant=='base':command+=['--legacy-only']
    with (out/f'audit-{variant}-{iteration}.json').open('w') as f, (out/f'audit-{variant}-{iteration}.stderr').open('w') as e:
        subprocess.run(command,env=env,stdout=f,stderr=e,check=True)
    print('AUDIT_DONE',variant,iteration,flush=True)
print('ALL_AUDITS_DONE',flush=True)
