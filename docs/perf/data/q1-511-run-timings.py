#!/usr/bin/env python3
"""Paired #511 measurements in an isolated snapshot tree; never stop other workloads."""
import argparse,json,os,pathlib,subprocess,time
p=argparse.ArgumentParser();p.add_argument('root',type=pathlib.Path);args=p.parse_args();root=args.root;out=root/'artifacts'
def stat():
    result={}
    for line in pathlib.Path('/proc/stat').read_text().splitlines():
        fields=line.split()
        if fields[0].startswith('cpu'):
            nums=list(map(int,fields[1:9])); result[fields[0]]=(sum(nums),nums[3]+nums[4])
    return result
def busy(a,b,key):
    total=b[key][0]-a[key][0];idle=b[key][1]-a[key][1]
    return 100*(total-idle)/max(1,total)
def record(data):
    with (out/'host-conditions.jsonl').open('a') as f:f.write(json.dumps(data)+'\n')
# Both independent correctness/build sessions must finish before timing.
for _ in range(120):
    ready=all(path.exists() and marker in path.read_text() for path,marker in [
        (out/'base-alloc.log','test result: ok.'),(out/'alloc-release-final.log','2 passed'),
        (out/'clippy.log','Finished'),(out/'python-tests.log','OK')])
    if ready:break
    time.sleep(5)
else:raise RuntimeError('validation/build stages did not finish')
cores=[]
for cpu in range(1,20):
    pair=pathlib.Path(f'/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list').read_text().strip().split(',')
    if len(pair)==2:cores.append((cpu,int(pair[1]) if int(pair[0])==cpu else int(pair[0])))
cpu=sibling=None

def quiet(label):
    global cpu,sibling
    for _ in range(120):
        a=stat();time.sleep(2);b=stat()
        if cpu is None:cpu,sibling=min(cores,key=lambda pair:max(busy(a,b,f'cpu{pair[0]}'),busy(a,b,f'cpu{pair[1]}')))
        sample={'phase':label,'time':time.time(),'load':os.getloadavg(),'cpu':cpu,'sibling':sibling,
                'overall_busy_percent':busy(a,b,'cpu'),'cpu_busy_percent':busy(a,b,f'cpu{cpu}'),
                'sibling_busy_percent':busy(a,b,f'cpu{sibling}')}
        record(sample)
        if sample['overall_busy_percent']<10 and max(sample['cpu_busy_percent'],sample['sibling_busy_percent'])<1.1:
            print('QUIET',json.dumps(sample),flush=True);return
        time.sleep(5)
    raise RuntimeError('No quiet benchmark core; services left unchanged')
def run(label,command,env):
    quiet(label)
    a=stat();start=time.time()
    with (out/(label+'.log')).open('w') as f:
        child=subprocess.Popen(['taskset','-c',str(cpu),*command],env=env,stdout=f,stderr=subprocess.STDOUT,cwd=root)
        samples=[];last=a
        while child.poll() is None:
            time.sleep(1);now=stat();samples.append({'overall':busy(last,now,'cpu'),'sibling':busy(last,now,f'cpu{sibling}')});last=now
    record({'phase':label+'-done','time':time.time(),'duration':time.time()-start,'cpu':cpu,'sibling':sibling,'samples':samples,'exit_code':child.returncode})
    if child.returncode:raise RuntimeError(label+' failed')
    print('FINISHED',label,flush=True)
env=dict(os.environ,RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
for variant,iteration in [('base',1),('candidate',1),('candidate',2),('base',2)]:
    label=f'criterion-{variant}-{iteration}'
    caseenv=dict(env,CRITERION_HOME=str(out/label))
    bins=[x for x in (root/(variant+'-target')/'release/deps').glob('mwpm_decode-*') if x.is_file() and os.access(x,os.X_OK)]
    assert len(bins)==1,bins
    run(label,[str(bins[0]),'--bench','sparse/','--sample-size','15','--measurement-time','3','--warm-up-time','1','--noplot'],caseenv)
python='/tmp/aleph-518-K1bhhc/test-venv/bin/python'
for variant,iteration in [('base',1),('candidate',1),('candidate',2),('base',2)]:
    caseenv=dict(env,PYTHONPATH=str(root/variant/'crates/aleph-py/python')+':'+str(root/variant/'scripts/python'))
    run(f'python-{variant}-{iteration}',[python,str(root/'bench_mwpm_allocations.py'),'--repeats','7'],caseenv)
print('TIMINGS_DONE',flush=True)
