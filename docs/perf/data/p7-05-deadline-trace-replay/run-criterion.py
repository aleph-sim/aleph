"""Run predeclared statistical campaign, retain load trajectory; timings are descriptive."""
import pathlib,subprocess,os,time,json,threading,resource
ROOT=pathlib.Path('/tmp/aleph-547-508c4d0')
HZ=os.sysconf('SC_CLK_TCK')
def cpus():
    result={}
    for row in pathlib.Path('/proc/stat').read_text().splitlines():
        name,*v=row.split()
        if name in ['cpu3','cpu23']:
            x=list(map(int,v));result[name]={'total':sum(x[:8]),'busy':sum(x[:8])-x[3]-x[4]}
    return result
# Select a low-load start without stopping shared services.
preflights=[]
for _ in range(60):
    a=cpus();time.sleep(1);b=cpus()
    fractions={c:(b[c]['busy']-a[c]['busy'])/max(1,b[c]['total']-a[c]['total']) for c in a}
    preflights.append(fractions)
    if max(fractions.values())<=.02:break
else:raise RuntimeError('no quiet CPU3/23 start')
start=cpus();r0=resource.getrusage(resource.RUSAGE_CHILDREN);t=time.monotonic()
env=dict(os.environ,RAYON_NUM_THREADS='1',ALEPH_BENCH_OUTPUT=str(ROOT/'criterion'))
binary=pathlib.Path((ROOT/'bench-executable.txt').read_text().strip())
with (ROOT/'criterion.log').open('w') as log:
    p=subprocess.Popen(['taskset','-c','3',str(binary),'--bench'],env=env,stdout=log,stderr=subprocess.STDOUT)
    trajectory=[]
    while p.poll() is None:
        trajectory.append(dict(elapsed=time.monotonic()-t,cpus=cpus(),loadavg=os.getloadavg()))
        time.sleep(.5)
    code=p.wait()
end=cpus();r1=resource.getrusage(resource.RUSAGE_CHILDREN);elapsed=time.monotonic()-t
own=r1.ru_utime+r1.ru_stime-r0.ru_utime-r0.ru_stime
record=dict(protocol='Criterion pure event replay costs, 512 and4096 synthetic jobs, CPU3/Rayon1; validation/hash/report costs included; CPU3 affinity, Rayon1; services retained',preflights=preflights,trajectory=trajectory,elapsed_seconds=elapsed,own_cpu_seconds=own,external_cpu3_fraction=max(0,(end['cpu3']['busy']-start['cpu3']['busy'])/HZ-own)/elapsed,sibling23_busy=(end['cpu23']['busy']-start['cpu23']['busy'])/max(1,end['cpu23']['total']-start['cpu23']['total']),exit_code=code)
(ROOT/'criterion-load.json').write_text(json.dumps(record,indent=2)+'\n')
assert code==0,code
print('criterion complete',elapsed,'seconds')
