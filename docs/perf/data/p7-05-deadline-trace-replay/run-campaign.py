"""Fixed P7-05 golden adapter and replay campaign; shared services remain running; no result-based retries."""
from pathlib import Path
import json,os,subprocess,time,resource
root=Path('/tmp/aleph-547-508c4d0')
def cpus():
 d={}
 for line in Path('/proc/stat').read_text().splitlines():
  n,*v=line.split()
  if n in ['cpu3','cpu23']:
   v=list(map(int,v));d[n]={'total':sum(v[:8]),'busy':sum(v[:8])-v[3]-v[4]}
 return d
preflights=[]
for _ in range(60):
 a=cpus();time.sleep(1);b=cpus();fractions={c:(b[c]['busy']-a[c]['busy'])/max(1,b[c]['total']-a[c]['total']) for c in a};preflights.append(fractions)
 if max(fractions.values())<=.02:break
else:raise RuntimeError('CPU3 and sibling23 remained busy; no service was stopped')
start=cpus();t=time.monotonic();usage0=resource.getrusage(resource.RUSAGE_CHILDREN)
with (root/'campaign.log').open('w') as log:
 p=subprocess.Popen(['taskset','-c','3',(root/'example-executable.txt').read_text().strip(),'campaign',str(root/'campaign'),'2048'],env=dict(os.environ,RAYON_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)
 trajectory=[]
 while p.poll() is None:
  trajectory.append(dict(elapsed=time.monotonic()-t,cpus=cpus(),load=os.getloadavg()));time.sleep(.5)
 code=p.wait()
end=cpus();usage1=resource.getrusage(resource.RUSAGE_CHILDREN);elapsed=time.monotonic()-t;own=usage1.ru_utime+usage1.ru_stime-usage0.ru_utime-usage0.ru_stime
record=dict(protocol='Q7-07 full and early golden policies on same 2048 ordered inputs, seed2024; 30 fixed replay cases; no independent resampling; CPU3, Rayon1; no service changes',preflights=preflights,trajectory=trajectory,exit_code=code,elapsed_seconds=elapsed,own_cpu_seconds=own,external_cpu3_fraction=max(0,(end['cpu3']['busy']-start['cpu3']['busy'])/os.sysconf('SC_CLK_TCK')-own)/elapsed,sibling23_busy_fraction=(end['cpu23']['busy']-start['cpu23']['busy'])/max(1,end['cpu23']['total']-start['cpu23']['total']))
(root/'campaign-load.json').write_text(json.dumps(record,indent=2)+'\n');assert code==0,code
print('Campaign complete',elapsed)
