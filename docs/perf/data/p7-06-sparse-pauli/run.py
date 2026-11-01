"""Predeclared paired API campaign, cold process RSS, Criterion and interference logging.
Shared services retained. No timing-based retries or best-run selection.
"""
import pathlib,subprocess,json,os,time,resource
ROOT=pathlib.Path('/tmp/aleph-548-f6a7808');HZ=os.sysconf('SC_CLK_TCK')
def select(path,kind):
 matches=[]
 for line in path.read_text().splitlines():
  x=json.loads(line)
  if x.get('reason')=='compiler-artifact' and x.get('executable') and kind in x['target']['kind'] and x['target']['name']=='sparse_expectation':matches.append(x['executable'])
 assert len(matches)==1,matches
 return matches[0]
def cpus():
 out={}
 for line in pathlib.Path('/proc/stat').read_text().splitlines():
  name,*v=line.split()
  if name in ['cpu3','cpu23']:
   a=list(map(int,v));out[name]=dict(total=sum(a[:8]),busy=sum(a[:8])-a[3]-a[4])
 return out
def run(label,args,env=None):
 pre=[]
 for _ in range(60):
  a=cpus();time.sleep(1);b=cpus();f={k:(b[k]['busy']-a[k]['busy'])/max(1,b[k]['total']-a[k]['total']) for k in a};pre.append(f)
  if max(f.values())<=.02:break
 else:raise RuntimeError('no quiet CPU3/sibling23 starting interval')
 a=cpus();u=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic();trajectory=[]
 with (ROOT/(label+'.log')).open('w') as log:
  p=subprocess.Popen(['taskset','-c','3',*args],env=env,stdout=log,stderr=subprocess.STDOUT)
  while p.poll() is None:
   trajectory.append(dict(elapsed=time.monotonic()-start,cpus=cpus(),loadavg=os.getloadavg()));time.sleep(.5)
  code=p.wait()
 elapsed=time.monotonic()-start;b=cpus();v=resource.getrusage(resource.RUSAGE_CHILDREN);own=v.ru_utime+v.ru_stime-u.ru_utime-u.ru_stime
 (ROOT/(label+'-load.json')).write_text(json.dumps(dict(preflights=pre,trajectory=trajectory,elapsed_seconds=elapsed,own_cpu_seconds=own,external_cpu3_fraction=max(0,(b['cpu3']['busy']-a['cpu3']['busy'])/HZ-own)/elapsed,sibling23_busy_fraction=(b['cpu23']['busy']-a['cpu23']['busy'])/max(1,b['cpu23']['total']-a['cpu23']['total']),exit_code=code),indent=2)+'\n')
 assert code==0,(label,code)
example=select(ROOT/'example-artifacts.jsonl','example');bench=select(ROOT/'bench-artifacts.jsonl','bench')
(ROOT/'executables.json').write_text(json.dumps(dict(example=example,bench=bench),indent=2)+'\n')
env=dict(os.environ,RAYON_NUM_THREADS='1')
run('campaign',[example,'--audit'],env)
# Separate cold processes: RSS includes executable/runtime, inputs/plans, and allocator state.
# Per-query sparse payload estimates and SV state bytes are distinct metrics in stdout.
rss=ROOT/'rss';rss.mkdir(exist_ok=True)
names=['ghz18-clifford','ghz18-rotation','local128','ghz128','cancel16']+[f'product16-k{k}' for k in [0,2,4,8,12,16]]
for name in names:
 modes=['sparse','sparse-cone','mps']+(['cone-sv'] if name=='local128' else [] if name=='ghz128' else ['sv'])
 for mode in modes:
  with (rss/f'{name}-{mode}.json').open('w') as out:
   subprocess.run(['/usr/bin/time','-v','-o',str(rss/f'{name}-{mode}.time'),'taskset','-c','3',example,'--one',name,mode],env=env,stdout=out,check=True)
run('criterion',[bench,'--bench'],dict(env,ALEPH_BENCH_OUTPUT=str(ROOT/'criterion')))
(ROOT/'validation-complete').write_text('success\n')
print('all campaigns complete')
