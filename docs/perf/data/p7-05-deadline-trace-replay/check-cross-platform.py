import json, pathlib, subprocess, math, sys
# Usage: python check-cross-platform.py EXTRACTED_CAMPAIGN REPLAY_EXECUTABLE
root=pathlib.Path(sys.argv[1])
out=root.parent/'mac-replay';out.mkdir(exist_ok=True)
checks=[]
def eq(a,b,path=''):
 if isinstance(a,float):assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(path,a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys(),path
  for k in a:eq(a[k],b[k],path+'/'+k)
 elif isinstance(a,list):
  assert len(a)==len(b),path
  for i,(v,w) in enumerate(zip(a,b)):eq(v,w,path+'/'+str(i))
 else:assert a==b,(path,a,b)
s=json.loads((root/'summary.json').read_text())
for f in sorted(root.glob('*-replay.json'))+sorted(root.glob('*-no-loss-control.json')):
 expected=json.loads(f.read_text())
 if 'no-loss-control' in f.name:trace=f.name.replace('no-loss-control','no-loss-trace')
 else:
  case=next(c for c in s['cases'] if c['name']+'-replay.json'==f.name)
  trace=case['trace_file']
 cfg=out/(f.stem+'-config.json');cfg.write_text(json.dumps(expected['config']))
 result=out/f.name
 subprocess.run([sys.argv[2],'replay',str(root/trace),str(cfg),str(result)],check=True,capture_output=True)
 eq(expected,json.loads(result.read_text()))
 checks.append(dict(receipt=f.name,jobs=len(expected['jobs']),exact_integer_and_job_fields=True,float_tolerance=1e-12))
(root.parent/'cross-platform.json').write_text(json.dumps(dict(platform='macOS arm64; recorded Linux service times replayed, no host timings recaptured',checks=checks),indent=2)+'\n')
print(len(checks),'receipts identical across Linux/Mac')
