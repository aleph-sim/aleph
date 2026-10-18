"""Replay recorded near-load tuples at 2/5/10-period deadlines; no timing recapture.
Usage: python deadline-sweep.py EXTRACTED_CAMPAIGN OUTDIR REPLAY_EXECUTABLE
Then: python scripts/python/check_deadline_replay.py OUTDIR
"""
import json,pathlib,subprocess,sys,shutil
source,out,binary=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),sys.argv[3]
out.mkdir(exist_ok=True,parents=True)
for pattern in ['golden-*-trace.json','golden-*-no-loss-control.json']:
 for f in source.glob(pattern):shutil.copyfile(f,out/f.name)
original=json.loads((source/'summary.json').read_text());cases=[]
for profile in [0,1]:
 trace=json.loads((source/f'golden-f{profile}-near-trace.json').read_text())
 period=original['early_mean_service_ns']
 for multiple in [2,5,10]:
  for j in trace['jobs']:j['deadline_after_ns']=multiple*period
  name=f'f{profile}-d{multiple}-trace.json';(out/name).write_text(json.dumps(trace))
  for workers in [1,2]:
   for policy in ['FlagOnly','FallbackOnFlag']:
    template=next(c for c in original['cases'] if c['name']==f'golden-f{profile}-near-w{workers}-{policy}')
    cfg=out/'config.json';cfg.write_text(json.dumps(template['config']))
    label=f'f{profile}-d{multiple}-w{workers}-{policy}'
    path=out/(label+'-replay.json')
    subprocess.run([binary,'replay',str(out/name),str(cfg),str(path)],check=True,capture_output=True)
    result=json.loads(path.read_text());result.pop('jobs');result.pop('queue_levels')
    result.update(name=label,trace_file=name,submitted=len(trace['jobs']),deadline_multiple=multiple)
    cases.append(result)
(out/'summary.json').write_text(json.dumps(dict(protocol='Recorded Linux near-load tuples only; deterministic deadline sensitivity, no new capture',cases=cases),indent=2)+'\n')
print(len(cases),'deadline cases')
