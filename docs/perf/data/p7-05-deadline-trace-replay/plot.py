"""Plot recorded deterministic trace receipts, without recapturing timings.
Usage: python plot.py EXTRACTED_CAMPAIGN OUTPUT_DIRECTORY
"""
import json,pathlib,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root,out=map(pathlib.Path,sys.argv[1:]);out.mkdir(parents=True,exist_ok=True)
fig,ax=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
r=json.loads((root/'rare-spikes-FlagOnly-replay.json').read_text())
rows=[j for j in r['jobs'] if 110<=j['id']<=180]
x=[j['id'] for j in rows if j['primary_start_ns'] is not None]
y=[(j['terminal_ns']-j['arrival_ns']) for j in rows if j['primary_start_ns'] is not None]
ax[0].plot(x,y,'o-',ms=3,label='Completed jobs')
dropped=[j['id'] for j in rows if j['primary_start_ns'] is None]
ax[0].scatter(dropped,[0]*len(dropped),marker='x',color='tab:red',label='Overflow (no latency sample)')
ax[0].axhline(500,color='black',ls='--',label='Deadline')
ax[0].set(xlabel='Ordered job identity',ylabel='End-to-end latency (ns)',title='Rare spikes: mean service 138 ns < period 160 ns')
ax[0].legend(fontsize=8)
summary=json.loads((root/'summary.json').read_text())
names=['correlated-flags-FlagOnly','correlated-flags-FallbackOnFlag']
c=[next(c for c in summary['cases'] if c['name']==n) for n in names]
x=[0,1];w=.23
for shift,key,label in [(-w,'decoded_logical_failures','Decoded logical'),(0,'deadline_misses','Late output'),(w,'operational_failures','Union SLA')]:
 ax[1].bar([v+shift for v in x],[v[key] for v in c],w,label=label)
ax[1].set(xticks=x,xticklabels=['Flag only','Fallback on flag'],ylabel='Jobs (512 submitted)',title='Clustered flags: repair can worsen the SLA')
ax[1].legend(fontsize=8)
fig.suptitle('Synthetic finite-queue controls — illustrative costs, not silicon measurements',fontsize=11)
for ext in ['png','svg']:fig.savefig(out/f'queue-policy.{ext}',dpi=150)
