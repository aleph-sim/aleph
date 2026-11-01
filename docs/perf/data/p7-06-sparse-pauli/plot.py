"""Scientific figure from the valid retained campaign; no timing recapture.
Usage: python plot.py SUMMARY_JSON OUTPUT_DIRECTORY
"""
import pathlib,json,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
summary=json.loads(pathlib.Path(sys.argv[1]).read_text());out=pathlib.Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
cases=sorted([c for c in summary['cases'] if c['name'].startswith('product16-k')],key=lambda c:int(c['name'].split('-k')[1]))
ks=[int(c['name'].split('-k')[1]) for c in cases]
fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
for mode,label in [('sparse','Sparse, whole circuit'),('sparse-cone','Sparse + cone'),('sv','FP64 SV reference'),('mps','MPS (no recorded truncation)')]:
 rows=[next(x for x in c['modes'] if x['mode']==mode) for c in cases]
 axes[0].plot(ks,[x['median_total_ns']/1e3 for x in rows],marker='o',label=label)
 axes[1].plot(ks,[x['cold_peak_rss_kib']/1024 for x in rows],marker='o',label=label)
 if mode=='sparse':
  for k,row in zip(ks,rows):
   if k in [0,4,8,12,16]:axes[0].annotate(f"{row['stats']['peak_terms']:,} term" + ("s" if row['stats']['peak_terms'] != 1 else ""),(k,row['median_total_ns']/1e3),xytext=(0,10),textcoords='offset points',ha='center',fontsize=8)
axes[0].set(yscale='log',xlabel='Non-Clifford Ry rotations (16 qubits)',ylabel='Median total public API time (µs)',title='Exact term expansion eventually loses to SV and MPS')
axes[1].set(xlabel='Non-Clifford Ry rotations (16 qubits)',ylabel='Cold process peak RSS (MiB)',title='RSS includes executable, plans and allocator overhead')
axes[0].margins(y=.18)
axes[0].legend(fontsize=8,loc='upper left');axes[1].legend(fontsize=8,loc='upper left')
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('Product-state stress case — 10 paired observations per API, all retained',fontsize=11)
for ext in ['png','svg']:fig.savefig(out/f'term-growth.{ext}',dpi=150)
