#!/usr/bin/env python3
"""Plot paired static curves and deliberately conservative dynamic intervals from raw receipts."""
import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
x=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
colors=['#176b91','#b35b22']
for ax,code,names in zip(axes,['surface-d3-r3','gross-bb'],[['MWPM','Union-Find'],['Relay FP64','Relay Q8.3']]):
 for f,label in enumerate(names):
  for p,style in [(0.01,'--'),(0.03,'-')]:
   rows=[c for c in x['cases'] if c['name'].startswith(f'{code}-f{f}-p{p}-ratio')]
   ratios=[float(c['name'].split('-ratio')[1]) for c in rows];ys=[100*c['result']['paired_difference'] for c in rows]
   lo=[y-100*c['result']['difference_interval']['lower'] for y,c in zip(ys,rows)];hi=[100*c['result']['difference_interval']['upper']-y for y,c in zip(ys,rows)]
   ax.errorbar(ratios,ys,yerr=[lo,hi],marker='o' if f==0 else 's',linestyle=style,color=colors[f],capsize=3,label=f'{label}, true p={p}')
 ax.axhline(0,color='#777777',linewidth=.7);ax.set_xscale('log',base=4);ax.set_xticks([.25,1,4],['0.25x','1x','4x']);ax.set_xlabel('Assumed / true uniform noise rate');ax.set_title('Surface d3/r3' if code.startswith('surface') else 'Gross BB capacity');ax.set_ylabel('LER difference (percentage points)');ax.legend(fontsize=8)
fig.suptitle('Static prior mismatch: paired 95% marginal intervals\n64 independent streams × 128 shots; no adjustment across sweep points',fontsize=12)
fig.savefig(out/'static-curves.png',dpi=160);fig.savefig(out/'static-curves.svg');plt.close(fig)
fig,ax=plt.subplots(figsize=(10,5),layout='constrained');rows=[c for c in x['cases'] if c['name'].endswith(('drift','bursts'))]
labels=[]
for i,c in enumerate(rows):
 r=c['result'];y=100*r['paired_difference'];ci=r['difference_interval'];ax.errorbar(y,i,xerr=[[y-100*ci['lower']],[100*ci['upper']-y]],fmt='o',color=colors[0],capsize=3);labels.append(c['name'].replace('surface-d3-r3-f0','Surface MWPM').replace('surface-d3-r3-f1','Surface UF').replace('gross-bb-f0','Gross BB FP64').replace('gross-bb-f1','Gross BB Q8.3'))
ax.axvline(0,color='#777777',linewidth=.8);ax.set_yticks(range(len(rows)),labels);ax.invert_yaxis();ax.set_xlabel('Assumed minus fixed-base reference LER (percentage points)');ax.set_title('Drift and bursts: uncertainty uses independent streams\n64 streams; conservative 95% Hoeffding intervals remain wide');fig.savefig(out/'dynamic-intervals.png',dpi=160);fig.savefig(out/'dynamic-intervals.svg')
