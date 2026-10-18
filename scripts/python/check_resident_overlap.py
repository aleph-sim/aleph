#!/usr/bin/env python3
"""Independent P8-14 paired-block analysis; usage CAMPAIGN_DIRECTORY.
Every allocation-inclusive sample is retained. Ratios are descriptive, not CIs.
"""
import csv
import json
import math
import pathlib
import re
import statistics
import sys


def check(root):
    manifest=json.loads((root/'manifest.json').read_text())
    records=manifest['measurements']
    cells=manifest['cells']
    assert len(cells)==6 and len(records)==160
    assert len({r['label'] for r in records})==160
    summaries=[]
    for model,n,m in cells:
        selected=[r for r in records if r['argv'][1:3]==[model,str(n)] and r['argv'][-1]==str(m)]
        modes=['resident','overlap','combined']+(['resident-large'] if model=='locality' and n-m==4 else [])
        assert len(selected)==8*len(modes)
        medians={};blocks={}
        for mode in modes:
            arm=[r for r in selected if r['mode']==mode];assert len(arm)==8
            values=[]
            for r in arm:
                with (root/(r['label']+'.csv')).open() as f: assert list(csv.DictReader(f))==r['rows']
                assert len(r['rows'])==1;row=r['rows'][0]
                assert [row['model'],row['n'],row['m'],row['mode'],row['hot']]==[model,str(n),str(m+2 if mode=='resident-large' else m),'resident' if mode=='resident-large' else mode,str(m)]
                assert math.isfinite(float(row['norm'])) and abs(float(row['norm'])-1)<1e-9
                assert int(row['ns'])>0;values.append(int(row['ns']))
                log=(root/(r['label']+'.log')).read_text()
                metadata=dict(re.findall(r'(\w+)=([^\s]+)',log))
                if model=='locality': flags=([True]*(2*m-1)+[False])*4
                elif model=='alternating': flags=[True,False]*16
                elif model=='short': flags=[True,True,False]*4
                else:
                    tile=int(row['m'])
                    flags=[]
                    for q in range(n):
                        flags.append(q<tile)
                        flags.extend(k<tile for k in range(q+1,n))
                passes=0;low=False
                for flag in flags:
                    if not flag or not low:passes+=1
                    low=flag
                assert metadata['modelled']=='true'
                assert int(metadata['state_bytes'])==16<<n
                assert int(metadata['passes_per_gate'])==len(flags)
                assert int(metadata['residency_passes'])==passes
                assert int(metadata['resident_transfer_bytes'])==(32<<n)*passes
            medians[mode]=statistics.median(values)
            blocks[mode]=[statistics.median([int(r['rows'][0]['ns']) for r in arm if r['label'].startswith(f'b{block}-')]) for block in range(4)]
        ratios=[a/b for a,b in zip(blocks['resident'],blocks['combined'])]
        speedup=medians['resident']/medians['combined']
        equal=medians.get('resident-large',medians['resident'])/medians['combined']
        heldout=(model,n,m)!=('locality',24,20)
        summaries.append(dict(model=model,n=n,m=m,median_ns=medians,block_ns=blocks,speedup_resident=speedup,block_speedup_resident=ratios,speedup_best_existing=min(medians['resident'],medians['overlap'])/medians['combined'],block_speedup_best_existing=[min(a,b)/c for a,b,c in zip(blocks['resident'],blocks['overlap'],blocks['combined'])],speedup_equal_payload=equal,heldout=heldout,meets_gate=heldout and speedup>=1.05 and min(ratios)>1 and equal>1))
    return dict(status='PASS',invocations=len(records),cells=summaries,verdict='GO_OPT_IN' if any(c['meets_gate'] for c in summaries) else 'NO_GO',scope='Fixed allocation-inclusive API endpoints; four paired blocks, no outlier deletion; equal-payload larger synchronous tiles control locality-cell adoption.')


if __name__=='__main__': print(json.dumps(check(pathlib.Path(sys.argv[1])),indent=2))
