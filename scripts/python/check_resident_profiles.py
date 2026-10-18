#!/usr/bin/env python3
"""P8-14 independent Nsight stage/copy accounting; usage PROFILE_DIRECTORY.
Only instrumented timelines are analyzed here; their durations do not select the go gate.
"""
import csv
import json
import math
import pathlib
import sqlite3
import statistics
import sys


def overlap(intervals):
    events=[]
    for kind, spans in intervals.items():
        for start,end in spans:
            events.extend([(start,kind,1),(end,kind,-1)])
    events.sort()
    active={k:0 for k in intervals};previous=events[0][0];union=hd=copy_compute=0
    for time,kind,delta in events:
        width=time-previous
        if any(active.values()):union+=width
        if active['H'] and active['D']:hd+=width
        if active['K'] and (active['H'] or active['D']):copy_compute+=width
        active[kind]+=delta;previous=time
    return dict(busy_union_ns=union,h2d_d2h_overlap_ns=hd,copy_compute_overlap_ns=copy_compute)


def check(root):
    manifest=json.loads((root/'manifest.json').read_text())
    assert len(manifest['controls'])==16 and len(manifest['profiles'])==9
    controls=[]
    for model in ['locality','alternating']:
        rows=[]
        for r in manifest['controls']:
            if r['argv'][1]!=model:continue
            with (root/(r['label']+'.csv')).open() as f:assert list(csv.DictReader(f))==r['rows']
            assert len(r['rows'])==1
            row=r['rows'][0]
            assert (row['n'],row['m'],row['hot'],row['mode'])==('24','20','20','sync')
            assert math.isfinite(float(row['norm'])) and abs(float(row['norm'])-1)<1e-9
            assert int(row['ns'])>0;rows.append(int(row['ns']))
        assert len(rows)==8
        controls.append(dict(model=model,median_ns=statistics.median(rows),samples_ns=rows))
    profiles=[]
    for r in manifest['profiles']:
        assert r['exit_code']==0
        model,n,m,mode,hot=r['argv'][-5:];n=int(n);m=int(m);assert hot==str(m)
        c=sqlite3.connect(f'file:{root/(r["label"]+".sqlite")}?mode=ro',uri=True)
        copies=list(c.execute('select start,end,bytes,copyKind,streamId from CUPTI_ACTIVITY_KIND_MEMCPY where bytes>=?',(16<<m,)))
        assert all(k in [1,2] for _,_,_,k,_ in copies)
        spans={'H':[(a,b) for a,b,_,k,_ in copies if k==1], 'D':[(a,b) for a,b,_,k,_ in copies if k==2]}
        first=min(a for a,_ in spans['H']);last=max(b for _,b in spans['D'])
        kernels=list(c.execute('select start,end,streamId from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=?',(first,last)))
        assert kernels
        spans['K']=[(a,b) for a,b,_ in kernels]
        per_gate=8*m if model=='locality' else 12
        passes=per_gate if mode=='overlap' else 8
        expected_direction=(16<<n)*passes
        for kind in [1,2]:assert sum(size for _,_,size,k,_ in copies if k==kind)==expected_direction
        expected_launches=(4*(2*m-1)*(1<<(n-m))+4*(1<<(n-m-1))) if model=='locality' else (8*(1<<(n-m))+4*(1<<(n-m-2)))
        assert len(kernels)==expected_launches
        apis=list(c.execute('select a.start,a.end,s.value from CUPTI_ACTIVITY_KIND_RUNTIME a join StringIds s on s.id=a.nameId'))
        api_counts={}
        for _,_,name in apis:
            if 'Memcpy' in name or 'Synchronize' in name:api_counts[name]=api_counts.get(name,0)+1
        assert not any('Memcpy' in name and 'Async' not in name for name in api_counts)
        pinned=[(a,b) for a,b,name in apis if 'MemHostAlloc' in name or 'MemAllocHost' in name]
        assert len(pinned)==1
        stats=overlap(spans)
        profiles.append(dict(label=r['label'],mode=mode,model=model,n=n,m=m,api_counts=api_counts,observed_passes=passes,tile_copy_count=len(copies),h2d_bytes=expected_direction,d2h_bytes=expected_direction,kernel_count=len(kernels),h2d_streams=sorted({stream for _,_,_,k,stream in copies if k==1}),d2h_streams=sorted({stream for _,_,_,k,stream in copies if k==2}),kernel_streams=sorted({stream for _,_,stream in kernels}),setup_from_pinned_alloc_to_first_copy_ns=first-pinned[0][0],pinned_alloc_api_ns=pinned[0][1]-pinned[0][0],initial_memset_bytes=c.execute('select sum(bytes) from CUPTI_ACTIVITY_KIND_MEMSET').fetchone()[0],gpu_pipeline_span_ns=last-first,fill_to_first_kernel_ns=min(a for a,_ in spans['K'])-first,drain_after_last_kernel_ns=last-max(b for _,b in spans['K']),summed_h2d_ns=sum(b-a for a,b in spans['H']),summed_d2h_ns=sum(b-a for a,b in spans['D']),summed_kernel_ns=sum(b-a for a,b in spans['K']),**stats))
        assert profiles[-1]['setup_from_pinned_alloc_to_first_copy_ns']>=0
        c.close()
    return dict(status='PASS',controls=controls,profiles=profiles,scope='Post-primary instrumented CUDA timelines; allocation/setup interval includes host initialization, ring allocation and stream setup. Fill/drain are descriptive global endpoints, not independent additive stage costs. Busy-time sums overlap and must not be added to infer API time.')


if __name__=='__main__':print(json.dumps(check(pathlib.Path(sys.argv[1])),indent=2))
