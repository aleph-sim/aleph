#!/usr/bin/env python3
"""Independent resource-availability/Lindley oracle for independent-block trace receipts.
No heap/event engine is reused. Dependent-window cases are covered by hand timelines in Rust.
Usage: python check_deadline_replay.py CAMPAIGN_DIR
"""
import json,math,sys
from pathlib import Path
root=Path(sys.argv[1])
def ns(service):
    if service['kind']=='Nanoseconds':return service['ns']
    return (service['cycles']*10**9+service['clock_hz']-1)//service['clock_hz']
def pct(values):
    if not values:return None
    s=sorted(values);return dict(count=len(s),p50_ns=s[math.ceil(.5*len(s))-1],p95_ns=s[math.ceil(.95*len(s))-1],p99_ns=s[math.ceil(.99*len(s))-1],max_ns=s[-1])
def oracle(trace,c):
    assert trace['unit']=='IndependentBlock'
    assert all(j['predecessor'] is None for j in trace['jobs'])
    available=[0]*c['workers'];scheduled=[];rows=[];primary_busy=0;maxq=0
    for j in trace['jobs']:
        row=dict(id=j['id'],arrival_ns=j['arrival_ns'],primary_start_ns=None,primary_end_ns=None,fallback_start_ns=None,terminal_ns=j['arrival_ns'],disposition='PrimaryOverflow',flagged=False,fallback_overflow=False,deadline_missed=False,decoded_logical_failure=None,final_valid=None,commit_satisfies_dependency=False)
        arrival=j['arrival_ns'];start=max(arrival,min(available));waiting=sum(s>arrival for s in scheduled)
        if start>arrival and waiting>=c['queue_capacity']:rows.append(row);continue
        duration=ns(j['primary']['service'])+c['input_overhead_ns']+c['output_overhead_ns'];end=start+duration
        available[available.index(min(available))]=end;scheduled.append(start);primary_busy+=duration
        row.update(primary_start_ns=start,primary_end_ns=end,terminal_ns=end,flagged=not j['primary']['valid'])
        maxq=max(maxq,waiting+int(start>arrival));rows.append(row)
    tail_available=[0]*c['fallback_workers'];tail_scheduled=[];tail_busy=0;max_tail=0
    order=sorted((i for i,r in enumerate(rows) if r['primary_end_ns'] is not None),key=lambda i:(rows[i]['primary_end_ns'],i))
    for i in order:
        j=trace['jobs'][i];r=rows[i];out=j['primary'];end=r['primary_end_ns']
        if c['policy']=='FallbackOnFlag' and r['flagged']:
            assert j['fallback'] is not None
            start=max(end,min(tail_available));waiting=sum(s>end for s in tail_scheduled)
            if start>end and waiting>=c['fallback_queue_capacity']:
                r['fallback_overflow']=True
                if c['fallback_overflow']=='DropJob':r['disposition']='FallbackOverflow';continue
            else:
                out=j['fallback'];cost=ns(out['service'])+c['fallback_transfer_ns']+c['output_overhead_ns'];end=start+cost
                tail_available[tail_available.index(min(tail_available))]=end;tail_scheduled.append(start);tail_busy+=cost
                r['fallback_start_ns']=start;max_tail=max(max_tail,waiting+int(start>r['primary_end_ns']))
        r['terminal_ns']=end;r['decoded_logical_failure']=out['logical_failure'];r['final_valid']=out['valid']
        r['deadline_missed']=j['deadline_after_ns'] is not None and end-j['arrival_ns']>j['deadline_after_ns']
        r['disposition']='DiscardedLate' if r['deadline_missed'] and c['late_commit']=='DiscardLate' else 'Committed'
        r['commit_satisfies_dependency']=r['disposition']=='Committed'
    first=trace['jobs'][0]['arrival_ns'] if rows else 0;horizon=max((r['terminal_ns'] for r in rows),default=first)-first
    final=[(j,r) for j,r in zip(trace['jobs'],rows) if r['disposition'] in ('Committed','DiscardedLate')]
    service=[]
    for j,r in final:
        cost=ns(j['primary']['service'])+c['input_overhead_ns']+c['output_overhead_ns']
        if r['fallback_start_ns'] is not None:cost+=ns(j['fallback']['service'])+c['fallback_transfer_ns']+c['output_overhead_ns']
        service.append(cost)
    operational=[]
    for r in rows:
        if c['operational_failure']=='LogicalOrLateOrMissing' and (r['disposition']!='Committed' or r['deadline_missed']):operational.append(True)
        else:operational.append(r['decoded_logical_failure'])
    metrics=dict(submitted=len(rows),committed=sum(r['disposition']=='Committed' for r in rows),flags=sum(r['flagged'] for r in rows),final_invalid=sum(r['final_valid'] is False for r in rows),deadline_misses=sum(r['deadline_missed'] for r in rows),overflows=sum(r['disposition']=='PrimaryOverflow' for r in rows),fallback_overflows=sum(r['fallback_overflow'] for r in rows),dependency_cancellations=0,decoded_logical_failures=sum(r['decoded_logical_failure'] is True for r in rows),decoded_known=sum(r['decoded_logical_failure'] is not None for r in rows),operational_failures=sum(v is True for v in operational),operational_unknown=sum(v is None for v in operational),horizon_ns=horizon,max_queue=maxq,max_fallback_queue=max_tail,
        primary_wait=pct([r['primary_start_ns']-r['arrival_ns'] for r in rows if r['primary_start_ns'] is not None]),fallback_wait=pct([r['fallback_start_ns']-r['primary_end_ns'] for r in rows if r['fallback_start_ns'] is not None]),service=pct(service),end_to_end=pct([r['terminal_ns']-r['arrival_ns'] for _,r in final]),mean_queue=sum(r['primary_start_ns']-r['arrival_ns'] for r in rows if r['primary_start_ns'] is not None)/horizon if horizon else None,utilization=primary_busy/(horizon*c['workers']) if horizon else None,fallback_utilization=tail_busy/(horizon*c['fallback_workers']) if horizon else None,committed_throughput_per_second=sum(r['disposition']=='Committed' for r in rows)*1e9/horizon if horizon else None)
    return rows,metrics
summary=json.loads((root/'summary.json').read_text());checks=[]
for case in summary['cases']:
    trace=json.loads((root/case['trace_file']).read_text());rows,metrics=oracle(trace,case['config'])
    for key,expected in metrics.items():
        actual=case[key]
        if isinstance(expected,float):assert math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-12),(case['name'],key,actual,expected)
        else:assert actual==expected,(case['name'],key,actual,expected)
    raw=root/(case['name']+'-replay.json')
    if raw.exists():assert json.loads(raw.read_text())['jobs']==rows,case['name']
    checks.append(dict(name=case['name'],metrics='PASS',full_job_receipts='PASS' if raw.exists() else 'not retained; deterministic replay recipe retained'))
for profile in [0,1]:
    name='golden-fixed-budget' if profile==0 else 'golden-early-exit'
    trace=json.loads((root/(name+'-trace.json')).read_text());control=json.loads((root/f'golden-f{profile}-no-loss-control.json').read_text())
    assert control['decoded_logical_failures']==sum(j['primary']['logical_failure'] is True for j in trace['jobs'])
    assert control['overflows']==0 and control['deadline_misses']==0
print(json.dumps(dict(method='Independent availability schedules for primary and fallback, direct waiting-interval integrals; no Rust event heap reused',cases=checks,no_loss_controls='PASS'),indent=2))
