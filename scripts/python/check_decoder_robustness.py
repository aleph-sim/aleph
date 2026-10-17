#!/usr/bin/env python3
"""Independent interval, latent-law and tiny-model oracle for P7-04 receipts.
Usage: python check_decoder_robustness.py OUTDIR/summary.json
Requires NumPy, SciPy, Stim, PyMatching; no private SDK or hardware required.
"""
import json, math, sys
from pathlib import Path
import numpy as np
from scipy.stats import beta
import stim
import pymatching

def cp(n, k, alpha):
    return (float(beta.ppf(alpha/2,k,n-k+1)) if k else 0.0,
            float(beta.ppf(1-alpha/2,k+1,n-k)) if k<n else 1.0)

def check_row(row):
    r=row['result']; o=r['options']; n=o['streams']*o['steps']; alpha=1-o['confidence']; streams=r['streams']
    a=sum(s['assumed_failures'] for s in streams); b=sum(s['reference_failures'] for s in streams)
    plus=sum(s['worse'] for s in streams); minus=sum(s['better'] for s in streams)
    assert a-b == plus-minus
    assert r['decoder_calls']==2*n
    if r['process']['kind']=='Static':
        ac=cp(n,a,alpha);bc=cp(n,b,alpha);pc=cp(n,plus,alpha/2);mc=cp(n,minus,alpha/2)
        dc=(max(-1,pc[0]-mc[1]),min(1,pc[1]-mc[0]))
    else:
        radius=math.sqrt(math.log(2/alpha)/(2*o['streams']))
        ac=(max(0,a/n-radius),min(1,a/n+radius));bc=(max(0,b/n-radius),min(1,b/n+radius))
        dc=(max(-1,(a-b)/n-2*radius),min(1,(a-b)/n+2*radius))
    for name,expected in [('assumed_interval',ac),('reference_interval',bc),('difference_interval',dc)]:
        got=r[name]; assert abs(got['lower']-expected[0])<1e-10,(row['name'],name)
        assert abs(got['upper']-expected[1])<1e-10,(row['name'],name)
    law={}
    if r['process']['kind']=='MarkovBurst':
        p=r['process']; on=p['onset']; off=p['recovery']; q=0.; active=0.;aa=0.;starts=0.
        for _ in range(o['steps']):
            aa+=q*(1-off);starts+=(1-q)*on;q=(1-q)*on+q*(1-off);active+=q
        actual=sum(s['active_steps'] for s in streams)/n
        actual_aa=sum(s['active_adjacent_pairs'] for s in streams)/(o['streams']*max(1,o['steps']-1))
        expected=active/o['steps']; expected_aa=aa/max(1,o['steps']-1)
        # Descriptive tolerance, fixed in advance; unit tests separately check conditional transitions.
        assert abs(actual-expected)<.05,(row['name'],actual,expected)
        assert abs(actual_aa-expected_aa)<.05
        law=dict(active_fraction=actual,expected_active_fraction=expected,
                 active_adjacent_fraction=actual_aa,expected_active_adjacent_fraction=expected_aa,
                 starts=sum(s['burst_starts'] for s in streams),expected_starts=starts*o['streams'],
                 duration_mean_asymptotic=1/off,censoring='runs ending at a stream boundary are right-censored')
    return dict(name=row['name'],interval_checks='PASS',burst_law=law)

def tiny_oracle(row):
    r=row['result']; generating=stim.DetectorErrorModel(r['generating_dem']);generating_errors=[e for e in generating.flattened() if e.type=='error']
    truths=[]
    for field in ['assumed_dem','reference_dem']:
        model=stim.DetectorErrorModel(r[field]);m=pymatching.Matching()
        for e in model.flattened():
            if e.type!='error':continue
            det=[t.val for t in e.targets_copy() if t.is_relative_detector_id()];obs={t.val for t in e.targets_copy() if t.is_logical_observable_id()}
            assert len(det)<=1
            if det:
                p=e.args_copy()[0];m.add_boundary_edge(det[0],fault_ids=obs,weight=math.log1p(-p)-math.log(p),merge_strategy='smallest-weight')
        expected=0.
        for mask in range(1<<len(generating_errors)):
            dets=np.zeros(generating.num_detectors,dtype=np.uint8);obs=np.zeros(generating.num_observables,dtype=np.uint8);mass=1.
            for i,e in enumerate(generating_errors):
                fired=bool(mask>>i&1);p=e.args_copy()[0];mass*=p if fired else 1-p
                if fired:
                    for t in e.targets_copy():
                        if t.is_relative_detector_id():dets[t.val]^=1
                        elif t.is_logical_observable_id():obs[t.val]^=1
            expected+=mass*bool(np.any(m.decode(dets)!=obs))
        truths.append(expected)
    assert abs(truths[1]-.1925)<1e-12
    for expected,key in zip(truths,['assumed_interval','reference_interval']):
        assert r[key]['lower']<=expected<=r[key]['upper']
    delta=truths[0]-truths[1];assert r['difference_interval']['lower']<=delta<=r['difference_interval']['upper']
    # Separate independent sampler/control at 100k draws; never substitutes for native outcome replay.
    dets,obs,_=generating.compile_sampler(seed=546).sample(100000)
    count=int(np.count_nonzero(obs[:,0]));observed=count/100000
    exact_obs=0.
    for mask in range(1<<len(generating_errors)):
        mass=1.;parity=0
        for i,e in enumerate(generating_errors):
            fired=bool(mask>>i&1);p=e.args_copy()[0];mass*=p if fired else 1-p
            if fired and any(t.is_logical_observable_id() and t.val==0 for t in e.targets_copy()):parity^=1
        exact_obs+=mass*parity
    assert abs(observed-exact_obs)<.01
    return dict(assumed_truth=truths[0],reference_truth=truths[1],difference_truth=delta,
                matching_policy='explicit smallest-weight boundary edges; default DEM importer has different parallel-edge merge semantics',
                stim_observable0_frequency=observed,exact_observable0_probability=exact_obs)

report=json.loads(Path(sys.argv[1]).read_text())
print(json.dumps(dict(cases=[check_row(c) for c in report['cases']],tiny=check_row(report['tiny']),tiny_oracle=tiny_oracle(report['tiny'])),indent=2))

# Optional independent fixed-endpoint coverage validation.
if len(sys.argv)>2:
    cov=json.loads(Path(sys.argv[2]).read_text())
    base=stim.DetectorErrorModel(cov['base_dem']); mechanisms=[e for e in base.flattened() if e.type=='error']
    base_p=[e.args_copy()[0] for e in mechanisms]
    classifiers=[]
    for field in ['assumed_dem','reference_dem']:
        matching=pymatching.Matching()
        for e in stim.DetectorErrorModel(cov[field]).flattened():
            if e.type!='error':continue
            dets=[t.val for t in e.targets_copy() if t.is_relative_detector_id()]
            obs={t.val for t in e.targets_copy() if t.is_logical_observable_id()}
            if dets:
                p=e.args_copy()[0];matching.add_boundary_edge(dets[0],fault_ids=obs,weight=math.log1p(-p)-math.log(p),merge_strategy='smallest-weight')
        failures=[]
        for mask in range(1<<len(mechanisms)):
            ds=np.zeros(base.num_detectors,dtype=np.uint8);os_=np.zeros(base.num_observables,dtype=np.uint8)
            for i,e in enumerate(mechanisms):
                if mask>>i&1:
                    for t in e.targets_copy():
                        if t.is_relative_detector_id():ds[t.val]^=1
                        elif t.is_logical_observable_id():os_[t.val]^=1
            failures.append(bool(np.any(matching.decode(ds)!=os_)))
        classifiers.append(failures)
    def exact_rates(ps):
        totals=[0.,0.]
        for mask in range(1<<len(ps)):
            mass=math.prod(p if mask>>i&1 else 1-p for i,p in enumerate(ps))
            for i,fs in enumerate(classifiers):totals[i]+=mass*fs[mask]
        return totals
    checks=[]
    for c in cov['campaigns']:
        law=c['process']; q=0.;a=0.;b=0.
        for t in range(c['steps']):
            ps=list(base_p)
            if law['kind']=='LinearDrift':
                f=t/(c['steps']-1);ps=[(1-f)*p+f*end for p,end in zip(ps,law['end_probabilities'])]
            if law['kind']=='MarkovBurst':
                q=(1-q)*law['onset']+q*(1-law['recovery'])
                for i in law['affected']:ps[i]=min(1,ps[i]*law['multiplier'])
                baseline=exact_rates(base_p);active=exact_rates(ps);rates=[(1-q)*lo+q*hi for lo,hi in zip(baseline,active)]
            else:rates=exact_rates(ps)
            a+=rates[0]/c['steps'];b+=rates[1]/c['steps']
        assert abs(c['expected_assumed']-a)<1e-12
        assert abs(c['expected_reference']-b)<1e-12
        counts=[sum(r[key]['lower']<=truth<=r[key]['upper'] for r in c['intervals']) for key,truth in [('assumed',a),('reference',b),('difference',a-b)]]
        assert counts==c['covered']
        for count,ci in zip(counts,c['coverage_binomial95']):
            lo,hi=cp(c['trials'],count,.05);assert abs(ci['lower']-lo)<1e-10 and abs(ci['upper']-hi)<1e-10
        checks.append(dict(process=law['kind'],trials=c['trials'],covered=counts,truth=[a,b,a-b]))
    # A separate output file preserves one valid JSON object for each oracle stage.
    Path(sys.argv[2]+'.oracle.json').write_text(json.dumps(dict(coverage=checks),indent=2)+'\n')
