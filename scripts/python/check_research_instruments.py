#!/usr/bin/env python3
"""Stdlib-only independent identity/interval/analytic checks for P7-00 integration artifacts.
Usage: python check_research_instruments.py RUN_ROOT [--compare OTHER_RUN_ROOT]
Native Rust import/replay/minimality checks happen inside the studies; this checker does not
pretend to independently implement MWPM/UF or validate the correlated-process confidence proof.
"""
import argparse, hashlib, json, math, pathlib

def load(path):
    def unique(pairs):
        d = {}
        for k, v in pairs:
            if k in d: raise ValueError(f'duplicate JSON key {k}')
            d[k] = v
        return d
    return json.loads(path.read_text(), object_pairs_hook=unique)

def sha(data): return hashlib.sha256(data).hexdigest()

def file_at(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path

def close(a, b, tol=1e-10):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tol, (a,b)

def cdf(n,k,p):
    return math.fsum(math.comb(n,j)*p**j*(1-p)**(n-j) for j in range(k+1))

def root_cdf(n,k,target):
    lo,hi=0.,1.
    for _ in range(100):
        m=(lo+hi)/2
        if cdf(n,k,m)>target: lo=m
        else: hi=m
    return (lo+hi)/2

def check(root):
    root=pathlib.Path(root)
    q=load(root/'qec/study.json'); o=load(root/'observables/study.json')
    assert q['study_version']==1 and o['study_version']==1
    assert o['initial_state']=='zero' and o['precision']=='f64'
    surface=q['surface']; n=surface['shots']; k=surface['failures']
    assert n==512 and 0<k<=n
    close(surface['interval95']['lower'],root_cdf(n,k-1,.975) if k else 0.)
    close(surface['interval95']['upper'],root_cdf(n,k,.025) if k<n else 1.)
    assert sha((root/'qec/surface.dem').read_bytes())==surface['dem_sha256']
    witnesses=[surface['witness'],*surface['conditional_witnesses'],*q['tiny_exact']['witnesses'],*q['tiny_dynamic']['witnesses']]
    versions=set(); stable=[]
    for descriptor in witnesses:
        w=load(file_at(root/'qec',descriptor['file'])); versions.add(w['version'])
        assert sha(w['dem'].encode())==w['dem_sha256']==descriptor['dem_sha256']
        assert w['version']==descriptor['schema_version'] and w['seed']==descriptor['seed'] and w['shot']==descriptor['shot_or_rank']
        assert w['faults']==sorted(set(w['faults'])) and set(w['faults'])<=set(w['original_faults'])
        assert w['outcome']['logical_failure'] and w['minimization']['status']=='OneMinimal'
        # Rust serde tuple JSON: [dem_sha256, decoder struct (declared field order), faults].
        encoded=json.dumps([w['dem_sha256'],w['decoder'],w['faults']],separators=(',',':'),ensure_ascii=False).encode()
        assert sha(encoded)==descriptor['signature']
        for replay in descriptor['cross_decoder_replays']:
            assert replay['decoder']['kind'] in ['Mwpm','UnionFind']
        assert len(descriptor['cross_decoder_replays'])==2
        if w['version']==1: assert 'sampling' not in w
        elif w['version']==2: assert w['sampling']['kind'] in ['FaultCountV1','FaultCountEnumerationV1']
        elif w['version']==3: assert w['sampling']['kind']=='RobustnessV1'
        else: raise AssertionError('unsupported witness schema')
        stable.append(dict(file=descriptor['file'],signature=descriptor['signature'],version=w['version'],source_outcome=w['outcome'],replays=descriptor['cross_decoder_replays']))
    assert versions=={1,2,3}
    tiny=q['tiny_exact']; truth=.1+.03-2*.1*.03
    close(tiny['truth'],truth)
    assert tiny['decoder_calls']==8 and tiny['interval']['lower']<=truth<=tiny['interval']['upper']
    assert tiny['interval']['upper']-tiny['interval']['lower']<1e-12
    exact=load(root/'qec/tiny-exact.json'); assert exact['decoder_calls']==sum(row['decoder_calls'] for row in exact['strata'])==8
    assert all(row['mode']=='ExactEnumeration' for row in exact['strata'])
    strat=load(root/'qec/surface-stratified.json')
    assert strat['dem_sha256']==surface['dem_sha256'] and strat['decoder_calls']==surface['stratified_actual_calls']
    assert strat['declared_decoder_calls']==surface['stratified_declared_calls']==256
    assert strat['decoder_calls']==sum(row['decoder_calls'] for row in strat['strata'])<=256
    assert strat['failure_probability']==surface['stratified_interval'] and strat['unsampled_mass']==surface['omitted_mass']
    robust=load(root/'qec/tiny-robustness.json')
    for prefix in ['generating','assumed','reference']:
        assert sha(robust[prefix+'_dem'].encode())==robust[prefix+'_sha256']
    assert robust['generating_sha256']==robust['reference_sha256']!=robust['assumed_sha256']
    assert robust['process']['kind']=='MarkovBurst' and robust['options']['streams']==4 and robust['options']['steps']==32
    assert robust['decoder_calls']==256
    assert 'stream' in robust['interval_method'].lower()
    close(robust['paired_difference'],robust['assumed_rate']-robust['reference_rate'])
    for descriptor in q['tiny_dynamic']['witnesses']:
        w=load(file_at(root/'qec',descriptor['file'])); assert w['dem_sha256']==robust['assumed_sha256']
        assert sha(w['sampling']['generating_dem'].encode())==robust['generating_sha256']
    queries=[]
    assert len(o['cases'])==4
    for c in o['cases']:
        circuit=file_at(root/'observables',c['qasm_file']).read_bytes(); pauli=file_at(root/'observables',c['observable_file']).read_bytes()
        assert sha(circuit)==c['qasm_sha256'] and sha(pauli)==c['observable_sha256']
        assert sha(b'p7-query-v1\0'+len(circuit).to_bytes(8,'little')+circuit+pauli)==c['query_sha256']
        reference=c['reference']
        if c['case'].startswith('ghz'): close(reference,math.cos(.37)+.375)
        elif c['case'].startswith('product'): close(reference,math.cos(.37)**c['qubits'])
        else: assert c['case']=='local128' and c['reference_kind']=='P7-03 compact FP64 SV'
        for r in c['sparse']: close(r['value'],reference)
        assert [r['use_cones'] for r in c['sparse']]==[False,True]
        close(c['mps']['value'],reference); assert 0<=c['mps']['discarded_weight']<=1e-12
        for endpoint in ['full_sv','compact_sv']:
            r=c[endpoint]
            if r['status']=='complete': close(r['value'],reference)
            else: assert 'value' not in r and r['status'] in ['resource_limit','not_attempted']
        if c['qubits']>20: assert c['full_sv']['status']=='not_attempted'
        if c['case']=='ghz128': assert c['compact_sv']['status']=='resource_limit'
        if c['case'].startswith('product'): assert c['term_limit_control']==dict(status='resource_limit',required=16,limit=8)
        queries.append(dict(case=c['case'],query_sha256=c['query_sha256'],reference=reference,statuses=[c[e]['status'] for e in ['full_sv','compact_sv']]))
    return dict(schema_version=1,checks='PASS',ordinary_counts=dict(shots=n,failures=k,selected_shot=surface['selected_shot']),witness_versions=sorted(versions),witnesses=stable,tiny_exact_truth=truth,surface_calls=strat['decoder_calls'],dynamic_streams=robust['streams'],queries=queries,scope='independent stdlib artifact hashes, CP numeric inversion and analytic controls; native libraries validate witness generation/replay and decoding; no new coverage/speed claim')

def compare(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: compare(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b): compare(x,y)
    elif isinstance(a,float): close(a,b)
    else: assert a==b,(a,b)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run_root');p.add_argument('--compare');args=p.parse_args()
    result=check(args.run_root)
    if args.compare: compare(result,check(args.compare));result['cross_run_comparison']='PASS'
    print(json.dumps(result,indent=2))
