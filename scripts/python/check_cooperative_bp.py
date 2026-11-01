#!/usr/bin/env python3
"""P8-10 independent receipts/paired-block/tail checker. Usage: CAMPAIGN_DIRECTORY.
Native CPU/GPU oracles establish decoding; this script validates retained identities,
mask checksums, counts and every timing row. Tail ranges are descriptive, not CIs.
"""
import csv
import json
import math
import pathlib
import statistics
import sys


def quantile(values, q):
    ordered=sorted(values)
    return ordered[math.ceil(q*len(ordered))-1]


def check(root):
    manifest=json.loads((root/'manifest.json').read_text())
    records=manifest['measurements']
    assert len(records)==576 and len(manifest['cells'])==22
    assert len({r['label'] for r in records})==576
    observations=0
    oracles={}
    for model,n,p in manifest['cells']:
        oracle=json.loads((root/'oracles'/f'{model}-n{n}-p{p}.json').read_text())
        assert oracle['seed']==560 and oracle['shots']==n and len(oracle['masks'])==n
        assert oracle['alpha']==.875 and oracle['max_iter']==64
        assert 0<=oracle['invalid']<=n and 0<=oracle['logical_failures']<=n
        assert all(len(oracle[k])==64 and int(oracle[k],16)>=0 for k in ['dem_sha256','input_sha256'])
        checksum=0
        assert all(isinstance(mask,int) and 0<=mask<(1<<64) for mask in oracle['masks'])
        for mask in oracle['masks']:checksum=(checksum*31+mask)&((1<<64)-1)
        oracles[(model,str(n),str(p))]=(oracle,checksum)
    for record in records:
        with (root/(record['label']+'.csv')).open() as f:assert list(csv.DictReader(f))==record['rows']
        model,n,p,repeats,mode=record['argv'][1:6]
        oracle,checksum=oracles[(model,n,p)]
        assert len(record['rows'])==int(repeats)
        metadata=dict(word.split('=',1) for word in (root/(record['label']+'.log')).read_text().split() if '=' in word)
        for name in ['dem_sha256','input_sha256']:assert metadata[name]==oracle[name]
        assert int(metadata['invalid'])==oracle['invalid'] and int(metadata['logical_failures'])==oracle['logical_failures']
        assert int(metadata['per_shot_bytes'])*int(n)<=int(metadata['retained_scratch_bytes'])<=2<<30
        for i,row in enumerate(record['rows']):
            assert (row['model'],row['shots'],row['rate'],row['mode'],row['repeat'])==(model,n,p,mode,str(i))
            assert int(row['ns'])>0 and int(row['checksum'])==checksum
        observations+=len(record['rows'])
    assert observations==10512
    cells=[]
    for model,n,p in manifest['cells']:
        rows=[r for r in records if not r['label'].startswith('control') and r['argv'][1:4]==[model,str(n),str(p)]]
        stats={}
        for mode in ['shot','warp','coop']:
            values=[int(t['ns']) for r in rows if r['argv'][5]==mode for t in r['rows']]
            assert len(values)==8*(64 if n==1 else 3)
            stats[mode]=dict(median_ns=statistics.median(values),p95_ns=quantile(values,.95),p99_ns=quantile(values,.99),max_ns=max(values))
        ratios=[]
        for block in range(4):
            medians={}
            for mode in stats:
                values=[int(t['ns']) for r in rows if r['label'].startswith(f'b{block}-') and r['argv'][5]==mode for t in r['rows']]
                assert len(values)==2*(64 if n==1 else 3)
                medians[mode]=statistics.median(values)
            ratios.append(min(medians['shot'],medians['warp'])/medians['coop'])
        speedup=min(stats['shot']['median_ns'],stats['warp']['median_ns'])/stats['coop']['median_ns']
        held_out=model=='surface11' or n in [33,4097] or p==.001
        cells.append(dict(model=model,shots=n,rate=p,stats=stats,speedup_best_reference=speedup,block_speedup_best_reference=ratios,held_out=held_out,meets_go=held_out and speedup>=1.2 and min(ratios)>1,invalid=oracles[(model,str(n),str(p))][0]['invalid'],logical_failures=oracles[(model,str(n),str(p))][0]['logical_failures']))
    controls=[]
    for model in ['surface7','surface11','gross']:
        ratios=[]
        for block in range(4):
            arms={arm:[int(t['ns']) for r in records if r['label'].startswith(f'control-b{block}-{model}-') and r['label'].endswith(arm) for t in r['rows']] for arm in ['original','shot']}
            assert all(len(a)==6 for a in arms.values())
            ratios.append(statistics.median(arms['original'])/statistics.median(arms['shot']))
        controls.append(dict(model=model,original_over_current_reference=ratios))
    return dict(status='PASS',observations=observations,invocations=576,verdict='GO_OPT_IN' if any(c['meets_go'] for c in cells) else 'NO_GO',cells=cells,original_controls=controls,scope='fixed observations and four paired blocks; no statistical confidence or universal speed guarantee; CPU outcomes on identical fault draws are unchanged')


if __name__=='__main__':
    print(json.dumps(check(pathlib.Path(sys.argv[1])),indent=2))
