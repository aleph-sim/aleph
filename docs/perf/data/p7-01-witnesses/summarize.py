import itertools,json,math,statistics,pathlib
r=json.loads(pathlib.Path(__file__).with_name('remote-campaign.json').read_text())
rows=[]
for case in r['accepted'][0]['cases']:
    values={s:[v['cases'][case]['point_estimate']/1000 for v in r['accepted'] if v['source']==s] for s in ['base','patch']}
    logs=[]
    for start in range(0,len(r['accepted']),4):
        block=r['accepted'][start:start+4]
        logs.append(statistics.mean(math.log(v['cases'][case]['point_estimate']) for v in block if v['source']=='patch')-statistics.mean(math.log(v['cases'][case]['point_estimate']) for v in block if v['source']=='base'))
    # Enumerate the 4^4 resamples of complete ABBA blocks, preserving temporal pairing.
    draws=sorted(math.exp(statistics.mean(b)) for b in itertools.product(logs,repeat=len(logs)))
    row=dict(case=case,baseline_median_us=statistics.median(values['base']),patch_median_us=statistics.median(values['patch']),ratio=math.exp(statistics.mean(logs)),bootstrap95=[draws[int(.025*(len(draws)-1))],draws[int(.975*(len(draws)-1))]],block_ratios=list(map(math.exp,logs)))
    rows.append(row)
result=dict(cases=rows,accepted=len(r['accepted']),rejected=sum(not x.get('accepted',False) for x in r['events']),max_accepted_external_cpu3=max(x['external_cpu3_fraction'] for x in r['accepted']),max_accepted_sibling23=max(x['sibling23_busy'] for x in r['accepted']),feature=r['feature'],uncertainty='Percentile bootstrap of four complete ABBA blocks; descriptive small-sample uncertainty, not a universal non-regression guarantee')
pathlib.Path(__file__).with_name('remote-summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
