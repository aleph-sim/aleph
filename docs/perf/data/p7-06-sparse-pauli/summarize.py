import pathlib,json,statistics,sys,re
root=pathlib.Path(sys.argv[1]);rows=[json.loads(x) for x in (root/'campaign.log').read_text().splitlines()];cases=[]
for name in sorted({x['case'] for x in rows}):
 modes=[]
 for mode in sorted({x['result']['mode'] for x in rows if x['case']==name}):
  r=[x for x in rows if x['case']==name and x['result']['mode']==mode];assert len(r)==10
  reference=r[0]['reference'];assert all(abs(x['result']['value']-reference)<1e-10 for x in r)
  cold=json.loads((root/'rss'/f'{name}-{mode}.json').read_text());assert abs(cold['result']['value']-reference)<1e-10
  costs=[x['result']['total_ns'] for x in r];rss=(root/'rss'/f'{name}-{mode}.time').read_text();peak=int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',rss)[1])
  stats={k:v for k,v in r[0]['result'].items() if k not in ['total_ns','planning_ns','execution_ns']}
  modes.append(dict(mode=mode,median_total_ns=statistics.median(costs),min_total_ns=min(costs),max_total_ns=max(costs),cold_peak_rss_kib=peak,stats=stats))
 cases.append(dict(name=name,reference=reference,modes=modes))
(root/'summary.json').write_text(json.dumps(dict(protocol='one warm oracle check per case/API; five forward/reverse paired blocks, ten observations per API, no result-based retries; total API timing includes compilation/allocation/execution/release/JSON adapter; cold RSS is separate process',oracle_tolerance=1e-10,observations=len(rows),cases=cases),indent=2)+'\n')
print('oracle and summary PASS',len(rows),'observations')
for c in cases:print(c['name'],[(r['mode'],round(r['median_total_ns']/1000,2),r['stats'].get('peak_terms'),r['cold_peak_rss_kib']) for r in c['modes']])
