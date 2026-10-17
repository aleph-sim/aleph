"""Pinned, sequential ABBA campaign; retain every rejected run and CPU guard receipt."""
import hashlib,json,os,pathlib,resource,subprocess,time,statistics
ROOT=pathlib.Path('/tmp/aleph-543-816ef96')
OUT=ROOT/'campaign';OUT.mkdir(exist_ok=True)
HZ=os.sysconf('SC_CLK_TCK');CPUS=[3,23]
def cpu():
    rows={}
    for line in pathlib.Path('/proc/stat').read_text().splitlines():
        key,*vals=line.split()
        if key in ['cpu3','cpu23']:
            v=list(map(int,vals));rows[int(key[3:])]=(sum(v[:8]),sum(v[:8])-v[3]-v[4])
    return rows
def usage(a,b,c):
    dt=(b[c][0]-a[c][0])/HZ
    busy=(b[c][1]-a[c][1])/HZ
    return dt,busy

def executable(source,bench):
    matches=list((ROOT/f'target-{source}/release/deps').glob(bench+'-*'))
    return next(p for p in matches if p.is_file() and p.suffix=='' and os.access(p,os.X_OK))
binaries={k:executable(k,'dem_experiment') for k in ['base','patch']}
binaries['feature']=executable('patch','failure_witness')
events=[];accepted=[];attempt=0
filter_re=r'dem_experiment/(sample_1024|mwpm_d3_total_1024|null_total/1024)$'
for slot,source in enumerate(['base','patch','patch','base']*4):
    for retry in range(30):
        attempt+=1;a=cpu();time.sleep(1);b=cpu()
        pre={str(c):usage(a,b,c)[1]/max(usage(a,b,c)[0],.01) for c in CPUS}
        if max(pre.values())>.02:
            events.append(dict(attempt=attempt,slot=slot,source=source,rejected='preflight',cpu_busy=pre))
            (OUT/'events.json').write_text(json.dumps(events,indent=2))
            continue
        directory=OUT/f'{attempt}-{source}';directory.mkdir()
        env=dict(os.environ,RAYON_NUM_THREADS='1',ALEPH_BENCH_OUTPUT=str(directory))
        a=cpu();t=time.monotonic();r0=resource.getrusage(resource.RUSAGE_CHILDREN)
        p=subprocess.run(['taskset','-c','3',str(binaries[source]),filter_re,'--bench'],env=env,capture_output=True,text=True)
        r1=resource.getrusage(resource.RUSAGE_CHILDREN);elapsed=time.monotonic()-t;b=cpu()
        own=(r1.ru_utime+r1.ru_stime)-(r0.ru_utime+r0.ru_stime)
        primary=max(0,usage(a,b,3)[1]-own)/elapsed
        sibling=usage(a,b,23)[1]/max(usage(a,b,23)[0],.01)
        (directory/'stdout.log').write_text(p.stdout+p.stderr)
        assert p.returncode==0,(p.returncode,p.stderr)
        rec=dict(attempt=attempt,slot=slot,source=source,elapsed_s=elapsed,own_cpu_s=own,external_cpu3_fraction=primary,sibling23_busy=sibling,preflight=pre)
        rec['accepted']=primary<=.03 and sibling<=.03
        events.append(rec);(OUT/'events.json').write_text(json.dumps(events,indent=2))
        if rec['accepted']:
            rec['cases']={str(p.parent.parent.relative_to(directory)):json.loads(p.read_text())['median'] for p in directory.glob('dem_experiment/**/new/estimates.json')}
            assert len(rec['cases'])==3,rec
            accepted.append(rec);break
    else:raise RuntimeError('unable to find bounded quiet-core slot')

# Explicit opt-in feature costs, with the same affinity and idle SMT sibling.
for retry in range(30):
    a=cpu();time.sleep(1);b=cpu()
    if max(usage(a,b,c)[1]/max(usage(a,b,c)[0],.01) for c in CPUS)>.02:continue
    directory=OUT/f'feature-{retry}';env=dict(os.environ,RAYON_NUM_THREADS='1',ALEPH_BENCH_OUTPUT=str(directory))
    a=cpu();t=time.monotonic();r0=resource.getrusage(resource.RUSAGE_CHILDREN)
    p=subprocess.run(['taskset','-c','3',str(binaries['feature']),'--bench'],env=env,capture_output=True,text=True)
    r1=resource.getrusage(resource.RUSAGE_CHILDREN);elapsed=time.monotonic()-t;b=cpu()
    assert p.returncode==0,p.stderr
    (OUT/f'feature-{retry}.log').write_text(p.stdout+p.stderr)
    ext=max(0,usage(a,b,3)[1]-(r1.ru_utime+r1.ru_stime-r0.ru_utime-r0.ru_stime))/elapsed
    sib=usage(a,b,23)[1]/max(usage(a,b,23)[0],.01)
    events.append(dict(feature=retry,external_cpu3_fraction=ext,sibling23_busy=sib,accepted=ext<=.03 and sib<=.03))
    if ext<=.03 and sib<=.03:
        feature={str(p.parent.parent.relative_to(directory)):json.loads(p.read_text())['median'] for p in directory.glob('failure_witness/**/new/estimates.json')};break
else:raise RuntimeError('no quiet feature slot')
report=dict(protocol='Four ABBA blocks on CPU3, SMT sibling23 guarded; 2% one-second preflight and <=3% run overlap gates; RAYON_NUM_THREADS=1; services retained',accepted=accepted,events=events,feature=feature,binaries={k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in binaries.items()},cpuinfo=pathlib.Path('/proc/cpuinfo').read_text().split('\n\n')[0],rustc=subprocess.check_output(['/root/.cargo/bin/rustc','--version'],text=True).strip())
(OUT/'report.json').write_text(json.dumps(report,indent=2))
print('DONE',len(accepted),'accepted',len(events)-len(accepted)-1,'rejected')
for case in accepted[0]['cases']:
    med={source:statistics.median(r['cases'][case]['point_estimate'] for r in accepted if r['source']==source)/1000 for source in ['base','patch']}
    print(case,med,'patch/base',med['patch']/med['base'])
print('FEATURE',feature)
