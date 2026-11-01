#!/usr/bin/env python3
"""Build/run all Phase 7 entry points plus both integration studies, retaining receipts.
Usage: python run_research_instruments.py NEW_OUTPUT_ROOT [--target-dir PATH]
Build output is outside artifact subdirectories. This is correctness smoke, not a benchmark:
small fixed counts exercise schemas/workflows and do NOT replace the child coverage campaigns.
"""
import argparse,hashlib,json,os,pathlib,subprocess,sys,time

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def run(out,repo,env,label,argv):
    start=time.monotonic()
    with (out/(label+'.log')).open('w') as log:
        subprocess.run(argv,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    return dict(label=label,argv=argv,elapsed_seconds=time.monotonic()-start,exit_code=0)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output');p.add_argument('--target-dir');p.add_argument('--snapshot-provenance', help='Source hash manifest exported from the original checkout');args=p.parse_args()
    repo=pathlib.Path(__file__).resolve().parents[2];out=pathlib.Path(args.output).resolve();out.mkdir()
    env=dict(os.environ,RAYON_NUM_THREADS='1',CARGO_BUILD_JOBS=os.environ.get('CARGO_BUILD_JOBS','4'))
    if args.target_dir:env['CARGO_TARGET_DIR']=str(pathlib.Path(args.target_dir).resolve())
    snapshot=json.loads(pathlib.Path(args.snapshot_provenance).read_text()) if args.snapshot_provenance else None
    paths=list(snapshot['source_sha256']) if snapshot else subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],cwd=repo,text=True).splitlines()
    revision=snapshot['base_revision'] if snapshot else subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    # Source-only manifest includes this integration patch but excludes all existing result archives.
    sources={f:sha(repo/f) for f in paths if (f.startswith('crates/') or f.startswith('scripts/python/')) and pathlib.Path(f).suffix in ['.rs','.toml','.py']}
    sources['Cargo.lock']=sha(repo/'Cargo.lock')
    if snapshot:assert sources==snapshot['source_sha256'], 'Snapshot does not match original checkout source hashes'
    executables={};builds=[]
    specs=[('aleph-qec','decoder_witness'),('aleph-qec','rare_events'),('aleph-qec','decoder_robustness'),('aleph-qec','robustness_coverage'),('aleph-qec','deadline_replay'),('aleph-qec','research_workflow'),('aleph-sv','expectation_query'),('aleph-stab','sparse_expectation'),('aleph-stab','research_observables')]
    for package,name in specs:
        artifact=out/f'build-{name}.jsonl';start=time.monotonic()
        with artifact.open('w') as output,(out/f'build-{name}.log').open('w') as log:
            subprocess.run(['cargo','build','--release','-p',package,'--example',name,'--message-format=json'],cwd=repo,env=env,stdout=output,stderr=log,check=True)
        matches=[x for line in artifact.read_text().splitlines() if (x:=json.loads(line)).get('reason')=='compiler-artifact' and x.get('executable') and x['target']['name']==name and 'example' in x['target']['kind'] and x['manifest_path'].endswith('/'+package+'/Cargo.toml')]
        assert len(matches)==1,(name,len(matches))
        binary=pathlib.Path(matches[0]['executable']);executables[name]=str(binary)
        builds.append(dict(package=package,name=name,binary_sha256=sha(binary),fresh=matches[0]['fresh'],seconds=time.monotonic()-start))
    def exe(name,*args):return [executables[name],*map(str,args)]
    commands=[('qec-study',exe('research_workflow',out/'qec')),('observable-study',exe('research_observables',out/'observables')),('witness-capture',exe('decoder_witness','capture',out/'witness-capture')),('rare-events',exe('rare_events',out/'rare-events',8,1)),('robustness',exe('decoder_robustness',out/'robustness',4,8)),('coverage',exe('robustness_coverage',out/'coverage.json',8)),('deadlines',exe('deadline_replay','campaign',out/'deadlines',32)),('cone-query',exe('expectation_query')),('sparse-query',exe('sparse_expectation'))]
    receipts=[run(out,repo,env,label,argv) for label,argv in commands]
    for i,witness in enumerate(sorted((out/'qec').glob('*.json'))):
        obj=json.loads(witness.read_text())
        if 'faults' in obj and 'dem_sha256' in obj:receipts.append(run(out,repo,env,f'witness-replay-{i}',exe('decoder_witness','replay',witness)))
    receipts.append(run(out,repo,env,'deadline-oracle',[sys.executable,str(repo/'scripts/python/check_deadline_replay.py'),str(out/'deadlines')]))
    receipts.append(run(out,repo,env,'integration-oracle',[sys.executable,str(repo/'scripts/python/check_research_instruments.py'),str(out)]))
    for f,digest in sources.items():assert sha(repo/f)==digest,('source changed during smoke',f)
    manifest=dict(schema_version=1,base_revision=revision,source_sha256=sources,builds=builds,commands=receipts,platform=sys.platform,python=sys.version,rayon_num_threads=1,protocol='Correctness smoke only; all entry points, fixed small endpoints, no result-based retries. Each build uses its exact Cargo artifact, never filename guessing; use a fresh target for extracted snapshots. Timings reflect this smoke host and are not performance acceptance.',artifact_sha256={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file()})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',output=str(out),entry_points=len(specs),commands=len(receipts)),indent=2))
if __name__=='__main__':main()
