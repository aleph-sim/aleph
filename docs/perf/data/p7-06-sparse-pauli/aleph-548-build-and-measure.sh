set -eu
cd /tmp/aleph-548-f6a7808
test ! -f measurement-started
cd source
python3 - <<'PY'
import json,pathlib,hashlib
x=json.load(open('../aleph-548-source-sha256.json'))
assert all(hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()==h for f,h in x.items())
for f in x: pathlib.Path(f).touch()
PY
# tar preserves original source mtimes, which may predate an earlier compilation.
# Delete only this package's artifacts in this isolated target before rebuilding both profiles.
CARGO_TARGET_DIR=../target /root/.cargo/bin/cargo clean -p aleph-stab --release > ../clean.log 2>&1
CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR=../target nice -n 10 /root/.cargo/bin/cargo build --release -p aleph-stab --example sparse_expectation --message-format=json > ../example-artifacts.jsonl 2> ../final-example-build.log
CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR=../target nice -n 10 /root/.cargo/bin/cargo bench -p aleph-stab --bench sparse_expectation --no-run --message-format=json > ../bench-artifacts.jsonl 2> ../final-bench-build.log
CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR=../target nice -n 10 /root/.cargo/bin/cargo test --release -p aleph-stab > ../final-full-tests.log 2>&1
cd ..
python3 - <<'PY'
import json,pathlib
for name,kind in [('example-artifacts.jsonl','example'),('bench-artifacts.jsonl','bench')]:
    artifacts=[json.loads(s) for s in pathlib.Path(name).read_text().splitlines()]
    target=[a for a in artifacts if a.get('reason')=='compiler-artifact' and a.get('executable') and a['target']['name']=='sparse_expectation' and kind in a['target']['kind']]
    assert len(target)==1 and target[0]['fresh'] is False,(name,target)
    libraries=[a for a in artifacts if a.get('reason')=='compiler-artifact' and a['target']['name']=='aleph_stab' and 'lib' in a['target']['kind']]
    assert len(libraries)==1 and libraries[0]['fresh'] is False,(name,libraries)
print('both target executables rebuilt, fresh=false')
PY
touch measurement-started
python3 aleph-548-run.py
python3 aleph-548-summary.py .
