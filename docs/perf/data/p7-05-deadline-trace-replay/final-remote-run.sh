set -eu
replay_root=/tmp/aleph-547-508c4d0
cd "$replay_root"
test ! -e final-run-started
touch final-run-started
mkdir schema-mismatch-audit
mv campaign schema-mismatch-audit/campaign
mv campaign.log schema-mismatch-audit/campaign.log
mv campaign-load.json schema-mismatch-audit/campaign-load.json
cp final-build.log schema-mismatch-audit/failed-build.log
cp oracle.json schema-mismatch-audit/failed-oracle-output.json
tar -xzf aleph-547-followup.tar.gz -C source
python3 - <<'PY'
from pathlib import Path
import hashlib,json
root=Path('/tmp/aleph-547-508c4d0')
x=json.loads((root/'aleph-547-hashes.json').read_text())
assert x=={f:hashlib.sha256((root/'source'/f).read_bytes()).hexdigest() for f in x}
PY
cd source
CARGO_TARGET_DIR=../target /root/.cargo/bin/cargo test --release -p aleph-qec --test deadline_replay > ../final-focused-tests.log 2>&1
CARGO_TARGET_DIR=../target /root/.cargo/bin/cargo build --release -p aleph-qec --example deadline_replay --message-format=json > ../example-artifacts.jsonl 2> ../final-build.log
CARGO_TARGET_DIR=../target /root/.cargo/bin/cargo bench -p aleph-qec --bench deadline_replay --no-run --message-format=json > ../bench-artifacts.jsonl 2> ../bench-build.log
python3 ../select-binaries.py
python3 ../aleph-547-run.py
python3 scripts/python/check_deadline_replay.py ../campaign > ../oracle.json
python3 ../aleph-547-criterion.py
touch ../final-run-completed
