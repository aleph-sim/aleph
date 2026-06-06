# Phase-4 benchmark pipeline

Produces `docs/perf/phase4.md` by comparing aleph and Qiskit Aer over the shared
circuit corpus `scripts/qiskit-baseline/circuits/*.qasm` (the single source of
truth — both sides run the *same* committed QASM).

**Methodology:** single-thread on both sides, for a fair comparison against the
Stage-0 baseline. aleph runs the optimized pipeline (`run_optimized`: gate
fusion + AVX-512); Aer runs single-thread-pinned. (aleph is also
multi-thread-capable — see Phase 2 — but the report table is single-thread for
parity.) Run on an idle EPYC box; n=30 needs a 16 GiB state vector.

## 1. Aer baseline (Python, single-thread)

```
cd scripts/qiskit-baseline
.venv/bin/python run.py --workloads qft_n10,qft_n15,qft_n20,qft_n25,qft_n30
```

This regenerates the corpus (idempotent — byte-identical for an unchanged
qiskit) and times Aer, writing `results-qiskit.json`. Snapshot its QFT rows to
the committed Phase-4 Aer source:

```
cp results-qiskit.json ../../docs/perf/data/phase4-aer.json
```

(kept separate from the Stage-0 `results-qiskit.json` so that file's full-family
data is not clobbered by a QFT-only run.)

## 2. aleph timings (Rust, single-thread)

n ≤ 25 via criterion (10 samples); n=30 via single-shot `oneshot` (criterion's
10-sample minimum is prohibitive at 16 GiB / ~9 min per run):

```
RUSTFLAGS="-C target-cpu=native" RAYON_NUM_THREADS=1 \
  cargo bench -p aleph-benches --bench phase4_qft -- --sample-size 10

# n=30 single-shot (median of a couple of runs):
RUSTFLAGS="-C target-cpu=native" RAYON_NUM_THREADS=1 \
  cargo build --release -p aleph-benches --bin oneshot
RAYON_NUM_THREADS=1 ./target/release/oneshot \
  scripts/qiskit-baseline/circuits/qft_n30.qasm   # prints "elapsed_ms <ms>"
```

## 3. Extract aleph medians → unified JSON

```
python3 scripts/bench-report/extract_criterion.py \
    --criterion-root target/criterion --group phase4_qft --family qft \
    --out docs/perf/data/phase4-aleph.json
# then append the n=30 oneshot median as a {"n":30,"family":"qft",
# "aleph_ms_median":<ms>,"aleph_rsd":0.0} row.
```

## 4. Render the report

```
python3 scripts/bench-report/report.py \
    --aleph docs/perf/data/phase4-aleph.json \
    --aer   docs/perf/data/phase4-aer.json \
    --meta  docs/perf/data/phase4-meta.json \
    --out   docs/perf/phase4.md
```

`report.py` is deterministic (same inputs → byte-identical output) and
family-agnostic. Adding a family (P4-02..07): generate its corpus, add it to
`run.py` + an aleph bench, drop the two measurement snapshots into the JSONs, and
re-run step 4 — its rows appear automatically, no tooling changes.

## Adding Grover (P4-02) — worked example

Grover is *appended* to the already-measured QFT JSONs (re-timing QFT costs
hours). Run from an idle EPYC box (`uptime` ~0, no `cargo bench`/`bencher run`
in `pgrep`), single-thread both sides.

Note: the 34 MB `grover_n16_iters201.qasm` corpus is **not committed**
(gitignored — see `circuits/.gitignore`). Step (a) below regenerates the whole
grover corpus, including n=16, as a side effect (`transpile_and_export` runs for
every matrix entry before timing), so the n=16 oneshot / scaling-bench in step
(b) find the file present. If you skip step (a), run
`python scripts/qiskit-baseline/run.py --gen-only` first.

```
# (a) Aer: time only the grover keys -> results-qiskit.json (grover rows only).
#     Also (re)generates circuits/grover_n*_iters*.qasm including the n=16 blob.
cd scripts/qiskit-baseline
taskset -c 0 .venv/bin/python run.py \
    --workloads grover_n4,grover_n8,grover_n12,grover_n16
cd ../..

# (b) aleph: criterion for n=4,8,12; oneshot for n=16
RUSTFLAGS="-C target-cpu=native" RAYON_NUM_THREADS=1 \
  cargo bench -p aleph-benches --bench phase4_grover -- --sample-size 10
RUSTFLAGS="-C target-cpu=native" RAYON_NUM_THREADS=1 \
  cargo build --release -p aleph-benches --bin oneshot
RAYON_NUM_THREADS=1 ./target/release/oneshot \
  scripts/qiskit-baseline/circuits/grover_n16_iters201.qasm   # prints elapsed_ms

# (c) extract aleph grover medians to a temp file
python3 scripts/bench-report/extract_criterion.py \
    --criterion-root target/criterion --group phase4_grover --family grover \
    --out /tmp/phase4-aleph-grover.json

# (d) merge grover into the existing phase4 JSONs (preserves QFT rows), passing
#     the n=16 oneshot median as the argument:
python3 - "$ONESHOT_N16_MS" <<'PY'
import json, sys
from pathlib import Path
n16_ms = float(sys.argv[1])
base = Path("docs/perf/data")
aleph = json.loads((base/"phase4-aleph.json").read_text())
aleph["workloads"].update(
    json.loads(Path("/tmp/phase4-aleph-grover.json").read_text())["workloads"])
aleph["workloads"]["grover_n16"] = {
    "n": 16, "family": "grover", "aleph_ms_median": n16_ms, "aleph_rsd": 0.0}
(base/"phase4-aleph.json").write_text(json.dumps(aleph, indent=2) + "\n")
aer = json.loads((base/"phase4-aer.json").read_text())
grover_aer = json.loads(Path("scripts/qiskit-baseline/results-qiskit.json").read_text())
aer["workloads"].update(
    {k: v for k, v in grover_aer["workloads"].items() if v["family"] == "grover"})
(base/"phase4-aer.json").write_text(json.dumps(aer, indent=2))
print("merged grover rows:",
      sorted(k for k in aleph["workloads"] if k.startswith("grover")))
PY

# (e) re-render the report (Grover section appears automatically, sorted before QFT)
python3 scripts/bench-report/report.py \
    --aleph docs/perf/data/phase4-aleph.json \
    --aer   docs/perf/data/phase4-aer.json \
    --meta  docs/perf/data/phase4-meta.json \
    --out   docs/perf/phase4.md
```

The whole EPYC run is driven by `scripts/bench-report` plus the qiskit-baseline
harness; tests (`test_extract.py`, `test_report.py`) run with system `python3`
(stdlib `unittest`, no pytest dependency).
