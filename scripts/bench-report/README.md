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

The whole EPYC run is driven by `scripts/bench-report` plus the qiskit-baseline
harness; tests (`test_extract.py`, `test_report.py`) run with system `python3`
(stdlib `unittest`, no pytest dependency).
