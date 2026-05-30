# Qiskit Aer baseline (Phase 1)

Reproducibility harness for the Phase-1 exit report
[`docs/perf/phase1.md`](../../docs/perf/phase1.md). Produces a single-thread,
same-circuit comparison between aleph and Qiskit Aer across the full matrix:

| family             | n ∈            | notes                                   |
|--------------------|----------------|-----------------------------------------|
| `ghz`              | 15, 20, 22, 25 | H + CNOT chain                          |
| `qft`              | 15, 20, 22, 25 | textbook QFT, no closing SWAPs          |
| `grover`           | 15, 20, 22, 25 | 1 marked state, 5 iterations (`_iters5`)|
| `random_brickwall` | 15, 20, 22, 25 | depth 20, deterministic angles (`_d20`) |

Both sides load the SAME committed `circuits/*.qasm` (aleph via `aleph-parser`,
Aer via `qasm3`), so gate counts are identical by construction. Circuits are
generated deterministically by `run.py` and committed; `requirements.txt` pins
the Qiskit version so a re-run reproduces them byte-for-byte (Grover's `mcx`
decomposition varies by Qiskit version; QFT/random are stable).

## Workload naming

`run.py` and `benches/benches/qiskit_baseline.rs` must agree on names:
`ghz_n{n}`, `qft_n{n}`, `grover_n{n}_iters5`, `random_brickwall_n{n}_d20`.

## Regenerating the circuits

```bash
cd scripts/qiskit-baseline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py --gen-only          # writes all 16 circuits/*.qasm, no timing
```

## Reproducing on EPYC (the authoritative measurement)

EPYC + AVX-512 is the comparison target; local numbers are not authoritative
(the Rust bench runs scalar paths on non-x86-AVX-512 hosts). Run on an
otherwise-idle host; do NOT push to `benches/**` mid-run (CI Bench shares the
runner).

```bash
# 0. Sync + build the one-shot RSS binary
ssh root@195.154.249.85
cd /tmp/aleph-forensics && git clone https://github.com/<owner>/aleph.git && cd aleph
git checkout <branch>
RUSTFLAGS="-C target-cpu=native" cargo build --release -p aleph-benches --bin oneshot

# 1. Time Qiskit Aer over the full matrix, single-thread-pinned
cd scripts/qiskit-baseline
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 0 python run.py            # -> results-qiskit.json
# (subset: python run.py --workloads qft_n25,ghz_n25)

# 2. Time aleph over the same QASM, AVX-512 verified
cd ../..
ALEPH_BENCH_FULL_MATRIX=1 RUSTFLAGS="-C target-cpu=native" \
  cargo bench -p aleph-benches --bench qiskit_baseline -- --save-baseline phase1-final

# 3. Peak RSS (per family at the headline n)
/usr/bin/time -v ./target/release/oneshot scripts/qiskit-baseline/circuits/qft_n25.qasm 2>&1 \
  | grep 'Maximum resident'
```

`ALEPH_BENCH_FULL_MATRIX=1` enables all 16 cells; without it the bench runs a
cheap CI subset (n ≤ 20, no grover) that stays under the Bench workflow's
30-minute timeout. Aer timing uses fewer runs at large n (`timing_runs_for`:
10 at n≤20, 5 at n=22, 3 at n=25); the aleph side shrinks criterion's
`sample_size` similarly (`sample_budget_for`). Both are disclosed in the
report's RSD/sample-count table.

## Reproducing locally (M-series, etc.)

Same commands minus `taskset` and `OMP_*` pinning. Local circuit generation is
deterministic and matches EPYC; only the timing numbers are non-authoritative.

## Output

- `results-qiskit.json` — Aer median/stdev + post-transpile gate counts.
- `target/criterion/**/estimates.json` — aleph medians (or the bencher.dev upload).
- The report is [`docs/perf/phase1.md`](../../docs/perf/phase1.md); the Stage-0
  snapshot it supersedes is `docs/perf/phase1-vs-qiskit.md`.
