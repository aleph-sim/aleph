# Qiskit Aer baseline (Phase 1, Stage 0)

Reproducibility harness for `docs/perf/phase1-vs-qiskit.md`. Produces a
single-thread, same-circuit comparison between aleph and Qiskit Aer across
QFT-20, Grover-20 (10 iters), and random-brickwall-20.

## Reproducing on EPYC

```bash
# 1. Time Qiskit Aer
ssh root@195.154.249.85
cd /tmp/aleph-forensics                 # NOT the GH Actions runner workdir
git clone https://github.com/<owner>/aleph.git && cd aleph
git checkout <branch>
cd scripts/qiskit-baseline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 0 python run.py
# Produces results-qiskit.json and writes circuits/*.qasm if missing.

# 2. Time aleph against the same QASM files
cd ../..
RUSTFLAGS="-C target-cpu=native" cargo bench \
  --bench qiskit_baseline -- --save-baseline phase1-baseline
```

## Reproducing locally (M-series, Linux, etc.)

Same commands minus `taskset` and `OMP_*` pinning, but **note**: local numbers
are not authoritative. EPYC + AVX-512 is the comparison target. The Rust bench
runs scalar code paths on non-x86-AVX-512 hosts.

## Circuits

The three workloads:

- `circuits/qft_n20.qasm` — Nielsen-Chuang § 5.1 QFT, no closing SWAPs.
- `circuits/grover_n20_iters10.qasm` — Grover-20, 1 marked state (|0…01⟩), 10 iterations.
- `circuits/random_brickwall_n20_d20.qasm` — brick-wall random circuit (Rz/Rx 1q
  layers + alternating CNOT pairs), depth 20, deterministic angles
  `cos(layer + qubit*0.37)`.

All three are transpiled by Qiskit to the basis
`[h, x, z, rz, rx, ry, cx, cz, ccx, p]` at `optimization_level=0` so we measure
engines, not the transpiler.

## Updating the QASM files

`run.py` regenerates `circuits/*.qasm` deterministically on every run. To
refresh after a Qiskit version bump, simply re-run; commit the diff.
