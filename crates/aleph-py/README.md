# aleph-sim

**aleph** is a high-performance quantum circuit simulator written in Rust, with
pluggable backends: full state-vector (SIMD + multi-threaded), MPS (matrix
product state), and stabilizer (tableau). It is benchmarked against Qiskit Aer
(state vector + MPS) and Stim (stabilizer) — every parity-matrix cell is at or
below 1.2× its reference, most well below 1× (aleph faster); see the
[parity report](https://github.com/aleph-sim/aleph/blob/main/docs/perf/parity.md).

## Install

```bash
pip install aleph-sim
```

The package is named `aleph-sim` (the name `aleph` is taken on PyPI); the
Python module is still `import aleph`. Wheels: Linux x86_64 (manylinux_2_28)
and macOS arm64, Python ≥ 3.10 (abi3). They are also attached to each
[GitHub release](https://github.com/aleph-sim/aleph/releases).

## Quickstart

```python
import aleph

c = aleph.Circuit(2)
c.h(0)
c.cx(0, 1)

result = aleph.run(c, shots=1024, seed=0)
print(result.counts())        # {'00': ~512, '11': ~512}
print(result.statevector())   # 4 amplitudes
```

## Threads

Since v0.3 (P3-13) the wheels link rayon: wide-bond MPS operations use a
thread pool sized to the visible CPUs (small-bond operations always run
sequentially via a size threshold). Set `RAYON_NUM_THREADS` to bound the pool,
e.g. in cgroup-limited containers where the visible CPU count overstates the
quota.

## Noise

```python
import aleph

c = aleph.Circuit(2)
c.h(0); c.cx(0, 1)

nm = aleph.NoiseModel()
nm.add_all_qubit_quantum_error(aleph.depolarizing_error(0.01, 1), ["h"])
nm.add_quantum_error(aleph.depolarizing_error(0.02, 2), ["cx"], [0, 1])
nm.add_readout_error([[0.98, 0.02], [0.03, 0.97]], 0)

print(aleph.run(c, shots=100_000, noise=nm, seed=7).counts())
```

Error factories mirror Qiskit Aer names (`depolarizing_error`, `amplitude_damping_error`, `phase_damping_error`, `pauli_error`, `bit_flip_error`, `phase_flip_error`). Noise runs on the state-vector backend as per-shot Monte-Carlo trajectories. Attach errors by Aer gate mnemonic (`"h"`, `"cx"`); unknown names raise `ValueError`.

## QEC decoding

aleph ships MWPM, Union-Find, BP, BP-OSD and relay-BP decoders for stim
Detector Error Models, usable directly or through sinter:

```bash
pip install "aleph-sim[sinter]"
```

```python
import sinter, stim, aleph.sinter
circ = stim.Circuit.generated("surface_code:rotated_memory_x", distance=5, rounds=5,
                              after_clifford_depolarization=0.003)
stats = sinter.collect(tasks=[sinter.Task(circuit=circ)], num_workers=4,
                       decoders=["aleph-mwpm", "aleph-relay-bp-osd", "pymatching"],
                       custom_decoders=aleph.sinter.decoders(), max_shots=100_000)
```

aleph's batch decode parallelizes across shots on all cores, so under
`sinter.collect(num_workers=N)` set `RAYON_NUM_THREADS=1` (or a small number)
to avoid running `num_workers × cores` threads.

Throughput (`scripts/python/bench_qec.py`, macOS-26.7-arm64-arm-64bit / arm,
Apple Silicon Mac, 10 logical CPUs; `surface d=9` reduced to 5,000 shots —
aleph relay-bp-osd is slow enough that 20,000 would take >60s — every other
row is 20,000). Every aleph decoder is measured on all 10 cores and on one
thread (`RAYON_NUM_THREADS=1`); pymatching's `decode_batch` runs on one
thread. `mwpm` rows below are post-[Sparse Blossom](https://github.com/aleph-sim/aleph/blob/main/docs/perf/q1-03b-sparse-blossom.md)
(best of two runs per cell; `pymatching` rows are also best of two, from
this same session; every other row is a single run from the Task 8 session):

| DEM | decoder | threads | shots | shots/s |
|---|---|---:|---:|---:|
| surface d=5 | aleph mwpm | 10 | 20,000 | 2,925,616 |
| surface d=5 | aleph mwpm | 1 | 20,000 | 637,482 |
| surface d=5 | aleph union-find-weighted | 10 | 20,000 | 2,912,569 |
| surface d=5 | aleph union-find-weighted | 1 | 20,000 | 567,173 |
| surface d=5 | aleph bp-osd | 10 | 20,000 | 19,154 |
| surface d=5 | aleph bp-osd | 1 | 20,000 | 3,005 |
| surface d=5 | aleph relay-bp-osd | 10 | 20,000 | 4,635 |
| surface d=5 | aleph relay-bp-osd | 1 | 20,000 | 792 |
| surface d=5 | pymatching | 1 | 20,000 | 1,510,493 |
| surface d=9 | aleph mwpm | 10 | 5,000 | 552,252 |
| surface d=9 | aleph mwpm | 1 | 5,000 | 101,399 |
| surface d=9 | aleph union-find-weighted | 10 | 5,000 | 409,647 |
| surface d=9 | aleph union-find-weighted | 1 | 5,000 | 82,500 |
| surface d=9 | aleph relay-bp-osd | 10 | 5,000 | 339 |
| surface d=9 | aleph relay-bp-osd | 1 | 5,000 | 60 |
| surface d=9 | pymatching | 1 | 5,000 | 211,187 |
| color d=5 | aleph bp-osd | 10 | 20,000 | 27,962 |
| color d=5 | aleph bp-osd | 1 | 20,000 | 4,702 |
| color d=5 | aleph relay-bp | 10 | 20,000 | 6,115 |
| color d=5 | aleph relay-bp | 1 | 20,000 | 1,080 |
| color d=5 | aleph relay-bp-osd | 10 | 20,000 | 6,001 |
| color d=5 | aleph relay-bp-osd | 1 | 20,000 | 1,074 |

**Per core, pymatching is still faster, but the [Sparse Blossom
rewrite](https://github.com/aleph-sim/aleph/blob/main/docs/perf/q1-03b-sparse-blossom.md)
(closing [#331](https://github.com/aleph-sim/aleph/issues/331)) narrows the
gap sharply at the distance where it matters.** Single-threaded, `mwpm`
reaches 0.42× pymatching's throughput at d=5 (was 0.49× with the old
dense-blossom matcher — essentially unchanged, small syndromes are dominated
by per-shot fixed costs) and **0.48× at d=9 (was 0.16×, a ~3× improvement)**
— consistent with the criterion benchmark in the perf record, where the
sparse/dense ratio *grows* with distance because the new matcher's cost
scales with the defect count rather than `O(D²)`. **On the whole machine,
`mwpm` now wins outright at both distances: 1.94× pymatching's one thread at
d=5 (2,925,616 vs. 1,510,493 shots/s) and 2.62× at d=9 (552,252 vs.
211,187).** The d=9 10-thread number itself also rose across the fix below
(431,448 → 552,252, +28%), so part of that 2.62× reflects the rise rather
than a pure ratio-vs-pymatching improvement; the cause of the d=9 rise is
unconfirmed (see below).

An earlier build of this rewrite regressed multi-thread throughput at small
distances: 10-core `mwpm` at d=5 measured 743,079 and 914,767 shots/s across
two full runs, well under the old dense-blossom matcher's 2,196,645, while
`union-find-weighted` (an unrelated decoder, measured in the same runs)
stayed flat (2,516,185 / 2,912,569 vs. its own old 2,476,563), ruling out a
machine-load explanation. The cause: the sparse matcher's per-shot scratch
state came from one process-wide `Mutex<Vec<State>>` pool
(`crates/aleph-qec/src/sparse_blossom/mod.rs`), locked twice per decode. At
d=5 (~9.5 defects, ~1.5 µs/decode) ten rayon threads contending on that
single mutex dominated the actual matching work; at d=9 (~56 defects, ~10 µs
decode) the matching itself was large enough that the lock wasn't the
bottleneck, which is why only the small-distance cell regressed. The fix
replaced the mutex pool with a thread-local cache: each `SparseMatcher` gets
a unique id at construction, and `decode` finds-or-allocates its `State` in
the *calling thread's own* `Vec`, so the hot path never takes a lock. That
took 10-core `mwpm` at d=5 from 914,767 to 2,925,616 shots/s (best of two
runs on an idle box) — above the old dense matcher's level — while
single-thread throughput at both distances held steady within noise
(d=5: 654,255 → 637,482; d=9: 100,976 → 101,399). The d=9 *10-thread* number
also rose (431,448 → 552,252, +28%), even though d=9's ~10 µs decodes were
never expected to be lock-bound (see the "Thread-local state cache" section
of [the perf record](https://github.com/aleph-sim/aleph/blob/main/docs/perf/q1-03b-sparse-blossom.md))
— this may be the
same fix (removing the mutex's cache-line ping-pong helps even when it isn't
the dominant cost) or may just be run-to-run variance on a shared dev box;
it was not re-measured against the pre-fix build under controlled conditions,
so it's reported honestly rather than folded into the fix's headline claim.

`decode_batch_bit_packed` releases the GIL and splits shots across cores, so
this is what a single-process caller sees; a sinter run with many workers
already uses the cores and gets the per-core ratio instead.

`ldpc`'s `SinterBpOsdDecoder` was installed but has no color-code row here:
its `compile_decoder_for_dem` raises `NotImplementedError` on this ldpc
version for a non-decomposed (hypergraph) DEM, so the benchmark skips it
rather than fail. aleph's iterative BP-family decoders (`bp-osd`, `relay-bp`,
`relay-bp-osd`) are two to three orders of magnitude lower throughput than
the one-shot matching decoders — expected, since each shot runs multiple BP
iterations (and, for `-osd`, a post-processing ordered-statistics step)
rather than a single matching pass.

## Using aleph decoders from CUDA-Q QEC

`aleph.cudaq` registers every aleph decoder inside NVIDIA's `cudaq_qec`, so a
CUDA-Q / NVQLink workflow can A/B them against `nv-qldpc-decoder`,
`nv-fusion-decoder` or `pymatching` without leaving its harness:

```bash
pip install "aleph-sim[cudaq]"      # pulls cudaq-qec (cu12/cu13 auto-selected) and scipy
```

```python
import numpy as np, stim, cudaq_qec as qec
import aleph.qec, aleph.cudaq                       # the import registers aleph-* decoders

circ = stim.Circuit.generated("surface_code:rotated_memory_x", distance=5, rounds=5,
                              after_clifford_depolarization=0.003)
H, O, rates = aleph.cudaq.dem_to_matrices(circ.detector_error_model(decompose_errors=True))
dets, obs = circ.compile_detector_sampler().sample(10_000, separate_observables=True)

dec = qec.get_decoder("aleph-mwpm", H, O=O, error_rate_vec=rates)    # or "aleph-relay-bp-osd", ...
res = dec.decode_batch(dets.astype(float).tolist())
pred = ((res.result > 0.5).astype(np.uint8) @ O.T) % 2
print("logical error rate:", (pred != obs).any(axis=1).mean())
```

Names: `aleph-mwpm`, `aleph-union-find`, `aleph-union-find-weighted`, `aleph-bp`,
`aleph-bp-osd`, `aleph-relay-bp`, `aleph-relay-bp-osd`. Keyword parameters are
those of `aleph.qec.Decoder` (`legs`, `alpha`, `gamma_min`, `gamma_max`, `seed`,
`osd_order`, `max_iter`); priors come from `error_rate_vec` (cudaq fills it in
when you pass a DEM string) or a scalar `error_rate`. Results are per-column
error estimates like cudaq's own decoders, so `O @ ê` gives the observable flips.

Matching decoders need a graph-like `H` (≤ 2 ones per column). cudaq's DEM
parser keeps a `^`-decomposed hyperedge as one column, so for `aleph-mwpm` /
`aleph-union-find*` pass `dem_to_matrices(dem)` rather than the DEM string
(the BP family takes either). For the gross [[144,12,12]] code, which cudaq has
no built-in for, `aleph.qec.gross_code_dem(rounds, p)` gives the circuit-level
model. A/B numbers against NVIDIA's decoders: `docs/perf/f6-cudaq-plugin.md`.

## Links

- Repository: <https://github.com/aleph-sim/aleph>
- Benchmarks: <https://github.com/aleph-sim/aleph/blob/main/docs/perf/parity.md>
