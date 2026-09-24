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
(best of two runs per cell; every other row is a single run):

| DEM | decoder | threads | shots | shots/s |
|---|---|---:|---:|---:|
| surface d=5 | aleph mwpm | 10 | 20,000 | 914,767 |
| surface d=5 | aleph mwpm | 1 | 20,000 | 654,255 |
| surface d=5 | aleph union-find-weighted | 10 | 20,000 | 2,912,569 |
| surface d=5 | aleph union-find-weighted | 1 | 20,000 | 567,173 |
| surface d=5 | aleph bp-osd | 10 | 20,000 | 19,154 |
| surface d=5 | aleph bp-osd | 1 | 20,000 | 3,005 |
| surface d=5 | aleph relay-bp-osd | 10 | 20,000 | 4,635 |
| surface d=5 | aleph relay-bp-osd | 1 | 20,000 | 792 |
| surface d=5 | pymatching | 1 | 20,000 | 1,539,142 |
| surface d=9 | aleph mwpm | 10 | 5,000 | 431,448 |
| surface d=9 | aleph mwpm | 1 | 5,000 | 100,976 |
| surface d=9 | aleph union-find-weighted | 10 | 5,000 | 409,647 |
| surface d=9 | aleph union-find-weighted | 1 | 5,000 | 82,500 |
| surface d=9 | aleph relay-bp-osd | 10 | 5,000 | 339 |
| surface d=9 | aleph relay-bp-osd | 1 | 5,000 | 60 |
| surface d=9 | pymatching | 1 | 5,000 | 214,667 |
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
reaches 0.43× pymatching's throughput at d=5 (was 0.49× with the old
dense-blossom matcher — essentially unchanged, small syndromes are dominated
by per-shot fixed costs) and **0.47× at d=9 (was 0.16×, a ~2.9× improvement)**
— consistent with the criterion benchmark in the perf record, where the
sparse/dense ratio *grows* with distance because the new matcher's cost
scales with the defect count rather than `O(D²)`. `union-find-weighted` (an
unrelated decoder, unaffected by this rewrite) is 0.37× and 0.38×, roughly
where it was before, included as a stable cross-check that these are real
algorithm effects and not machine noise. **On the whole machine, the honest
result is mixed: `mwpm` now wins clearly at d=9 (2.01× pymatching's one
thread, up from 0.99× before) but no longer wins at d=5 (0.59×, down from
1.46× before).** The d=5 regression reproduced across two full runs (10-core
`mwpm` at d=5: 743,079 and 914,767 shots/s — noisy, but both well under the
old 2,196,645), while `union-find-weighted` at the same cell stayed flat
(2,516,185 / 2,912,569 vs. the old 2,476,563), which rules out a machine-load
explanation. The likely cause is architectural, not yet profiled down: unlike
the old per-thread-cheap dense blossom, the sparse matcher's per-shot state is
drawn from one process-wide `Mutex<Vec<State>>` pool
(`crates/aleph-qec/src/sparse_blossom/mod.rs`), locked twice per decode: at
d=5 (~9.5 defects) each decode is fast enough that ten rayon threads
contending on that single mutex plausibly dominates the actual matching work,
whereas at d=9 (~56 defects) the matching itself is large enough that the
lock is no longer the bottleneck. Sharding the pool (e.g. thread-local or
per-rayon-worker) is a candidate follow-up, not attempted here.
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

## Links

- Repository: <https://github.com/aleph-sim/aleph>
- Benchmarks: <https://github.com/aleph-sim/aleph/blob/main/docs/perf/parity.md>
