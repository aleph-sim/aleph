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
thread:

| DEM | decoder | threads | shots | shots/s |
|---|---|---:|---:|---:|
| surface d=5 | aleph mwpm | 10 | 20,000 | 2,196,645 |
| surface d=5 | aleph mwpm | 1 | 20,000 | 733,746 |
| surface d=5 | aleph union-find-weighted | 10 | 20,000 | 2,476,563 |
| surface d=5 | aleph union-find-weighted | 1 | 20,000 | 576,536 |
| surface d=5 | aleph bp-osd | 10 | 20,000 | 18,555 |
| surface d=5 | aleph bp-osd | 1 | 20,000 | 2,987 |
| surface d=5 | aleph relay-bp-osd | 10 | 20,000 | 4,530 |
| surface d=5 | aleph relay-bp-osd | 1 | 20,000 | 756 |
| surface d=5 | pymatching | 1 | 20,000 | 1,503,434 |
| surface d=9 | aleph mwpm | 10 | 5,000 | 202,386 |
| surface d=9 | aleph mwpm | 1 | 5,000 | 33,684 |
| surface d=9 | aleph union-find-weighted | 10 | 5,000 | 392,584 |
| surface d=9 | aleph union-find-weighted | 1 | 5,000 | 82,226 |
| surface d=9 | aleph relay-bp-osd | 10 | 5,000 | 323 |
| surface d=9 | aleph relay-bp-osd | 1 | 5,000 | 55 |
| surface d=9 | pymatching | 1 | 5,000 | 204,928 |
| color d=5 | aleph bp-osd | 10 | 20,000 | 27,827 |
| color d=5 | aleph bp-osd | 1 | 20,000 | 4,628 |
| color d=5 | aleph relay-bp | 10 | 20,000 | 6,042 |
| color d=5 | aleph relay-bp | 1 | 20,000 | 1,047 |
| color d=5 | aleph relay-bp-osd | 10 | 20,000 | 5,729 |
| color d=5 | aleph relay-bp-osd | 1 | 20,000 | 1,040 |

**Per core, pymatching is faster.** Single-threaded, aleph's dense-blossom
`mwpm` reaches 0.49× pymatching's throughput at d=5 and 0.16× at d=9, and
`union-find-weighted` 0.38× and 0.40×: aleph's per-shot matching algorithm is
asymptotically slower than PyMatching's Sparse Blossom (see
[docs/perf/qec-q1-mwpm.md](https://github.com/aleph-sim/aleph/blob/main/docs/perf/qec-q1-mwpm.md); the Sparse Blossom rewrite is tracked in
[#331](https://github.com/aleph-sim/aleph/issues/331) and not yet done).
**On the whole machine, aleph is faster or level.** With all 10 cores
against pymatching's one thread, `mwpm` is 1.46× pymatching at d=5 and
level (0.99×) at d=9, and `union-find-weighted` is 1.65× and 1.92×.
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
