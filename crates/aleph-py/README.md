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

```python
pip install "aleph-sim[sinter]"

import sinter, stim, aleph.sinter
circ = stim.Circuit.generated("surface_code:rotated_memory_x", distance=5, rounds=5,
                              after_clifford_depolarization=0.003)
stats = sinter.collect(tasks=[sinter.Task(circuit=circ)], num_workers=4,
                       decoders=["aleph-mwpm", "aleph-relay-bp-osd", "pymatching"],
                       custom_decoders=aleph.sinter.decoders(), max_shots=100_000)
```

Throughput (`scripts/python/bench_qec.py`, macOS-26.7-arm64-arm-64bit / arm,
Apple Silicon Mac; `surface d=9` reduced to 5,000 shots — aleph relay-bp-osd
is slow enough that 20,000 would take >60s — every other row is 20,000):

| DEM | decoder | shots | shots/s |
|---|---|---:|---:|
| surface d=5 | aleph mwpm | 20,000 | 2,368,604 |
| surface d=5 | aleph union-find-weighted | 20,000 | 2,822,235 |
| surface d=5 | aleph bp-osd | 20,000 | 8,513 |
| surface d=5 | aleph relay-bp-osd | 20,000 | 2,120 |
| surface d=5 | pymatching | 20,000 | 1,343,431 |
| surface d=9 | aleph mwpm | 5,000 | 137,484 |
| surface d=9 | aleph union-find-weighted | 5,000 | 188,903 |
| surface d=9 | aleph relay-bp-osd | 5,000 | 141 |
| surface d=9 | pymatching | 5,000 | 94,203 |
| color d=5 | aleph bp-osd | 20,000 | 19,519 |
| color d=5 | aleph relay-bp | 20,000 | 3,060 |
| color d=5 | aleph relay-bp-osd | 20,000 | 2,866 |

`ldpc`'s `SinterBpOsdDecoder` was installed but has no color-code row here:
its `compile_decoder_for_dem` raises `NotImplementedError` on this ldpc
version for a non-decomposed (hypergraph) DEM, so the benchmark skips it
rather than fail. aleph's dense-blossom `mwpm` and `union-find-weighted`
outrun pymatching's `decode_batch` by ~1.5–2× at both distances measured
here — `decode_batch_bit_packed` releases the GIL and parallelizes across
shots with rayon, which offsets aleph's per-shot matching algorithm being
asymptotically slower than PyMatching's Sparse Blossom (see
[docs/perf/qec-q1-mwpm.md](https://github.com/aleph-sim/aleph/blob/main/docs/perf/qec-q1-mwpm.md);
the Sparse Blossom rewrite itself is tracked in
[#331](https://github.com/aleph-sim/aleph/issues/331) and not yet done).
aleph's iterative BP-family decoders (`bp-osd`, `relay-bp`, `relay-bp-osd`)
are two to three orders of magnitude lower throughput than the one-shot
matching decoders — expected, since each shot runs multiple BP iterations
(and, for `-osd`, a post-processing ordered-statistics step) rather than a
single matching pass.

## Links

- Repository: <https://github.com/aleph-sim/aleph>
- Benchmarks: <https://github.com/aleph-sim/aleph/blob/main/docs/perf/parity.md>
