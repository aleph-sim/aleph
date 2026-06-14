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
and macOS arm64, Python ≥ 3.12 (abi3). They are also attached to each
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

## Links

- Repository: <https://github.com/aleph-sim/aleph>
- Benchmarks: <https://github.com/aleph-sim/aleph/blob/main/docs/perf/parity.md>
