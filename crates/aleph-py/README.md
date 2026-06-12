# aleph-sim

**aleph** is a high-performance quantum circuit simulator written in Rust, with
pluggable backends: full state-vector (SIMD + multi-threaded), MPS (matrix
product state), and stabilizer (tableau). It is benchmarked against Qiskit Aer
(state vector + MPS) and Stim (stabilizer) — every parity-matrix cell is at or
below 1.2× its reference, most well below 1× (aleph faster); see the
[parity report](https://github.com/ruslan-splynx/aleph/blob/main/docs/perf/parity.md).

## Install

```bash
pip install aleph-sim
```

The package is named `aleph-sim` (the name `aleph` is taken on PyPI); the
Python module is still `import aleph`. Wheels: Linux x86_64 (manylinux_2_28)
and macOS arm64, Python ≥ 3.12 (abi3). They are also attached to each
[GitHub release](https://github.com/ruslan-splynx/aleph/releases).

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

## Links

- Repository: <https://github.com/ruslan-splynx/aleph>
- Benchmarks: <https://github.com/ruslan-splynx/aleph/blob/main/docs/perf/v0.1.md>
