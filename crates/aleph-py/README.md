# aleph-sim

**aleph** is a high-performance quantum circuit simulator written in Rust, with
pluggable backends: full state-vector (SIMD + multi-threaded), MPS (matrix
product state), and stabilizer (tableau). It is benchmarked against Qiskit Aer
(state vector) and Stim (stabilizer); see the
[v0.1 benchmark report](https://github.com/ruslan-splynx/aleph/blob/main/docs/perf/v0.1.md).

## Install

For v0.1, install the wheel from the
[GitHub release](https://github.com/ruslan-splynx/aleph/releases) — PyPI
publication is planned. The package is named `aleph-sim` (the name `aleph` is
taken on PyPI); the Python module is still `import aleph`.

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
