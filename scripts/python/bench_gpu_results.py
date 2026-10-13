"""One fresh process per sample; select the baseline/candidate wheel via PYTHONPATH.

Measures public run/counts and optional lazy statevector reads separately. RSS is
process peak (Linux KiB/macOS bytes), not an allocation estimate. Diagnostics are
optional for historical wheels. Use interleaved processes outside this script.
"""
import argparse
import gc
import json
import os
import resource
import sys
import time

import aleph
from aleph import _native

parser = argparse.ArgumentParser()
parser.add_argument("--backend", required=True)
parser.add_argument("--qubits", type=int, default=24)
parser.add_argument("--read", action="store_true")
args = parser.parse_args()


def stats():
    fn = getattr(_native, "_gpu_result_stats", None)
    if fn:
        m, d, h, t = fn()
        return dict(materializations=m, device_amplitude_bytes=d,
                    cached_host_bytes=h, dtoh_bytes=t)
    # Optional baseline-only diagnostic instrumentation: transfer counter only.
    fn = getattr(_native, "_gpu_dtoh_bytes", None)
    return dict(materializations=None, device_amplitude_bytes=None,
                cached_host_bytes=None, dtoh_bytes=fn() if fn else None)


def memory():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if sys.platform == "darwin" else peak * 1024
    rss_bytes = None
    if sys.platform == "linux":
        with open("/proc/self/statm") as f:
            rss_bytes = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    return dict(peak_rss_bytes=peak_bytes, rss_bytes=rss_bytes)


def stage():
    return dict(**stats(), **memory())


circuit = aleph.Circuit(args.qubits)
circuit.h(0)
circuit.s(0)
circuit.cx(0, args.qubits - 1)
before = stage()
start = time.perf_counter()
result = aleph.run(circuit, backend=args.backend, shots=32, seed=7)
counts = result.counts()
run_ms = (time.perf_counter() - start) * 1000
assert sum(counts.values()) == 32
assert set(counts) == {"0" * args.qubits, "1" + "0" * (args.qubits - 2) + "1"}
after_run = stage()
read_ms = repeat_ms = None
after_read = None
if args.read:
    start = time.perf_counter()
    amplitudes = result.statevector()
    read_ms = (time.perf_counter() - start) * 1000
    assert amplitudes.dtype.name == "complex128" and amplitudes.shape == (1 << args.qubits,)
    assert abs(amplitudes[0] - 2 ** -0.5) < 1e-6
    assert abs(amplitudes[(1 << (args.qubits - 1)) + 1] - 1j * 2 ** -0.5) < 1e-6
    del amplitudes
    start = time.perf_counter()
    amplitudes = result.statevector()
    repeat_ms = (time.perf_counter() - start) * 1000
    after_read = stage()
    del amplitudes
del result
gc.collect()
after_drop = stage()
print(json.dumps(dict(backend=args.backend, qubits=args.qubits, read=args.read,
    run_counts_ms=run_ms, first_read_ms=read_ms, repeat_read_ms=repeat_ms,
    before=before, after_run=after_run, after_read=after_read, after_drop=after_drop)))
