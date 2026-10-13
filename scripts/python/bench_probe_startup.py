"""Public Python startup and SV-construction counters for P8-08.

Run interleaved fresh processes with the baseline/candidate wheel on PYTHONPATH.
The first run includes discovery/initialization; repeats run in the same interpreter.
The internal counter is optional so historical wheels remain measurable.
"""
import argparse
import json
import statistics
import time

parser = argparse.ArgumentParser()
parser.add_argument('--backend', default='sv')
parser.add_argument('--repeats', type=int, default=20)
args = parser.parse_args()
if args.repeats < 1:
    parser.error('--repeats must be positive')
start = time.perf_counter()
import aleph
from aleph import _native
import_ms = (time.perf_counter() - start) * 1000
counter = getattr(_native, '_gpu_backend_init_count', lambda: None)
before = counter()
c = aleph.Circuit(2)
c.h(0)
c.cx(0, 1)
times = []
counts = None
for _ in range(args.repeats):
    start = time.perf_counter()
    result = aleph.run(c, shots=32, seed=7, backend=args.backend)
    times.append((time.perf_counter() - start) * 1000)
    got = result.counts()
    assert set(got) <= {'00', '11'} and sum(got.values()) == 32
    assert counts is None or got == counts
    counts = got
after = counter()
print(json.dumps(dict(backend=args.backend, calls=args.repeats, import_ms=import_ms,
    first_run_ms=times[0], warm_median_ms=statistics.median(times[1:] or times),
    sv_backend_inits=None if before is None else after-before)))
