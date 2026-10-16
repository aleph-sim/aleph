"""One isolated Python process. Explicit hardware failure is always fatal."""
import time
START = time.perf_counter()
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys


def main():
    if not __debug__:
        raise RuntimeError('hardware correctness guards require Python without -O')
    request = json.loads(Path(sys.argv[1]).read_text())
    s, arm = request['spec'], request['arm']
    import numpy as np
    import aleph
    from aleph import _native
    startup_ms = (time.perf_counter()-START)*1000
    source = Path(request['qasm']).read_text()
    backend = 'cuda' if s['precision'] == 'f64' else 'cuda-f32'
    kwargs = dict(backend=backend, shots=s['shots'], seed=s['seed'], cuda_optimize=arm['optimizer'])
    # Import/API incompatibility or unavailable GPU must fail this subprocess;
    # no try/except changes those failures into successful smoke tests.
    stats_fn = _native._gpu_result_stats
    def stats():
        return dict(zip(('materializations', 'device_bytes', 'host_bytes', 'dtoh_bytes'), stats_fn()))
    for _ in range(s['warmup_calls']):
        warm = aleph.run(aleph.Circuit.from_qasm(source), **kwargs)
        warm.statevector()
        del warm
        gc.collect()
    reference = None
    if request['preflight']:
        # The exact timed workload has a full CPU-amplitude oracle before the
        # broad experiment. This cost is outside the reported API caller clock.
        reference = aleph.run(aleph.Circuit.from_qasm(source), backend='sv', shots=0).statevector()
    before = stats()
    started = time.perf_counter()
    t = time.perf_counter()
    circuit = aleph.Circuit.from_qasm(source)
    parse_ms = (time.perf_counter()-t)*1000
    t = time.perf_counter()
    result = aleph.run(circuit, **kwargs)
    run_ms = (time.perf_counter()-t)*1000
    after_run = stats()
    counts = result.counts()
    diagnostics = result.execution_stats()
    t = time.perf_counter()
    amplitudes = result.statevector()
    materialization_ms = (time.perf_counter()-t)*1000
    after_read = stats()
    del result
    gc.collect()
    after_drop = stats()
    caller_ms = (time.perf_counter()-started)*1000
    assert sum(counts.values()) == s['shots']
    assert diagnostics['optimized'] == float(arm['optimizer'])
    assert diagnostics['tf32_effective'] == 0 and diagnostics['tf32_launches_process_delta'] == 0
    assert after_run['materializations'] == before['materializations']
    assert after_run['dtoh_bytes'] - before['dtoh_bytes'] == 16 + s['shots']*8
    width = 16 if s['precision'] == 'f64' else 8
    assert after_run['device_bytes']-before['device_bytes'] == (1 << s['qubits'])*width
    assert after_read['dtoh_bytes']-after_run['dtoh_bytes'] == (1 << s['qubits'])*width
    assert after_read['materializations']-before['materializations'] == 1
    assert after_drop['device_bytes'] == before['device_bytes']
    assert after_drop['host_bytes'] == before['host_bytes']
    assert np.isfinite(amplitudes).all()
    assert abs(np.vdot(amplitudes, amplitudes).real-1) < s['error_budget']['norm_atol']
    if reference is not None:
        np.testing.assert_allclose(amplitudes, reference, atol=s['error_budget']['amplitude_atol'], rtol=0)
    counters = dict(diagnostics)
    counters.update({k+'_delta': after_read[k]-before[k] for k in before})
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    counters['peak_rss_bytes'] = peak if sys.platform == 'darwin' else peak*1024
    native = Path(_native.__file__)
    print(json.dumps(dict(protocol=1, status='executed', route=s['route'],
        hardware_executed=True, qubits=s['qubits'], shots_observed=sum(counts.values()),
        warmup_calls=s['warmup_calls'], caller_ms=caller_ms,
        oracle_executed=reference is not None,
        imported_artifact=dict(path=str(native), sha256=hashlib.sha256(native.read_bytes()).hexdigest()),
        counters=counters, stages=dict(startup_ms=startup_ms, parse_compile_ms=parse_ms,
            execution_ms=diagnostics['execution_ms'], sampling_ms=diagnostics['sampling_ms'],
            materialization_ms=materialization_ms, formatting_ms=None, run_api_ms=run_ms))))


if __name__ == '__main__':
    main()
