"""Profile CUDA-Q plugin marshalling on the same split model and seeded shots.

Run baseline/candidate in separate processes with RAYON_NUM_THREADS=1 and identical
SDK versions. Timings exclude model construction and include all returned results.
"""
import argparse
import cProfile
import hashlib
import json
import platform
import statistics
import time

import numpy as np
import stim
import cudaq_qec as cq
import aleph
import aleph.qec as aq
import aleph.cudaq as ac


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shots', type=int, default=20_000)
    p.add_argument('--distance', type=int, default=5)
    p.add_argument('--repeats', type=int, default=7)
    p.add_argument('--profile')
    args = p.parse_args()
    if args.shots < 1 or args.repeats < 1:
        p.error('shots and repeats must be positive')
    circ = stim.Circuit.generated('surface_code:rotated_memory_x', distance=args.distance,
                                  rounds=args.distance, after_clifford_depolarization=.003)
    dets, obs = circ.compile_detector_sampler(seed=20261016).sample(
        args.shots, separate_observables=True)
    H, O, rates = ac.dem_to_matrices(circ.detector_error_model(decompose_errors=True))
    native = aq.Decoder(aq.dem_from_matrices(H, O, rates), 'mwpm')
    plugin = cq.get_decoder('aleph-mwpm', H, O=O, error_rate_vec=rates)
    floats = dets.astype(np.float64)
    rows = floats.tolist()
    reference, conv = native.decode_batch_errors(dets)
    results = {}

    def check_plugin(r):
        np.testing.assert_array_equal(np.asarray(r.result) > .5, reference)
        np.testing.assert_array_equal(r.converged, conv)
        return digest(np.asarray(r.result) > .5, np.asarray(r.converged))

    calls = {
        'native_errors': lambda: native.decode_batch_errors(dets),
        'plugin_list': lambda: plugin.decode_batch(rows),
        'plugin_numpy': lambda: plugin.decode_batch(floats),
        'plugin_bits': lambda: plugin.decode_batch(dets),
        'plugin_end_to_end_list': lambda: plugin.decode_batch(dets.astype(np.float64).tolist()),
        'plugin_end_to_end_numpy': lambda: plugin.decode_batch(dets.astype(np.float64)),
        'to_list': lambda: dets.astype(np.float64).tolist(),
        'to_bits_numpy': lambda: ac._to_bits(floats, H.shape[0]),
        'to_bits_list': lambda: ac._to_bits(rows, H.shape[0]),
        'result_conversion': lambda: reference.astype(np.float64),
        'sdk_result': lambda: cq.BatchDecoderResult(result=reference.astype(np.float64),
                                                   converged=conv, opt_results=None,
                                                   batch_opt_results=None),
    }
    for name, call in calls.items():
        r = call()
        signature = check_plugin(r) if name.startswith('plugin_') or name == 'sdk_result' else None
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            r = call()
            times.append(time.perf_counter() - start)
        results[name] = dict(seconds=times, median_seconds=statistics.median(times),
                             shots_per_second=args.shots / statistics.median(times),
                             signature=signature)
    if args.profile:
        profiler = cProfile.Profile()
        profiler.runcall(calls['plugin_list'])
        profiler.dump_stats(args.profile)
    pred = (reference @ O.T) % 2
    print(json.dumps(dict(python=platform.python_version(), numpy=np.__version__,
                          stim=stim.__version__, cudaq_qec=getattr(cq, '__version__', None),
                          native_module=aleph._native.__file__, shots=args.shots,
                          distance=args.distance, detectors=H.shape[0], errors=H.shape[1],
                          input_hash=digest(dets, obs), error_hash=digest(reference, conv),
                          logical_mismatches=int(np.any(pred != obs, axis=1).sum()),
                          nonconverged=int((~conv).sum()), results=results), indent=2))


if __name__ == '__main__':
    main()
