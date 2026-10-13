"""Real-GPU public result tests; opt in with ALEPH_TEST_GPU_BACKENDS.

Example: ALEPH_TEST_GPU_BACKENDS=cuda,cuda-f32 python -m unittest discover
-s scripts/python -p test_gpu_results.py -v. Explicit opt-in must execute,
so device/initialization failures fail rather than masquerade as smoke passes.
"""
import gc
import math
import os
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import aleph
from aleph import _native

BACKENDS = [b for b in os.environ.get("ALEPH_TEST_GPU_BACKENDS", "").split(",") if b]


@unittest.skipUnless(BACKENDS, "set ALEPH_TEST_GPU_BACKENDS to execute hardware tests")
class TestGpuResults(unittest.TestCase):
    def test_counts_lazy_cache_order_threads_and_cleanup(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                # Create on one Python worker; use after its backend/thread exits.
                circuit = aleph.Circuit(12)
                circuit.h(0)
                circuit.s(0)
                circuit.cx(0, 11)
                circuit.x(3)
                before = _native._gpu_result_stats()
                with ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(aleph.run, circuit, backend=backend,
                                         shots=16, seed=7).result()
                after_run = _native._gpu_result_stats()
                width = 16 if backend == "cuda" else 8
                self.assertEqual(after_run[0], before[0])
                self.assertEqual(after_run[1] - before[1], (1 << 12) * width)
                self.assertEqual(after_run[2], before[2])
                # CUDA scalar + sample readout; no state-sized transfer.
                self.assertLessEqual(after_run[3] - before[3], 16 * 8 + 256)
                self.assertEqual(sum(result.counts().values()), 16)
                self.assertEqual(set(result.counts()), {"000000001000", "100000001001"})
                with ThreadPoolExecutor(max_workers=4) as pool:
                    arrays = list(pool.map(lambda _: result.statevector(), range(4)))
                after_read = _native._gpu_result_stats()
                self.assertEqual(after_read[0] - before[0], 1)
                self.assertEqual(after_read[1], before[1])
                self.assertEqual(after_read[2] - before[2], (1 << 12) * 16)
                expected_dtoh = 0 if backend == "metal" else (1 << 12) * width
                self.assertEqual(after_read[3] - after_run[3], expected_dtoh)
                expected = np.zeros(1 << 12, dtype=np.complex128)
                expected[8] = 1 / math.sqrt(2)
                expected[2057] = 1j / math.sqrt(2)
                for array in arrays:
                    self.assertEqual(array.dtype, np.complex128)
                    np.testing.assert_allclose(array, expected, atol=1e-6)
                arrays[0][:] = 0
                np.testing.assert_allclose(result.statevector(), expected, atol=1e-6)
                self.assertEqual(_native._gpu_result_stats()[0], after_read[0])
                del result
                gc.collect()
                self.assertEqual(_native._gpu_result_stats()[1:3], before[1:3])
                np.testing.assert_allclose(arrays[1], expected, atol=1e-6)

    def test_counts_only_cleanup_and_collapsed_final_state(self):
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                before = _native._gpu_result_stats()
                circuit = aleph.Circuit(4)
                circuit.h(0)
                result = aleph.run(circuit, backend=backend, shots=0, seed=2)
                self.assertEqual(result.counts(), {})
                self.assertEqual(_native._gpu_result_stats()[0], before[0])
                del result
                gc.collect()
                self.assertEqual(_native._gpu_result_stats()[1:3], before[1:3])
                circuit = aleph.Circuit.from_qasm('''OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q; bit[1] c;
x q[1]; h q[0]; measure q[0] -> c[0];''')
                result = aleph.run(circuit, backend=backend, shots=32, seed=3)
                counts = result.counts()
                self.assertEqual(len(counts), 1)
                expected = np.zeros(4, dtype=np.complex128)
                expected[int(next(iter(counts)), 2)] = 1
                np.testing.assert_allclose(result.statevector(), expected, atol=1e-6)
                del result
                gc.collect()
                self.assertEqual(_native._gpu_result_stats()[1:3], before[1:3])
